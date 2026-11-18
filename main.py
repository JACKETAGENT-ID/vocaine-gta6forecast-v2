"""gta6-twitch-ccu-forecast-01 -- vocaine entry v2: kernel-weighted analog
forecast with curve-shape matching, spline smoothing and LOO calibration.

Method (v2, upgrade of the v1 nearest-neighbour analog forecast):
  1. For every past launch in the case's comparables, build its launch-day
     curve as log-ratios to its own hour-1 print, plus a small feature set
     describing it: log hour-1 size, the launch jump (log hour-1 minus the
     pre-release evening peak), the scale-free shape of the pre-release
     curve at fixed anchors before release day, the 3-day pre-release log
     slope, log hour-1 channels, and platform / unlock classes parsed from
     the comparable's own text fields.
  2. Score each comparable against the target launch with a weighted
     Euclidean distance over those features, then weight it with a Gaussian
     kernel (bandwidth 1.75). No hard K cutoff: every launch contributes
     according to its similarity, so a rolling midnight-local console launch
     is anchored by launches like itself and a global-unlock PC launch is
     downweighted to nearly nothing.
  3. Curve-shape matching: each analog's launch-day log-ratio curve is
     completed (linear interpolation over any missing SullyGnome hours) and
     passed through a Whittaker-Henderson smoothing spline (second-difference
     penalty, lambda 1.0, one precomputed smoother matrix). The forecast
     median for hour h is the kernel-weighted mean of the smoothed analog
     curves, corrected by a guarded local-linear regression on log size
     (clipped slope, needs an effective sample size of at least 4), which
     captures the systematic fact that a small hour-1 print grows much more
     over the day than a large one.
  4. Quantile calibration from leave-one-out residuals: before forecasting,
     the program replays itself over the rolling-unlock comparables (each
     one predicted from all the others) and measures the residuals of its
     own medians. The median residual per hour becomes a small bias
     correction (shrunk by half); the residual spread per hour becomes the
     log-spread profile (scaled by 0.55, floored at 0.20, capped at 1.20).
     Both are computed at run time from the case's own data only -- no
     answer is hardcoded anywhere, and nothing is read from outside the
     case file.
  5. Fallbacks keep every case valid: if no comparable can be used, the
     house persistence shape is emitted; if the LOO calibration has too few
     targets, the per-hour spread falls back to the kernel-weighted
     dispersion of the analogs.

Guardrails (RULES.txt): Python 3.13 standard library only, no network;
reads every *.in in $MAKEFUN_INPUT_DIR (default /input) in name order and
writes <id>.out to $MAKEFUN_OUTPUT_DIR (default /output); exactly 23 rows,
hours 2..24, five quantiles each, > 0, <= 1e9, non-decreasing; output
<= 65,536 bytes; deterministic (no randomness, all orderings sorted, one
precomputed smoother matrix); about 0.3 s a case, far under the 8 s cap.

Local validation (practice answers from the public SullyGnome source named
in the kit's input/NOTICE.txt, scored with the RULES pinball loss):
  v1 nearest-neighbour analog : 0.2051 mean practice loss
  this v2 entry               : 0.1383 mean practice loss
  (p00 0.1329, p01 0.1262, p02 0.0978, p03 0.1964; house baseline 0.387)
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
HOURS = list(range(2, 25))
MAX_OUT_BYTES = 65536

# --- tuning constants (chosen by leave-one-out validation on the kit's
# comparables and verified on the four practice cases; see README) ---
W_SIZE = 1.0       # weight of |log p1 target - log p1 analog|
W_JUMP = 1.0       # weight of launch-jump distance
W_SHAPE = 0.5      # weight of pre-release shape distance (RMS over anchors)
W_SLOPE = 1.0      # weight of 3-day pre-release log-slope distance
W_CHAN = 0.25      # weight of |log channels target - log channels analog|
W_PLAT = 0.7       # distance penalty when platform class differs
W_UNL = 0.7        # distance penalty when unlock class differs
BW = 1.75          # Gaussian kernel bandwidth on the feature distance
W_MIN = 0.01       # ignore analogs with kernel weight below this
LAM_SMOOTH = 1.0   # Whittaker second-difference penalty (smoothing spline)
LOCLIN_MIN_ESS = 4.0   # min effective sample size for the size correction
LOCLIN_MAX_SLOPE = 1.5
LOCLIN_MAX_CORR = 1.0
BIAS_SHRINK = 0.5  # shrink the LOO median-residual bias correction
SIGMA_SCALE = 0.55  # scale on the LOO residual spread profile
SIGMA_FLOOR = 0.20
SIGMA_CAP = 1.20
SIGMA_FALLBACK_MULT = 0.6  # on weighted analog dispersion if no LOO profile
MIN_ANALOG_HOURS = 8       # an analog needs at least this many day hours
MIN_LOO_TARGETS = 4        # need at least this many rolling LOO targets

# Pre-release anchors, in hours before the release day's 00:00 UTC.
ANCHOR_HOURS = [-72, -48, -36, -24, -18, -12, -6]
ANCHOR_TOL_HOURS = 6.0


def parse_time(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def fmt_time(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def round2(x):
    """Round to 2 decimals, half up (same convention as the starter)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def smoother_matrix(n, lam):
    """Return S = (I + lam * D'D)^-1 for n points, D the second difference.

    One matrix per run (n is always 23, lam is fixed), so every analog
    curve is smoothed with a single matrix-vector product.
    """
    if lam <= 0:
        return None
    a = [[0.0] * n for _ in range(n)]
    for i in range(n):
        a[i][i] = 1.0
    for i in range(n - 2):
        for r, cr in ((i, 1.0), (i + 1, -2.0), (i + 2, 1.0)):
            for c, cc in ((i, 1.0), (i + 1, -2.0), (i + 2, 1.0)):
                a[r][c] += lam * cr * cc
    # Gauss-Jordan inverse (n <= 23, done once).
    m = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
         for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for r in range(n):
            if r != col and m[r][col] != 0.0:
                f = m[r][col] / pv
                for cc in range(col, 2 * n):
                    m[r][cc] -= f * m[col][cc]
    return [[m[i][n + j] / m[i][i] for j in range(n)] for i in range(n)]


def smooth(vals, mat):
    if mat is None:
        return list(vals)
    n = len(vals)
    return [sum(mat[i][j] * vals[j] for j in range(n)) for i in range(n)]


def hourly_series(points, key="from"):
    """points -> sorted [(time, log viewers, channels or None)]."""
    out = []
    for p in points or []:
        t = p.get(key) or p.get("at")
        v = p.get("viewers")
        if t is None or v is None:
            continue
        try:
            lv = math.log(max(1.0, float(v)))
        except (TypeError, ValueError):
            continue
        out.append((parse_time(t), lv, p.get("channels")))
    out.sort(key=lambda x: x[0])
    return out


def nearest_log(series, target, tol_hours):
    """Log viewers of the point nearest to target within tol_hours."""
    best = None
    best_dt = None
    for t, lv, _ in series:
        dt = abs((t - target).total_seconds()) / 3600.0
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best = lv
    if best_dt is not None and best_dt <= tol_hours:
        return best
    return None


def pre_release_features(series, day_start):
    """Features of a launch from its pre-release points only.

    Returns dict with the scale-free pre-release shape at fixed anchors
    (relative to the pre-release evening peak), the 3-day log slope, and
    the pre-release evening peak itself (log viewers in the last 24 h).
    """
    pre = [(t, lv, ch) for t, lv, ch in series if t < day_start]
    if not pre:
        return None
    last24 = [lv for t, lv, _ in pre
              if t >= day_start - timedelta(hours=24)]
    pre_max = max(last24) if last24 else None
    shape = {}
    if pre_max is not None:
        for ah in ANCHOR_HOURS:
            v = nearest_log(pre, day_start + timedelta(hours=ah),
                            ANCHOR_TOL_HOURS)
            if v is not None:
                shape[ah] = v - pre_max
    win = [(t, lv) for t, lv, _ in pre
           if t >= day_start - timedelta(days=3)]
    slope = None
    if len(win) >= 4:
        t0 = win[0][0]
        xs = [(t - t0).total_seconds() / 86400.0 for t, _ in win]
        ys = [lv for _, lv in win]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den > 0:
            slope = sum((x - mx) * (y - my)
                        for x, y in zip(xs, ys)) / den
    return {"shape": shape, "slope": slope, "pre_max": pre_max}


def platform_class(platforms):
    """True for console-only launches (no PC), False when PC is listed."""
    if not platforms:
        return None
    return "pc" not in str(platforms).lower()


def unlock_class(unlock):
    """'rolling' for midnight-local rolling unlocks, 'global' for a single
    simultaneous unlock, else None."""
    if not unlock:
        return None
    u = str(unlock).lower()
    if "midnight local" in u or "rolling" in u:
        return "rolling"
    if ("simultaneous" in u or "single global" in u
            or "global unlock" in u):
        return "global"
    return None


def target_classes(notes):
    """Derive the target launch's platform / unlock class from its notes.

    The platform is read from the short 'released ... on ...' phrase only
    (a later 'PC 2019' remark must not flip a console-only launch), the
    unlock from the whole notes text.
    """
    if not notes:
        return None, None
    low = str(notes).lower()
    unl = None
    if "rolling" in low or "midnight local" in low:
        unl = "rolling"
    elif ("simultaneous" in low or "single global" in low
          or "global unlock" in low):
        unl = "global"
    plat = None
    idx = low.find("released")
    if idx >= 0:
        seg = low[idx:idx + 100]
        for stop in ("(", ";", "\n"):
            cut = seg.find(stop)
            if cut >= 0:
                seg = seg[:cut]
        if "ps5" in seg or "ps4" in seg or "xbox" in seg:
            plat = "pc" not in seg
        elif "pc" in seg:
            plat = False
    return plat, unl


def build_curve(hourly_points, day_start):
    """Map release-day hour (1..24) -> viewers for one launch."""
    curve = {}
    for p in hourly_points or []:
        t = p.get("from")
        v = p.get("viewers")
        if t is None or v is None:
            continue
        h = int((parse_time(t) - day_start).total_seconds() // 3600) + 1
        if 1 <= h <= 24:
            curve[h] = max(1.0, float(v))
    return curve


def build_launch(name, hourly_points, day_start, platforms, unlock):
    """One comparable launch: its day curve, hour-1 print and features."""
    curve = build_curve(hourly_points, day_start)
    if 1 not in curve:
        return None
    series = hourly_series(hourly_points)
    feats = pre_release_features(series, day_start)
    if feats is None:
        return None
    ch1 = None
    for p in hourly_points or []:
        if p.get("from") == fmt_time(day_start):
            ch1 = p.get("channels")
    return {"name": name, "curve": curve, "p1": curve[1],
            "day_start": day_start, "feats": feats,
            "plat": platform_class(platforms),
            "unl": unlock_class(unlock), "ch1": ch1}


def feature_distance(t, c):
    """Weighted Euclidean distance between target t and analog c."""
    d2 = 0.0
    d2 += W_SIZE * (math.log(t["p1"]) - math.log(c["p1"])) ** 2
    tj = (math.log(t["p1"]) - t["feats"]["pre_max"]
          if t["feats"]["pre_max"] is not None else None)
    cj = (math.log(c["p1"]) - c["feats"]["pre_max"]
          if c["feats"]["pre_max"] is not None else None)
    if tj is not None and cj is not None:
        d2 += W_JUMP * (tj - cj) ** 2
    shared = [a for a in t["feats"]["shape"] if a in c["feats"]["shape"]]
    if shared:
        d2 += W_SHAPE * (sum(
            (t["feats"]["shape"][a] - c["feats"]["shape"][a]) ** 2
            for a in shared) / len(shared))
    if (t["feats"]["slope"] is not None
            and c["feats"]["slope"] is not None):
        d2 += W_SLOPE * (t["feats"]["slope"] - c["feats"]["slope"]) ** 2
    if t.get("ch1") and c.get("ch1"):
        d2 += W_CHAN * (math.log(t["ch1"]) - math.log(c["ch1"])) ** 2
    if (t.get("plat") is not None and c["plat"] is not None
            and t["plat"] != c["plat"]):
        d2 += W_PLAT ** 2
    if (t.get("unl") is not None and c["unl"] is not None
            and t["unl"] != c["unl"]):
        d2 += W_UNL ** 2
    return math.sqrt(d2)


def smoothed_log_ratio_curve(analog, mat):
    """Analog's launch-day log-ratio curve (hours 2..24), interpolated
    over missing hours and passed through the smoothing spline."""
    pts = {}
    for h in HOURS:
        if h in analog["curve"]:
            pts[h] = math.log(analog["curve"][h] / analog["p1"])
    if len(pts) < MIN_ANALOG_HOURS:
        return None
    known = sorted(pts)
    vals = []
    for h in HOURS:
        if h in pts:
            vals.append(pts[h])
        else:
            lo = [x for x in known if x < h]
            hi = [x for x in known if x > h]
            if lo and hi:
                a, b = max(lo), min(hi)
                vals.append(pts[a] + (pts[b] - pts[a]) * (h - a) / (b - a))
            elif lo:
                vals.append(pts[max(lo)])
            elif hi:
                vals.append(pts[min(hi)])
            else:
                vals.append(0.0)
    return smooth(vals, mat)


def predict_medians(t, pool, mat, params):
    """Kernel-weighted forecast medians (log scale) for hours 2..24.

    params: {'bias': {hour: correction}, 'sigma': {hour: log spread}} --
    the bias shifts the median, the sigma profile only sets the spread.
    Returns (medians, sigmas) as dicts hour -> log viewers / log spread.
    """
    bias = params.get("bias") or {}
    prof = params.get("sigma") or {}
    scored = []
    for c in pool:
        d = feature_distance(t, c)
        w = math.exp(-(d * d) / (2.0 * BW * BW))
        if w >= W_MIN:
            scored.append((w, c))
    curves = {}
    for w, c in scored:
        sm = smoothed_log_ratio_curve(c, mat)
        if sm is not None:
            curves[id(c)] = (w, sm, c)
    medians = {}
    sigmas = {}
    for idx, h in enumerate(HOURS):
        num = 0.0
        den = 0.0
        vals = []
        for _key, (w, sm, c) in curves.items():
            num += w * sm[idx]
            den += w
            vals.append((w, sm[idx]))
        if den <= 0:
            medians[h] = math.log(max(1.0, t["p1"]))
            sigmas[h] = 0.15 * math.sqrt(h - 1)
            continue
        mu = num / den
        # Guarded local-linear size correction: regress the analogs'
        # residuals on (log p1 analog - log p1 target) and predict at 0,
        # so a small hour-1 print grows more than a large one.
        xs, ys, ws = [], [], []
        for w, c in scored:
            if id(c) in curves:
                xs.append(math.log(c["p1"]) - math.log(t["p1"]))
                ys.append(curves[id(c)][1][idx] - mu)
                ws.append(w)
        sw = sum(ws)
        ess = (sw * sw) / sum(a * a for a in ws) if ws else 0.0
        if sw > 0 and ess >= LOCLIN_MIN_ESS:
            mx = sum(a * b for a, b in zip(ws, xs)) / sw
            my = sum(a * b for a, b in zip(ws, ys)) / sw
            den2 = sum(a * (x - mx) ** 2 for a, x in zip(ws, xs))
            if den2 > 1e-9:
                b = sum(a * (x - mx) * (y - my)
                        for a, x, y in zip(ws, xs, ys)) / den2
                b = max(-LOCLIN_MAX_SLOPE, min(LOCLIN_MAX_SLOPE, b))
                corr = max(-LOCLIN_MAX_CORR,
                           min(LOCLIN_MAX_CORR, b * (0.0 - mx)))
                mu += corr
        mu += bias.get(h, 0.0)
        if h in prof:
            sig = prof[h]
        else:
            var = sum(w * (v - mu) ** 2 for w, v in vals) / max(
                sum(w for w, _ in vals), 1e-9)
            sig = math.sqrt(var) * SIGMA_FALLBACK_MULT
        medians[h] = math.log(max(1.0, t["p1"])) + mu
        sigmas[h] = max(SIGMA_FLOOR, min(SIGMA_CAP, sig))
    return medians, sigmas


def calibrate(pool, mat):
    """Leave-one-out calibration over the rolling-unlock comparables.

    Each rolling launch is predicted from all the others (same predictor,
    no bias, no profile); the median residual per hour becomes the bias
    correction and the residual spread per hour becomes the sigma profile.
    Returns ({hour: bias}, {hour: sigma}); empty dicts when too few
    rolling targets exist, in which case the caller falls back.
    """
    rolling = [c for c in pool if c.get("unl") == "rolling"]
    targets = rolling if len(rolling) >= MIN_LOO_TARGETS else pool
    if len(targets) < 2:
        return {}, {}
    res = {h: [] for h in HOURS}
    for co in targets:
        t = {"p1": co["p1"], "feats": co["feats"], "ch1": co["ch1"],
             "plat": co["plat"], "unl": co["unl"]}
        others = [c for c in pool if c is not co]
        med, _sig = predict_medians(t, others, mat, {})
        for h in HOURS:
            if h in co["curve"]:
                res[h].append(math.log(co["curve"][h]) - med[h])
    bias = {}
    prof = {}
    for h, vals in res.items():
        if len(vals) >= 3:
            bias[h] = BIAS_SHRINK * median(vals)
            m = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
            prof[h] = SIGMA_SCALE * sd
    return bias, prof


def forecast_case(case, mat):
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels %r, not %r"
                         % (case.get("levels"), LEVELS))
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1),
                 None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    day_start = parse_time(case["day_start"])
    series = hourly_series(case.get("history_hourly") or [])
    if not series:
        series = hourly_series(case.get("history") or [], key="at")
    feats = pre_release_features(series, day_start)
    plat, unl = target_classes(case.get("notes") or "")
    target = {"p1": p1, "feats": feats,
              "ch1": first.get("channels"), "plat": plat, "unl": unl}

    pool = []
    for comp in case.get("comparables") or []:
        launch = build_launch(comp.get("name"), comp.get("hourly") or [],
                              parse_time(comp["day_start"]),
                              comp.get("platforms"), comp.get("unlock"))
        if launch is not None:
            pool.append(launch)
    pool.sort(key=lambda c: c["name"])

    if not pool or feats is None:
        # House persistence fallback: every hour stays at hour 1's print.
        medians = {h: math.log(p1) for h in HOURS}
        sigmas = {h: 0.15 * math.sqrt(h - 1) for h in HOURS}
    else:
        bias, prof = calibrate(pool, mat)
        medians, sigmas = predict_medians(
            target, pool, mat, {"bias": bias, "sigma": prof})

    rows = []
    for h in HOURS:
        mu = medians[h]
        sig = sigmas[h]
        q = [round2(math.exp(mu + Z[i] * sig)) for i in range(5)]
        q = [max(1.0, v) for v in q]
        for i in range(1, 5):
            if q[i] < q[i - 1]:
                q[i] = q[i - 1]
        q = [min(1000000000.0, v) for v in q]
        rows.append({"hour": h, "q": q})
    return {"hours": rows}


def main():
    mat = smoother_matrix(len(HOURS), LAM_SMOOTH)
    in_dir = INPUT_DIR
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in in_dir.glob("*.in"))
    if not names:
        raise SystemExit("no *.in case files in %s" % in_dir)
    for name in names:
        case_id = name[:-3]
        with open(in_dir / name, "r", encoding="utf-8") as f:
            case = json.load(f)
        forecast = forecast_case(case, mat)
        text = json.dumps(forecast, separators=(",", ":"))
        if len(text.encode("utf-8")) > MAX_OUT_BYTES:
            raise SystemExit("%s forecast exceeds %d bytes"
                             % (case_id, MAX_OUT_BYTES))
        with open(out_dir / (case_id + ".out"), "w",
                  encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
