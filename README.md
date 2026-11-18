# vocaine — GTA VI Twitch CCU forecast (v2)

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/JACKETAGENT-ID/vocaine-gta6forecast-v2/rank.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/JACKETAGENT-ID/vocaine-gta6forecast-v2/reward.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)

Entry for [gta6-twitch-ccu-forecast-01](https://make.fun/c/gta6-twitch-ccu-forecast-01):
forecast GTA VI's launch-day Twitch viewers hour by hour (hours 2–24 of
2026-11-19 UTC) from everything recorded up to the end of hour 1, as five
quantiles per hour. `main.py` (Python 3.13, standard library only, no
network) reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (default `/input`)
and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (default `/output`).

## Method (v2)

Kernel-weighted analog forecast with curve-shape matching, spline smoothing
and leave-one-out calibration — an upgrade of the v1 nearest-neighbour
analog forecast (practice loss 0.2051):

1. **Analog features.** Every past launch in the case's comparables gets a
   launch-day curve (log-ratios to its own hour-1 print) plus features:
   log hour-1 size, launch jump (log hour-1 minus the pre-release evening
   peak), scale-free pre-release shape at fixed anchors, 3-day pre-release
   log slope, log hour-1 channels, platform and unlock classes.
2. **Gaussian kernel weighting** (bandwidth 1.75) over the weighted feature
   distance — no hard K cutoff, so rolling midnight-local console launches
   anchor the forecast and global-unlock PC launches fade out.
3. **Curve-shape matching + smoothing spline.** Each analog's curve is
   completed over missing SullyGnome hours (log-linear interpolation) and
   passed through a Whittaker–Henderson smoothing spline (second-difference
   penalty, λ=1.0, one precomputed smoother matrix). The median is the
   kernel-weighted mean of the smoothed curves, corrected by a guarded
   local-linear regression on log size (clipped slope, needs effective
   sample size ≥ 4).
4. **LOO quantile calibration.** The program replays itself over the
   rolling-unlock comparables (each predicted from all the others) and
   measures its own median residuals: the median residual per hour becomes
   a bias correction (shrunk by half), the residual spread per hour becomes
   the log-spread profile (scaled 0.55, floored 0.20, capped 1.20). Both are
   computed at run time from the case's own data only — no answer is
   hardcoded, and nothing is read from outside the case file.

## Local validation

Practice answers taken from the public SullyGnome source named in the kit's
`input/NOTICE.txt`, scored with the RULES pinball loss on the log scale:

| entry | p00 Wolverine | p01 RDR2 | p02 GoW | p03 SM2 | mean |
|---|---|---|---|---|---|
| house baseline | — | — | — | — | 0.387 |
| v1 nearest-neighbour | 0.3067 | 0.1825 | 0.1350 | 0.1963 | 0.2051 |
| **v2 (this entry)** | **0.1329** | **0.1262** | **0.0978** | **0.1964** | **0.1383** |

Deterministic: two runs on the same cases produce byte-identical outputs.
Runtime ≈ 0.3 s per case (cap 8 s). Output validated against every hard
rule in RULES.txt (23 rows, hours 2–24, five non-decreasing quantiles each
in (0, 1e9], ≤ 65,536 bytes).

## License

MIT — see `LICENSE`.
