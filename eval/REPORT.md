# PhysSplat evaluation report

_11 records; regenerated 2026-09-08 04:07_

## Leaderboard (composite 0-100)

| # | id | step | composite | stability | trans_150 | surface_pen | support_jaccard | change |
|---|---|---|---|---|---|---|---|---|
| 1 | exp1_rest_k24 | 150000 | 57.3 | 0.67 | 2.3cm | 1.91mm | 0.60 | rest_k24: {'K': 24, 'lr': 3e-06, 'rest_only': True} |
| 2 | exp2_rollout_k12 | 150000 | 49.5 | 0.46 | 2.4cm | 1.96mm | 0.50 | rollout_k12: {'K': 12, 'lr': 3e-06} |
| 3 | exp1_rollout_k8_long | 150000 | 47.2 | 0.46 | 2.3cm | 2.46mm | 0.50 | rollout_k8_long: {'K': 8, 'lr': 5e-06, 'steps': 6000} |
| 4 | exp5_rollout_k8 | 150000 | 44.1 | 0.42 | 3.9cm | 2.27mm | 0.60 | rollout_k8: {'K': 8, 'lr': 5e-06} |
| 5 | exp1_rollout_k4 | 150000 | 36.6 | 0.33 | 3.6cm | 3.46mm | 0.50 | rollout_k4: {'K': 4, 'lr': 1e-05} |
| 6 | exp3_ang_w0.5 | 153000 | 34.0 | 0.29 | 3.7cm | 2.37mm | 0.30 | ang_w0.5: {'ang_weight': 0.5} |
| 7 | exp4_polish_lr1e-5 | 153000 | 31.9 | 0.29 | 4.5cm | 2.63mm | 0.23 | polish_lr1e-5: {'lr': 1e-05, 'lr_final': 1e-05} |
| 8 | run01@150000 | 150000 | 31.5 | 0.17 | 4.2cm | 2.68mm | 0.42 | main run, 150k review |
| 9 | base@150000 | 150000 | 31.5 | 0.17 | 4.2cm | 2.68mm | 0.42 | baseline for improve loop |
| 10 | exp2_noise_5e-4 | 153000 | 24.0 | 0.00 | 6.2cm | 2.28mm | 0.24 | noise_5e-4: {'noise_std': 0.0005} |
| 11 | exp1_noise_2e-3 | 153000 | 22.5 | 0.12 | 19.1cm | 3.21mm | 0.33 | noise_2e-3: {'noise_std': 0.002} |

## Trend (evaluation order)

| time | id | step | composite | stability | trans_150 | axis_150 | surface_pen | support_jaccard | post_action_err | explosion | photo_drift |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-08 01:14 | run01@150000 | 150000 | 31.494 | 0.167 | 0.042 | 0.269 | 0.003 | 0.417 | 0.800 | 0.292 | 0.021 |
| 2026-09-08 01:18 | base@150000 | 150000 | 31.469 | 0.167 | 0.042 | 0.269 | 0.003 | 0.417 | 0.799 | 0.292 | 0.021 |
| 2026-09-08 01:28 | exp1_noise_2e-3 | 153000 | 22.535 | 0.125 | 0.191 | 0.396 | 0.003 | 0.325 | 1.069 | 0.292 | 0.010 |
| 2026-09-08 01:43 | exp1_rollout_k4 | 150000 | 36.552 | 0.333 | 0.036 | 0.260 | 0.003 | 0.500 | 0.575 | 0.292 | 0.012 |
| 2026-09-08 01:51 | exp2_noise_5e-4 | 153000 | 23.964 | 0.000 | 0.062 | 0.243 | 0.002 | 0.239 | 0.704 | 0.292 | 0.072 |
| 2026-09-08 01:59 | exp3_ang_w0.5 | 153000 | 33.951 | 0.292 | 0.037 | 0.335 | 0.002 | 0.300 | 0.386 | 0.292 | 0.025 |
| 2026-09-08 02:07 | exp4_polish_lr1e-5 | 153000 | 31.867 | 0.292 | 0.045 | 0.247 | 0.003 | 0.233 | 0.591 | 0.250 | 0.022 |
| 2026-09-08 02:25 | exp5_rollout_k8 | 150000 | 44.144 | 0.417 | 0.039 | 0.231 | 0.002 | 0.600 | 0.541 | 0.208 | 0.009 |
| 2026-09-08 02:57 | exp1_rollout_k8_long | 150000 | 47.184 | 0.458 | 0.023 | 0.253 | 0.002 | 0.500 | 0.302 | 0.250 | 0.011 |
| 2026-09-08 03:21 | exp2_rollout_k12 | 150000 | 49.461 | 0.458 | 0.024 | 0.201 | 0.002 | 0.500 | 0.279 | 0.250 | 0.008 |
| 2026-09-08 04:07 | exp1_rest_k24 | 150000 | 57.281 | 0.667 | 0.023 | 0.187 | 0.002 | 0.600 | 0.229 | 0.208 | 0.008 |

## Experiment decisions

- **exp1_noise_2e-3** (noise_2e-3: {'noise_std': 0.002}) from base@150000: **REJECT** -- composite 31.5 -> 22.5 (< margin 0.5)  
  deltas: composite -8.934, stability -0.042, trans_150 +0.149, surface_pen +0.001, support_jaccard -0.092
- **exp1_rollout_k4** (rollout_k4: {'K': 4, 'lr': 1e-05}) from base@150000: **ACCEPT** -- composite 31.5 -> 36.6 (accepted)  
  deltas: composite +5.083, stability +0.167, trans_150 -0.006, surface_pen +0.001, support_jaccard +0.083
- **exp2_noise_5e-4** (noise_5e-4: {'noise_std': 0.0005}) from exp1_rollout_k4: **REJECT** -- composite 36.6 -> 24.0 (< margin 0.5)  
  deltas: composite -12.588, stability -0.333, trans_150 +0.026, surface_pen -0.001, support_jaccard -0.261
- **exp3_ang_w0.5** (ang_w0.5: {'ang_weight': 0.5}) from exp1_rollout_k4: **REJECT** -- composite 36.6 -> 34.0 (< margin 0.5)  
  deltas: composite -2.601, stability -0.042, trans_150 +0.001, surface_pen -0.001, support_jaccard -0.200
- **exp4_polish_lr1e-5** (polish_lr1e-5: {'lr': 1e-05, 'lr_final': 1e-05}) from exp1_rollout_k4: **REJECT** -- composite 36.6 -> 31.9 (< margin 0.5)  
  deltas: composite -4.685, stability -0.042, trans_150 +0.009, surface_pen -0.001, support_jaccard -0.267
- **exp5_rollout_k8** (rollout_k8: {'K': 8, 'lr': 5e-06}) from exp1_rollout_k4: **ACCEPT** -- composite 36.6 -> 44.1 (accepted)  
  deltas: composite +7.592, stability +0.083, trans_150 +0.003, surface_pen -0.001, support_jaccard +0.100
- **exp1_rollout_k8_long** (rollout_k8_long: {'K': 8, 'lr': 5e-06, 'steps': 6000}) from exp5_rollout_k8: **ACCEPT** -- composite 44.1 -> 47.2 (accepted)  
  deltas: composite +3.041, stability +0.042, trans_150 -0.017, surface_pen +0.000, support_jaccard -0.100
- **exp2_rollout_k12** (rollout_k12: {'K': 12, 'lr': 3e-06}) from exp1_rollout_k8_long: **ACCEPT** -- composite 47.2 -> 49.5 (accepted)  
  deltas: composite +2.277, stability +0.000, trans_150 +0.001, surface_pen -0.001, support_jaccard +0.000
- **exp1_rest_k24** (rest_k24: {'K': 24, 'lr': 3e-06, 'rest_only': True}) from exp2_rollout_k12: **ACCEPT** -- composite 49.5 -> 57.3 (accepted)  
  deltas: composite +7.820, stability +0.208, trans_150 -0.001, surface_pen -0.000, support_jaccard +0.100

## Per-regime breakdown of leader `exp1_rest_k24`

| regime | n | trans_150 | stability | surface_pen |
|---|---|---|---|---|
| bundle | 3 | 0.9cm | 1.00 | 6.45mm |
| crosshatch | 2 | 4.0cm | 1.00 | 5.95mm |
| drop | 3 | 8.8cm | 0.00 | 0.00mm |
| pile | 1 | 10.3cm | 0.00 | 0.00mm |
| pyramid | 6 | 0.2cm | 0.83 | 2.44mm |
| scattered | 5 | 0.8cm | 0.60 | 0.00mm |
| stack | 4 | 0.6cm | 0.75 | 0.00mm |
