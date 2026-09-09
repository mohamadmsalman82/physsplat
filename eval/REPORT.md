# PhysSplat evaluation report

_22 records; regenerated 2026-09-09 02:20_

## Leaderboard (composite 0-100)

| # | id | step | composite | stability | trans_150 | surface_pen | support_jaccard | change |
|---|---|---|---|---|---|---|---|---|
| 1 | final_v2_ff_fade | 150000 | 63.6 | 0.83 | 1.9cm | 1.50mm | 0.60 | angular contact fade: torque vanishes as a body separates (KEPT) |
| 2 | final_v2_freeflight | 150000 | 62.3 | 0.83 | 2.1cm | 1.73mm | 0.60 | free-flight rule: zero residual for bodies touching nothing (analytic) |
| 3 | exp3_rest_k24_long | 150000 | 61.0 | 0.83 | 2.2cm | 1.57mm | 0.50 | rest_k24_long: {'K': 24, 'lr': 3e-06, 'rest_only': True, 'steps': 6000} |
| 4 | exp1_rest_k24 | 150000 | 58.7 | 0.71 | 2.1cm | 2.18mm | 0.60 | rest_k24: {'K': 24, 'lr': 3e-06, 'rest_only': True} |
| 5 | exp2_mixed_k12 | 150000 | 58.5 | 0.75 | 2.1cm | 2.43mm | 0.60 | mixed_k12: {'K': 12, 'lr': 3e-06} |
| 6 | exp1_mixed_k12 | 150000 | 58.3 | 0.79 | 2.1cm | 2.47mm | 0.60 | mixed_k12: {'K': 12, 'lr': 3e-06} |
| 7 | exp1_rest_k24 | 150000 | 57.3 | 0.67 | 2.3cm | 1.91mm | 0.60 | rest_k24: {'K': 24, 'lr': 3e-06, 'rest_only': True} |
| 8 | exp1_rest_k24_long | 150000 | 56.7 | 0.75 | 2.6cm | 1.60mm | 0.40 | rest_k24_long: K=24 rest-only, 6000 iters |
| 9 | exp1_mixed_k12 | 150000 | 54.1 | 0.71 | 2.5cm | 2.29mm | 0.50 | mixed_k12: {'K': 12, 'lr': 3e-06} |
| 10 | exp2_rollout_k12 | 150000 | 49.5 | 0.46 | 2.4cm | 1.96mm | 0.50 | rollout_k12: {'K': 12, 'lr': 3e-06} |
| 11 | final_v2_ff_energy_q | 150000 | 48.1 | 0.42 | 2.3cm | 2.00mm | 0.40 | energy rule restricted to quiescent bodies (REJECTED: still biases resting contact down) |
| 12 | exp1_rollout_k8_long | 150000 | 47.2 | 0.46 | 2.3cm | 2.46mm | 0.50 | rollout_k8_long: {'K': 8, 'lr': 5e-06, 'steps': 6000} |

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
| 2026-09-08 06:33 | exp1_rest_k24_long | 150000 | 56.658 | 0.750 | 0.026 | 0.176 | 0.002 | 0.400 | 0.297 | 0.208 | 0.008 |
| 2026-09-08 06:59 | exp1_mixed_k12 | 150000 | 54.093 | 0.708 | 0.025 | 0.218 | 0.002 | 0.500 | 0.288 | 0.250 | 0.008 |
| 2026-09-08 07:25 | exp1_mixed_k12 | 150000 | 58.307 | 0.792 | 0.021 | 0.208 | 0.002 | 0.600 | 0.273 | 0.250 | 0.008 |
| 2026-09-08 08:23 | exp1_rest_k24 | 150000 | 58.730 | 0.708 | 0.021 | 0.170 | 0.002 | 0.600 | 0.214 | 0.167 | 0.010 |
| 2026-09-08 08:49 | exp2_mixed_k12 | 150000 | 58.451 | 0.750 | 0.021 | 0.175 | 0.002 | 0.600 | 0.264 | 0.208 | 0.010 |
| 2026-09-08 10:19 | exp3_rest_k24_long | 150000 | 60.981 | 0.833 | 0.022 | 0.177 | 0.002 | 0.500 | 0.199 | 0.250 | 0.009 |
| 2026-09-08 22:19 | final_v2_freeflight | 150000 | 62.264 | 0.833 | 0.021 | 0.186 | 0.002 | 0.600 | 0.290 | 0.250 | 0.009 |
| 2026-09-09 02:20 | final_v2_ff_energy | 150000 | 42.357 | 0.375 | 0.029 | 0.210 | 0.002 | 0.271 | 0.570 | 0.167 | 0.018 |
| 2026-09-09 02:20 | final_v2_ff_energy_q | 150000 | 48.089 | 0.417 | 0.023 | 0.194 | 0.002 | 0.400 | 0.221 | 0.167 | 0.017 |
| 2026-09-09 02:20 | final_v2_ff_smooth | 150000 | 34.247 | 0.208 | 0.027 | 0.299 | 0.003 | 0.250 | 0.235 | 0.208 | 0.030 |
| 2026-09-09 02:20 | final_v2_ff_fade | 150000 | 63.557 | 0.833 | 0.019 | 0.180 | 0.002 | 0.600 | 0.276 | 0.250 | 0.017 |

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
- **exp1_mixed_k12** (mixed_k12: {'K': 12, 'lr': 3e-06}) from exp1_rest_k24: **REJECT** -- composite 57.3 -> 54.1 (< margin 0.5)  
  deltas: composite -3.189, stability +0.042, trans_150 +0.002, surface_pen +0.000, support_jaccard -0.100
- **exp1_mixed_k12** (mixed_k12: {'K': 12, 'lr': 3e-06}) from exp1_rest_k24_long: **ACCEPT** -- composite 56.7 -> 58.3 (accepted)  
  deltas: composite +1.649, stability +0.042, trans_150 -0.004, surface_pen +0.001, support_jaccard +0.200
- **exp1_rest_k24** (rest_k24: {'K': 24, 'lr': 3e-06, 'rest_only': True}) from exp1_mixed_k12: **REJECT** -- composite 58.3 -> 58.7 (< margin 0.5)  
  deltas: composite +0.423, stability -0.083, trans_150 -0.000, surface_pen -0.000, support_jaccard +0.000
- **exp2_mixed_k12** (mixed_k12: {'K': 12, 'lr': 3e-06}) from exp1_mixed_k12: **REJECT** -- composite 58.3 -> 58.5 (< margin 0.5)  
  deltas: composite +0.144, stability -0.042, trans_150 +0.000, surface_pen -0.000, support_jaccard +0.000
- **exp3_rest_k24_long** (rest_k24_long: {'K': 24, 'lr': 3e-06, 'rest_only': True, 'steps': 6000}) from exp1_mixed_k12: **ACCEPT** -- composite 58.3 -> 61.0 (accepted)  
  deltas: composite +2.674, stability +0.042, trans_150 +0.001, surface_pen -0.001, support_jaccard -0.100
