# Fusion analysis

Per target change in rel_d against baseline_self. Moving block bootstrap over targets in index order, 2000 resamples, seed 0. Each interval is the widest 95 percent interval over block lengths 8, 15, 30. Robust means every one of these intervals excludes zero (0.5 for the improved share). Mean abs diff is the paired mean of fused minus baseline rel_d in rel_d points.

## Deployable headline (joint solved scale, causal adjacent window 4)

| Sequence | Mean change % [CI] | Mean robust | Median change % [CI] | Median robust | Improved % [CI] | Mean abs diff [CI] |
|---|---|---|---|---|---|---|
| fr1_desk | -2.4 [-7.7, 4.2] | no | -4.5 [-8.1, 4.5] | no | 64 [47, 78] | -0.129 [-0.409, 0.186] |
| fr1_room | -4.5 [-7.9, -1.5] | yes | -2.3 [-6.5, -0.6] | yes | 74 [58, 89] | -0.256 [-0.520, -0.060] |
| fr1_360 | -5.9 [-11.6, -1.0] | yes | -5.8 [-11.3, -3.3] | yes | 76 [66, 87] | -0.398 [-0.690, -0.160] |
| fr3 | -4.3 [-8.7, -1.6] | yes | -3.8 [-6.9, -2.0] | yes | 71 [61, 89] | -0.220 [-0.452, -0.089] |
| pooled | -4.3 [-6.6, -2.1] | yes | -4.1 [-5.9, -1.9] | yes | 71 [66, 79] | -0.251 [-0.374, -0.132] |

Pooled is the mean of the per sequence values, bootstrapped within each sequence.

## Measured versus forecast (oracle scale, plain weighted mean)

| Sequence | Config | Forecast % [band] | Measured mean % [CI] | Robust | Measured over forecast | Forecast outside CI |
|---|---|---|---|---|---|---|
| fr1_desk | causal_adjacent_w1 | -2.2 [-3.1, -1.4] | -1.8 [-3.2, -0.5] | yes | 0.84 | no |
| fr1_desk | causal_adjacent_w2 | -3.8 [-5.5, -2.7] | -1.6 [-4.1, 1.0] | no | 0.41 | no |
| fr1_desk | causal_adjacent_w4 | -6.2 [-9.0, -4.5] | -0.7 [-5.2, 4.2] | no | 0.12 | yes |
| fr1_desk | centred_adjacent_w1 | -3.9 [-5.6, -2.8] | -1.7 [-3.0, -0.8] | yes | 0.45 | yes |
| fr1_desk | centred_adjacent_w2 | -6.4 [-9.2, -4.7] | -1.4 [-3.2, 0.0] | no | 0.22 | yes |
| fr1_desk | centred_adjacent_w4 | -10.6 [-15.0, -8.1] | -1.7 [-4.1, 0.9] | no | 0.16 | yes |
| fr1_desk | causal_diverse_w1 | -4.5 [-6.9, -3.2] | 0.0 [-3.4, 3.6] | no | -0.00 | yes |
| fr1_desk | causal_diverse_w2 | -7.8 [-11.6, -5.8] | 1.8 [-3.7, 7.8] | no | -0.23 | yes |
| fr1_desk | causal_diverse_w3 | -10.6 [-15.4, -7.9] | 3.4 [-3.0, 10.5] | no | -0.32 | yes |
| fr1_desk | causal_diverse_w4 | -12.9 [-18.6, -9.8] | 4.7 [-1.9, 12.3] | no | -0.37 | yes |
| fr1_desk | centred_diverse_w1 | -8.5 [-12.3, -6.4] | -0.5 [-2.7, 2.0] | no | 0.06 | yes |
| fr1_desk | centred_diverse_w2 | -13.6 [-19.4, -10.5] | -1.2 [-5.2, 2.6] | no | 0.09 | yes |
| fr1_room | causal_adjacent_w1 | -1.4 [-1.6, -1.0] | -2.1 [-3.1, -1.0] | yes | 1.54 | no |
| fr1_room | causal_adjacent_w2 | -2.2 [-2.7, -1.7] | -3.3 [-5.0, -1.7] | yes | 1.45 | no |
| fr1_room | causal_adjacent_w4 | -3.6 [-4.5, -2.7] | -5.1 [-8.0, -2.7] | yes | 1.42 | no |
| fr1_room | centred_adjacent_w1 | -2.3 [-2.8, -1.8] | -1.6 [-2.7, -0.4] | yes | 0.70 | no |
| fr1_room | centred_adjacent_w2 | -3.7 [-4.5, -2.8] | -2.2 [-3.9, -0.5] | yes | 0.61 | no |
| fr1_room | centred_adjacent_w4 | -6.4 [-8.0, -4.8] | -3.3 [-5.6, -0.9] | yes | 0.51 | yes |
| fr1_room | causal_diverse_w1 | -2.5 [-3.4, -1.8] | -3.9 [-6.1, -2.2] | yes | 1.54 | no |
| fr1_room | causal_diverse_w2 | -4.9 [-6.2, -3.5] | -5.9 [-9.5, -3.3] | yes | 1.21 | no |
| fr1_room | causal_diverse_w3 | -6.9 [-8.7, -4.9] | -6.7 [-11.9, -3.4] | yes | 0.97 | no |
| fr1_room | causal_diverse_w4 | -8.5 [-10.7, -6.1] | -7.0 [-14.1, -3.0] | yes | 0.83 | no |
| fr1_room | centred_diverse_w1 | -5.1 [-6.5, -3.8] | -2.6 [-4.4, -0.6] | yes | 0.50 | yes |
| fr1_room | centred_diverse_w2 | -8.8 [-10.9, -6.4] | -3.9 [-6.3, -1.0] | yes | 0.44 | yes |
| fr1_360 | causal_adjacent_w1 | -1.7 [-2.0, -1.4] | -2.4 [-4.3, -1.1] | yes | 1.37 | no |
| fr1_360 | causal_adjacent_w2 | -2.9 [-3.5, -2.3] | -3.3 [-7.0, -0.2] | yes | 1.12 | no |
| fr1_360 | causal_adjacent_w4 | -4.8 [-5.8, -3.6] | -3.5 [-10.1, 4.0] | no | 0.73 | no |
| fr1_360 | centred_adjacent_w1 | -3.0 [-3.6, -2.4] | -3.4 [-4.5, -2.3] | yes | 1.13 | no |
| fr1_360 | centred_adjacent_w2 | -5.0 [-6.0, -3.8] | -5.2 [-7.1, -3.3] | yes | 1.03 | no |
| fr1_360 | centred_adjacent_w4 | -9.3 [-11.7, -6.9] | -7.7 [-9.9, -4.7] | yes | 0.83 | no |
| fr1_360 | causal_diverse_w1 | -3.3 [-4.1, -2.3] | -3.2 [-8.6, 2.2] | no | 0.97 | no |
| fr1_360 | causal_diverse_w2 | -6.5 [-8.4, -4.5] | -2.3 [-11.1, 8.8] | no | 0.36 | no |
| fr1_360 | causal_diverse_w3 | -9.5 [-12.5, -6.5] | 0.0 [-11.9, 15.4] | no | -0.00 | no |
| fr1_360 | causal_diverse_w4 | -12.1 [-15.9, -8.2] | 3.9 [-11.6, 23.9] | no | -0.32 | yes |
| fr1_360 | centred_diverse_w1 | -7.3 [-9.2, -5.2] | -6.3 [-8.3, -3.6] | yes | 0.87 | no |
| fr1_360 | centred_diverse_w2 | -13.0 [-16.9, -9.1] | -6.6 [-9.3, -1.0] | yes | 0.51 | yes |
| fr3 | causal_adjacent_w1 | -1.6 [-2.6, -1.1] | -1.9 [-2.8, -1.2] | yes | 1.22 | no |
| fr3 | causal_adjacent_w2 | -2.4 [-3.9, -1.7] | -2.4 [-3.8, -1.2] | yes | 1.00 | no |
| fr3 | causal_adjacent_w4 | -3.5 [-5.6, -2.4] | -3.4 [-5.8, -1.6] | yes | 0.96 | no |
| fr3 | centred_adjacent_w1 | -2.4 [-3.9, -1.7] | -2.1 [-2.9, -1.3] | yes | 0.87 | no |
| fr3 | centred_adjacent_w2 | -3.5 [-5.6, -2.5] | -2.3 [-3.4, -1.2] | yes | 0.67 | yes |
| fr3 | centred_adjacent_w4 | -5.4 [-8.4, -3.8] | -3.1 [-4.3, -1.9] | yes | 0.57 | yes |
| fr3 | causal_diverse_w1 | -2.4 [-3.9, -1.7] | -2.2 [-3.9, -0.8] | yes | 0.91 | no |
| fr3 | causal_diverse_w2 | -4.2 [-6.6, -2.9] | -3.0 [-6.1, -0.7] | yes | 0.71 | no |
| fr3 | causal_diverse_w3 | -5.7 [-8.8, -3.9] | -3.5 [-7.8, -0.4] | yes | 0.61 | no |
| fr3 | causal_diverse_w4 | -6.8 [-10.4, -4.6] | -3.4 [-9.1, 0.3] | no | 0.50 | no |
| fr3 | centred_diverse_w1 | -4.3 [-6.7, -3.0] | -2.3 [-3.3, -1.3] | yes | 0.53 | yes |
| fr3 | centred_diverse_w2 | -6.9 [-10.5, -4.7] | -3.2 [-4.7, -2.0] | yes | 0.46 | yes |

## Controls

| Sequence | Control | Reference | Mean change % vs baseline [CI] | Robust | Mean change % vs reference [CI] | Robust |
|---|---|---|---|---|---|---|
| fr1_desk | shuffled_poses_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 212.0 [146.8, 252.9] | yes | 183.2 [125.1, 214.3] | yes |
| fr1_desk | pose_noise_t0mm_r0deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 9.0 [2.2, 17.9] | yes | 0.0 [0.0, 0.0] | no |
| fr1_desk | pose_noise_t5mm_r0deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 9.5 [2.3, 18.4] | yes | 0.4 [-0.6, 1.0] | no |
| fr1_desk | pose_noise_t10mm_r0deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 12.2 [4.3, 21.4] | yes | 2.8 [0.7, 3.8] | yes |
| fr1_desk | pose_noise_t20mm_r0deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 23.9 [13.7, 33.2] | yes | 13.7 [6.8, 16.8] | yes |
| fr1_desk | pose_noise_t0mm_r0.25deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 9.5 [2.5, 18.3] | yes | 0.6 [-0.4, 1.7] | no |
| fr1_desk | pose_noise_t0mm_r0.5deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 12.8 [5.1, 21.4] | yes | 3.8 [0.5, 6.1] | yes |
| fr1_desk | pose_noise_t0mm_r1deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 26.6 [14.1, 35.4] | yes | 16.9 [5.9, 22.3] | yes |
| fr1_desk | shuffled_poses_joint_causal_adjacent_w4 | joint_causal_adjacent_w4 | 145.8 [82.0, 195.6] | yes | 161.5 [82.6, 223.6] | yes |
| fr1_room | shuffled_poses_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 60.2 [50.3, 71.5] | yes | 72.2 [61.3, 86.7] | yes |
| fr1_room | pose_noise_t0mm_r0deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | -6.5 [-14.2, -2.1] | yes | 0.0 [0.0, 0.0] | no |
| fr1_room | pose_noise_t5mm_r0deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | -6.3 [-13.7, -1.8] | yes | 0.3 [-0.1, 0.9] | no |
| fr1_room | pose_noise_t10mm_r0deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | -5.3 [-12.2, -0.8] | yes | 1.5 [0.5, 3.1] | yes |
| fr1_room | pose_noise_t20mm_r0deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | -1.7 [-7.5, 2.8] | no | 5.7 [3.1, 9.2] | yes |
| fr1_room | pose_noise_t0mm_r0.25deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | -6.3 [-13.6, -1.9] | yes | 0.3 [-0.3, 1.0] | no |
| fr1_room | pose_noise_t0mm_r0.5deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | -4.9 [-11.7, -0.6] | yes | 1.9 [0.5, 3.6] | yes |
| fr1_room | pose_noise_t0mm_r1deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 0.4 [-5.1, 4.4] | no | 7.9 [4.2, 12.6] | yes |
| fr1_room | shuffled_poses_joint_causal_adjacent_w4 | joint_causal_adjacent_w4 | 57.8 [31.6, 79.0] | yes | 65.4 [37.2, 86.4] | yes |
| fr1_360 | shuffled_poses_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 152.8 [28.4, 301.8] | yes | 131.9 [38.8, 226.7] | yes |
| fr1_360 | pose_noise_t0mm_r0deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 2.8 [-12.6, 26.2] | no | 0.0 [0.0, 0.0] | no |
| fr1_360 | pose_noise_t5mm_r0deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 1.7 [-13.3, 24.8] | no | -0.8 [-1.9, 0.4] | no |
| fr1_360 | pose_noise_t10mm_r0deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 0.7 [-13.8, 22.9] | no | -1.3 [-3.6, 1.2] | no |
| fr1_360 | pose_noise_t20mm_r0deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 1.4 [-12.3, 21.8] | no | 0.2 [-3.0, 4.2] | no |
| fr1_360 | pose_noise_t0mm_r0.25deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 1.2 [-12.8, 22.8] | no | -0.8 [-1.9, 0.7] | no |
| fr1_360 | pose_noise_t0mm_r0.5deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 1.1 [-12.4, 22.1] | no | -0.5 [-2.6, 2.7] | no |
| fr1_360 | pose_noise_t0mm_r1deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 5.4 [-7.6, 25.3] | no | 4.6 [1.3, 10.4] | yes |
| fr1_360 | shuffled_poses_joint_causal_adjacent_w4 | joint_causal_adjacent_w4 | 76.9 [21.9, 109.4] | yes | 90.4 [32.5, 120.8] | yes |
| fr3 | shuffled_poses_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 142.1 [80.2, 202.1] | yes | 143.8 [94.0, 193.5] | yes |
| fr3 | pose_noise_t0mm_r0deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | -1.7 [-7.8, 2.4] | no | 0.0 [0.0, 0.0] | no |
| fr3 | pose_noise_t5mm_r0deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | -0.9 [-8.1, 3.5] | no | 0.7 [-0.6, 1.4] | no |
| fr3 | pose_noise_t10mm_r0deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 5.2 [-3.0, 10.1] | no | 6.9 [4.7, 8.3] | yes |
| fr3 | pose_noise_t20mm_r0deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 26.5 [15.2, 33.5] | yes | 28.4 [23.6, 31.6] | yes |
| fr3 | pose_noise_t0mm_r0.25deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 0.6 [-5.8, 5.4] | no | 2.5 [1.5, 3.7] | yes |
| fr3 | pose_noise_t0mm_r0.5deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 9.1 [1.1, 14.9] | yes | 11.2 [8.7, 13.8] | yes |
| fr3 | pose_noise_t0mm_r1deg_causal_diverse_w4_rung_d | causal_diverse_w4_rung_d | 33.3 [21.2, 41.0] | yes | 36.1 [30.2, 40.5] | yes |
| fr3 | shuffled_poses_joint_causal_adjacent_w4 | joint_causal_adjacent_w4 | 193.1 [146.0, 243.0] | yes | 209.6 [151.0, 275.9] | yes |

## No fusion control minus baseline_self

| Sequence | Targets | Mean diff [CI] | Max abs diff |
|---|---|---|---|
| fr1_desk | 100 | -1.86e-09 [-1.17e-07, 5.03e-08] | 1.49e-06 |
| fr1_room | 100 | 5.03e-08 [-1.68e-08, 1.32e-07] | 1.49e-06 |
| fr1_360 | 100 | -2.93e-08 [-1.21e-07, 2.61e-08] | 1.49e-06 |
| fr3 | 100 | 5.59e-08 [0.00e+00, 1.01e-07] | 1.12e-06 |

## Oracle ladder

| Sequence | Config | Mean change % [CI] | Mean robust | Median change % [CI] | Median robust | Improved % [CI] | Mean abs diff [CI] |
|---|---|---|---|---|---|---|---|
| fr1_desk | causal_adjacent_w1 | -1.8 [-3.2, -0.5] | yes | -2.2 [-3.7, -0.1] | yes | 61 [51, 72] | -0.101 [-0.176, -0.034] |
| fr1_desk | causal_adjacent_w2 | -1.6 [-4.1, 1.0] | no | -2.2 [-4.3, -0.5] | yes | 59 [50, 69] | -0.114 [-0.248, 0.018] |
| fr1_desk | causal_adjacent_w4 | -0.7 [-5.2, 4.2] | no | -2.2 [-6.4, 1.4] | no | 57 [48, 68] | -0.118 [-0.342, 0.107] |
| fr1_desk | causal_diverse_w1 | 0.0 [-3.4, 3.6] | no | -0.3 [-6.1, 2.1] | no | 52 [42, 64] | -0.070 [-0.240, 0.098] |
| fr1_desk | causal_diverse_w2 | 1.8 [-3.7, 7.8] | no | 0.1 [-4.9, 5.0] | no | 50 [38, 63] | -0.040 [-0.310, 0.225] |
| fr1_desk | causal_diverse_w3 | 3.4 [-3.0, 10.5] | no | 0.7 [-3.7, 9.8] | no | 44 [29, 59] | 0.005 [-0.317, 0.312] |
| fr1_desk | causal_diverse_w4 | 4.7 [-1.9, 12.3] | no | 3.8 [-3.0, 11.3] | no | 43 [29, 58] | 0.056 [-0.288, 0.377] |
| fr1_desk | centred_adjacent_w1 | -1.7 [-3.0, -0.8] | yes | -1.7 [-3.0, -1.0] | yes | 65 [56, 79] | -0.105 [-0.148, -0.077] |
| fr1_desk | centred_adjacent_w2 | -1.4 [-3.2, 0.0] | no | -2.2 [-4.2, -1.1] | yes | 63 [55, 73] | -0.114 [-0.184, -0.059] |
| fr1_desk | centred_adjacent_w4 | -1.7 [-4.1, 0.9] | no | -2.5 [-4.9, -0.9] | yes | 64 [54, 76] | -0.162 [-0.301, -0.048] |
| fr1_desk | centred_diverse_w1 | -0.5 [-2.7, 2.0] | no | -1.7 [-4.0, 2.6] | no | 60 [46, 72] | -0.109 [-0.243, 0.003] |
| fr1_desk | centred_diverse_w2 | -1.2 [-5.2, 2.6] | no | -2.5 [-4.8, 3.2] | no | 56 [45, 68] | -0.172 [-0.429, 0.028] |
| fr1_desk | causal_diverse_w4_rung_a | 4.7 [-1.9, 12.3] | no | 3.8 [-3.0, 11.3] | no | 43 [29, 58] | 0.056 [-0.288, 0.377] |
| fr1_desk | causal_diverse_w4_rung_b | 2.6 [-3.6, 9.4] | no | 1.7 [-4.9, 8.4] | no | 46 [33, 60] | -0.035 [-0.351, 0.261] |
| fr1_desk | causal_diverse_w4_rung_c | 3.8 [-2.6, 11.0] | no | 2.9 [-3.8, 9.8] | no | 44 [30, 58] | 0.015 [-0.318, 0.324] |
| fr1_desk | causal_diverse_w4_rung_d | 9.0 [2.2, 17.9] | yes | 4.3 [-0.6, 13.3] | no | 38 [26, 51] | 0.287 [0.020, 0.601] |
| fr1_desk | causal_diverse_w4_rung_e | 5.8 [-0.5, 13.8] | no | 3.9 [-1.8, 9.5] | no | 40 [27, 54] | 0.131 [-0.154, 0.443] |
| fr1_desk | causal_diverse_w4_rung_f | 4.8 [-0.7, 12.1] | no | 3.0 [-1.7, 8.2] | no | 42 [29, 56] | 0.103 [-0.137, 0.375] |
| fr1_desk | centred_adjacent_w4_rung_a | -1.7 [-4.1, 0.9] | no | -2.5 [-4.9, -0.9] | yes | 64 [54, 76] | -0.162 [-0.301, -0.048] |
| fr1_desk | centred_adjacent_w4_rung_b | -4.1 [-6.0, -1.8] | yes | -3.9 [-6.7, -1.3] | yes | 66 [58, 74] | -0.261 [-0.427, -0.137] |
| fr1_desk | centred_adjacent_w4_rung_c | -2.9 [-4.9, -0.8] | yes | -3.1 [-6.1, -1.4] | yes | 67 [57, 77] | -0.213 [-0.347, -0.109] |
| fr1_desk | centred_adjacent_w4_rung_d | -2.1 [-7.4, 2.9] | no | -5.8 [-9.9, -1.0] | yes | 65 [51, 87] | -0.171 [-0.406, 0.066] |
| fr1_desk | centred_adjacent_w4_rung_e | -2.8 [-7.1, 1.7] | no | -5.2 [-8.5, -1.8] | yes | 70 [58, 86] | -0.193 [-0.400, 0.006] |
| fr1_desk | centred_adjacent_w4_rung_f | -2.9 [-7.4, 2.0] | no | -5.7 [-8.5, -2.2] | yes | 70 [57, 86] | -0.188 [-0.402, 0.026] |
| fr1_room | causal_adjacent_w1 | -2.1 [-3.1, -1.0] | yes | -1.5 [-2.9, -0.2] | yes | 69 [59, 78] | -0.115 [-0.189, -0.042] |
| fr1_room | causal_adjacent_w2 | -3.3 [-5.0, -1.7] | yes | -2.1 [-4.2, -0.5] | yes | 72 [60, 86] | -0.176 [-0.294, -0.071] |
| fr1_room | causal_adjacent_w4 | -5.1 [-8.0, -2.7] | yes | -3.7 [-6.7, -1.3] | yes | 72 [55, 90] | -0.264 [-0.426, -0.121] |
| fr1_room | causal_diverse_w1 | -3.9 [-6.1, -2.2] | yes | -2.6 [-5.5, -1.2] | yes | 79 [68, 89] | -0.200 [-0.323, -0.099] |
| fr1_room | causal_diverse_w2 | -5.9 [-9.5, -3.3] | yes | -5.2 [-8.1, -2.0] | yes | 80 [68, 92] | -0.289 [-0.438, -0.157] |
| fr1_room | causal_diverse_w3 | -6.7 [-11.9, -3.4] | yes | -5.9 [-10.4, -2.1] | yes | 79 [65, 92] | -0.312 [-0.474, -0.169] |
| fr1_room | causal_diverse_w4 | -7.0 [-14.1, -3.0] | yes | -5.9 [-12.1, -1.3] | yes | 73 [57, 92] | -0.312 [-0.509, -0.159] |
| fr1_room | centred_adjacent_w1 | -1.6 [-2.7, -0.4] | yes | -0.9 [-1.9, 0.1] | no | 64 [49, 74] | -0.104 [-0.197, -0.014] |
| fr1_room | centred_adjacent_w2 | -2.2 [-3.9, -0.5] | yes | -1.6 [-3.2, -0.1] | yes | 70 [58, 83] | -0.147 [-0.284, -0.019] |
| fr1_room | centred_adjacent_w4 | -3.3 [-5.6, -0.9] | yes | -2.3 [-5.2, -0.3] | yes | 73 [54, 87] | -0.213 [-0.412, -0.036] |
| fr1_room | centred_diverse_w1 | -2.6 [-4.4, -0.6] | yes | -1.5 [-4.5, 0.2] | no | 64 [46, 74] | -0.169 [-0.326, -0.028] |
| fr1_room | centred_diverse_w2 | -3.9 [-6.3, -1.0] | yes | -4.0 [-6.7, -0.5] | yes | 73 [53, 87] | -0.247 [-0.465, -0.045] |
| fr1_room | causal_diverse_w4_rung_a | -7.0 [-14.1, -3.0] | yes | -5.9 [-12.1, -1.3] | yes | 73 [57, 92] | -0.312 [-0.509, -0.159] |
| fr1_room | causal_diverse_w4_rung_b | -8.3 [-14.7, -4.1] | yes | -7.0 [-12.4, -2.9] | yes | 79 [65, 94] | -0.383 [-0.568, -0.202] |
| fr1_room | causal_diverse_w4_rung_c | -7.7 [-14.5, -3.6] | yes | -6.7 [-13.1, -2.5] | yes | 77 [62, 92] | -0.349 [-0.538, -0.185] |
| fr1_room | causal_diverse_w4_rung_d | -6.5 [-14.2, -2.1] | yes | -5.6 [-13.0, -0.3] | yes | 69 [51, 91] | -0.285 [-0.534, -0.099] |
| fr1_room | causal_diverse_w4_rung_e | -6.9 [-14.2, -2.9] | yes | -5.3 [-12.4, -1.9] | yes | 76 [59, 93] | -0.292 [-0.470, -0.148] |
| fr1_room | causal_diverse_w4_rung_f | -7.6 [-13.9, -4.4] | yes | -7.2 [-11.7, -4.3] | yes | 83 [70, 96] | -0.331 [-0.484, -0.211] |
| fr1_room | centred_adjacent_w4_rung_a | -3.3 [-5.6, -0.9] | yes | -2.3 [-5.2, -0.3] | yes | 73 [54, 87] | -0.213 [-0.412, -0.036] |
| fr1_room | centred_adjacent_w4_rung_b | -4.0 [-7.2, -1.0] | yes | -3.2 [-7.3, -0.5] | yes | 75 [55, 87] | -0.261 [-0.512, -0.040] |
| fr1_room | centred_adjacent_w4_rung_c | -3.7 [-6.4, -1.0] | yes | -2.9 [-6.8, -0.5] | yes | 76 [56, 89] | -0.240 [-0.469, -0.040] |
| fr1_room | centred_adjacent_w4_rung_d | -4.5 [-9.7, -0.7] | yes | -2.8 [-11.2, -0.2] | yes | 66 [52, 82] | -0.312 [-0.721, -0.021] |
| fr1_room | centred_adjacent_w4_rung_e | -4.6 [-8.7, -1.0] | yes | -2.8 [-10.1, -0.2] | yes | 69 [52, 82] | -0.308 [-0.649, -0.034] |
| fr1_room | centred_adjacent_w4_rung_f | -3.1 [-6.4, -0.2] | yes | -1.4 [-6.9, 0.3] | no | 63 [44, 81] | -0.219 [-0.481, -0.006] |
| fr1_360 | causal_adjacent_w1 | -2.4 [-4.3, -1.1] | yes | -2.9 [-4.3, -1.3] | yes | 65 [58, 73] | -0.139 [-0.203, -0.093] |
| fr1_360 | causal_adjacent_w2 | -3.3 [-7.0, -0.2] | yes | -4.6 [-6.4, -2.2] | yes | 73 [65, 82] | -0.225 [-0.341, -0.134] |
| fr1_360 | causal_adjacent_w4 | -3.5 [-10.1, 4.0] | no | -6.9 [-9.6, -3.2] | yes | 79 [68, 91] | -0.332 [-0.520, -0.172] |
| fr1_360 | causal_diverse_w1 | -3.2 [-8.6, 2.2] | no | -4.2 [-6.3, -2.9] | yes | 76 [65, 89] | -0.251 [-0.404, -0.121] |
| fr1_360 | causal_diverse_w2 | -2.3 [-11.1, 8.8] | no | -5.9 [-8.5, -3.6] | yes | 78 [65, 94] | -0.342 [-0.569, -0.148] |
| fr1_360 | causal_diverse_w3 | 0.0 [-11.9, 15.4] | no | -6.4 [-9.9, -2.9] | yes | 76 [62, 94] | -0.358 [-0.640, -0.129] |
| fr1_360 | causal_diverse_w4 | 3.9 [-11.6, 23.9] | no | -6.0 [-8.8, -1.2] | yes | 72 [56, 93] | -0.329 [-0.660, -0.063] |
| fr1_360 | centred_adjacent_w1 | -3.4 [-4.5, -2.3] | yes | -3.2 [-5.3, -2.1] | yes | 67 [60, 71] | -0.174 [-0.238, -0.120] |
| fr1_360 | centred_adjacent_w2 | -5.2 [-7.1, -3.3] | yes | -4.6 [-6.4, -3.7] | yes | 69 [62, 75] | -0.255 [-0.338, -0.175] |
| fr1_360 | centred_adjacent_w4 | -7.7 [-9.9, -4.7] | yes | -5.9 [-8.2, -4.3] | yes | 79 [72, 85] | -0.370 [-0.475, -0.255] |
| fr1_360 | centred_diverse_w1 | -6.3 [-8.3, -3.6] | yes | -4.5 [-6.9, -3.2] | yes | 78 [70, 86] | -0.303 [-0.395, -0.197] |
| fr1_360 | centred_diverse_w2 | -6.6 [-9.3, -1.0] | yes | -6.4 [-9.0, -4.3] | yes | 79 [65, 92] | -0.404 [-0.543, -0.252] |
| fr1_360 | causal_diverse_w4_rung_a | 3.9 [-11.6, 23.9] | no | -6.0 [-8.8, -1.2] | yes | 72 [56, 93] | -0.329 [-0.660, -0.063] |
| fr1_360 | causal_diverse_w4_rung_b | 4.0 [-11.8, 27.3] | no | -6.8 [-9.6, -2.4] | yes | 72 [57, 91] | -0.363 [-0.675, -0.088] |
| fr1_360 | causal_diverse_w4_rung_c | 3.7 [-11.7, 24.9] | no | -6.5 [-9.4, -2.0] | yes | 72 [56, 93] | -0.353 [-0.669, -0.087] |
| fr1_360 | causal_diverse_w4_rung_d | 2.8 [-12.6, 26.2] | no | -7.6 [-10.9, -3.8] | yes | 78 [62, 95] | -0.422 [-0.737, -0.142] |
| fr1_360 | causal_diverse_w4_rung_e | 4.7 [-10.9, 27.6] | no | -5.4 [-9.9, -1.4] | yes | 73 [59, 91] | -0.311 [-0.579, -0.062] |
| fr1_360 | causal_diverse_w4_rung_f | -7.4 [-12.3, -4.4] | yes | -6.1 [-12.3, -3.5] | yes | 82 [70, 97] | -0.399 [-0.626, -0.261] |
| fr1_360 | centred_adjacent_w4_rung_a | -7.7 [-9.9, -4.7] | yes | -5.9 [-8.2, -4.3] | yes | 79 [72, 85] | -0.370 [-0.475, -0.255] |
| fr1_360 | centred_adjacent_w4_rung_b | -6.9 [-9.0, -4.5] | yes | -5.5 [-8.2, -4.0] | yes | 80 [73, 86] | -0.340 [-0.461, -0.234] |
| fr1_360 | centred_adjacent_w4_rung_c | -7.8 [-9.9, -5.1] | yes | -6.0 [-8.7, -4.5] | yes | 80 [73, 87] | -0.378 [-0.505, -0.260] |
| fr1_360 | centred_adjacent_w4_rung_d | -10.1 [-13.7, -4.0] | yes | -9.3 [-14.8, -5.6] | yes | 85 [72, 95] | -0.529 [-0.830, -0.224] |
| fr1_360 | centred_adjacent_w4_rung_e | -8.7 [-11.9, -4.0] | yes | -7.8 [-12.8, -4.7] | yes | 83 [72, 92] | -0.454 [-0.715, -0.229] |
| fr1_360 | centred_adjacent_w4_rung_f | -6.6 [-9.7, -3.4] | yes | -5.3 [-8.9, -1.8] | yes | 77 [68, 90] | -0.369 [-0.612, -0.163] |
| fr3 | causal_adjacent_w1 | -1.9 [-2.8, -1.2] | yes | -2.1 [-3.2, -1.2] | yes | 70 [60, 80] | -0.100 [-0.149, -0.068] |
| fr3 | causal_adjacent_w2 | -2.4 [-3.8, -1.2] | yes | -2.6 [-3.9, -1.1] | yes | 66 [57, 81] | -0.132 [-0.204, -0.081] |
| fr3 | causal_adjacent_w4 | -3.4 [-5.8, -1.6] | yes | -3.5 [-5.5, -1.6] | yes | 68 [58, 86] | -0.185 [-0.303, -0.107] |
| fr3 | causal_diverse_w1 | -2.2 [-3.9, -0.8] | yes | -2.8 [-4.6, -1.2] | yes | 71 [62, 84] | -0.118 [-0.202, -0.064] |
| fr3 | causal_diverse_w2 | -3.0 [-6.1, -0.7] | yes | -3.6 [-7.9, -1.7] | yes | 68 [57, 83] | -0.167 [-0.309, -0.072] |
| fr3 | causal_diverse_w3 | -3.5 [-7.8, -0.4] | yes | -3.8 [-9.6, 0.5] | no | 63 [48, 87] | -0.194 [-0.387, -0.066] |
| fr3 | causal_diverse_w4 | -3.4 [-9.1, 0.3] | no | -4.3 [-11.7, 2.6] | no | 62 [46, 89] | -0.207 [-0.452, -0.052] |
| fr3 | centred_adjacent_w1 | -2.1 [-2.9, -1.3] | yes | -2.4 [-3.6, -2.1] | yes | 71 [63, 81] | -0.113 [-0.154, -0.084] |
| fr3 | centred_adjacent_w2 | -2.3 [-3.4, -1.2] | yes | -2.4 [-4.1, -0.5] | yes | 64 [51, 74] | -0.132 [-0.180, -0.094] |
| fr3 | centred_adjacent_w4 | -3.1 [-4.3, -1.9] | yes | -3.8 [-4.9, -1.7] | yes | 68 [56, 79] | -0.172 [-0.233, -0.124] |
| fr3 | centred_diverse_w1 | -2.3 [-3.3, -1.3] | yes | -3.4 [-4.6, -2.6] | yes | 65 [59, 73] | -0.126 [-0.180, -0.087] |
| fr3 | centred_diverse_w2 | -3.2 [-4.7, -2.0] | yes | -4.2 [-5.3, -3.1] | yes | 71 [66, 80] | -0.172 [-0.256, -0.118] |
| fr3 | causal_diverse_w4_rung_a | -3.4 [-9.1, 0.3] | no | -4.3 [-11.7, 2.6] | no | 62 [46, 89] | -0.207 [-0.452, -0.052] |
| fr3 | causal_diverse_w4_rung_b | -5.1 [-11.5, -0.8] | yes | -6.6 [-13.1, -0.6] | yes | 68 [52, 92] | -0.271 [-0.554, -0.092] |
| fr3 | causal_diverse_w4_rung_c | -4.3 [-10.4, -0.5] | yes | -4.3 [-12.5, 0.4] | no | 65 [50, 90] | -0.241 [-0.501, -0.081] |
| fr3 | causal_diverse_w4_rung_d | -1.7 [-7.8, 2.4] | no | -4.0 [-12.2, 3.3] | no | 61 [46, 87] | -0.124 [-0.375, 0.039] |
| fr3 | causal_diverse_w4_rung_e | -2.6 [-7.5, 0.9] | no | -3.3 [-11.8, -0.7] | yes | 67 [52, 90] | -0.163 [-0.376, -0.024] |
| fr3 | causal_diverse_w4_rung_f | -2.8 [-6.8, 0.1] | no | -4.6 [-10.1, -1.2] | yes | 67 [52, 90] | -0.163 [-0.342, -0.047] |
| fr3 | centred_adjacent_w4_rung_a | -3.1 [-4.3, -1.9] | yes | -3.8 [-4.9, -1.7] | yes | 68 [56, 79] | -0.172 [-0.233, -0.124] |
| fr3 | centred_adjacent_w4_rung_b | -6.0 [-10.2, -3.0] | yes | -5.2 [-7.5, -4.0] | yes | 75 [67, 84] | -0.291 [-0.537, -0.140] |
| fr3 | centred_adjacent_w4_rung_c | -4.5 [-6.3, -3.1] | yes | -4.5 [-5.9, -3.5] | yes | 71 [60, 82] | -0.227 [-0.353, -0.148] |
| fr3 | centred_adjacent_w4_rung_d | 4.1 [-7.3, 22.0] | no | -5.4 [-8.4, 22.6] | no | 70 [40, 90] | 0.106 [-0.330, 0.777] |
| fr3 | centred_adjacent_w4_rung_e | 0.3 [-6.8, 13.1] | no | -5.8 [-8.0, -1.4] | yes | 77 [57, 90] | -0.021 [-0.299, 0.459] |
| fr3 | centred_adjacent_w4_rung_f | 0.2 [-7.0, 13.3] | no | -6.4 [-7.7, -3.1] | yes | 81 [60, 94] | -0.025 [-0.309, 0.468] |

## Edge group (oracle scale)

| Sequence | Config | Mean change % [CI] | Mean robust | Median change % [CI] | Median robust | Improved % [CI] | Mean abs diff [CI] |
|---|---|---|---|---|---|---|---|
| fr1_desk | rungb_causal_diverse_w4 | 2.6 [-3.6, 9.4] | no | 1.7 [-4.9, 8.4] | no | 46 [33, 60] | -0.035 [-0.351, 0.261] |
| fr1_desk | edge_src2_causal_diverse_w4 | 7.3 [0.3, 15.4] | yes | 4.8 [-2.4, 12.6] | no | 38 [25, 54] | 0.154 [-0.185, 0.496] |
| fr1_desk | edge_src5_causal_diverse_w4 | 8.6 [1.4, 17.1] | yes | 6.1 [-1.4, 12.7] | no | 36 [23, 52] | 0.214 [-0.110, 0.556] |
| fr1_desk | edge_tgt3_causal_diverse_w4 | 2.9 [-2.7, 9.1] | no | 2.1 [-4.4, 9.0] | no | 44 [30, 57] | -0.017 [-0.340, 0.276] |
| fr1_desk | edge_src2_tgt3_causal_diverse_w4 | 3.2 [-2.4, 9.3] | no | 2.5 [-3.7, 8.2] | no | 43 [28, 56] | 0.003 [-0.304, 0.297] |
| fr1_desk | rungb_centred_adjacent_w4 | -4.1 [-6.0, -1.8] | yes | -3.9 [-6.7, -1.3] | yes | 66 [58, 74] | -0.261 [-0.427, -0.137] |
| fr1_desk | edge_src2_centred_adjacent_w4 | 3.8 [-1.2, 8.7] | no | 4.4 [-0.5, 8.0] | no | 35 [16, 54] | 0.058 [-0.228, 0.310] |
| fr1_desk | edge_src5_centred_adjacent_w4 | 3.1 [-1.1, 7.8] | no | 3.0 [-0.6, 8.1] | no | 37 [19, 55] | 0.045 [-0.215, 0.322] |
| fr1_desk | edge_tgt3_centred_adjacent_w4 | -1.8 [-4.0, 0.2] | no | -1.5 [-3.4, -0.1] | yes | 63 [53, 75] | -0.166 [-0.350, -0.029] |
| fr1_desk | edge_src2_tgt3_centred_adjacent_w4 | -1.7 [-3.8, 0.2] | no | -1.2 [-3.1, -0.2] | yes | 61 [52, 71] | -0.158 [-0.334, -0.027] |
| fr1_desk | rungb_causal_adjacent_w4 | -1.1 [-5.8, 3.9] | no | -2.2 [-6.2, -0.3] | yes | 60 [50, 70] | -0.134 [-0.383, 0.102] |
| fr1_desk | edge_src2_causal_adjacent_w4 | 3.3 [-1.6, 8.8] | no | 0.8 [-1.0, 4.0] | no | 47 [37, 59] | 0.052 [-0.205, 0.307] |
| fr1_desk | edge_src5_causal_adjacent_w4 | 2.4 [-2.1, 7.6] | no | 0.2 [-2.0, 1.9] | no | 48 [38, 61] | 0.018 [-0.221, 0.258] |
| fr1_desk | edge_tgt3_causal_adjacent_w4 | 1.4 [-2.5, 5.3] | no | 0.0 [-2.6, 2.9] | no | 49 [38, 61] | -0.019 [-0.228, 0.176] |
| fr1_desk | edge_src2_tgt3_causal_adjacent_w4 | 1.5 [-2.3, 5.5] | no | 0.3 [-2.4, 2.3] | no | 49 [38, 61] | -0.011 [-0.215, 0.189] |
| fr1_desk | gt_ceiling_causal_adjacent_w1_noself | -60.6 [-65.8, -57.6] | yes | -64.6 [-68.3, -59.5] | yes | 100 [100, 100] | -3.192 [-4.029, -2.725] |
| fr1_desk | gt_ceiling_causal_diverse_w4_noself_src2_tgt3 | -64.8 [-71.6, -58.0] | yes | -66.8 [-72.8, -62.5] | yes | 100 [100, 100] | -3.315 [-4.034, -2.975] |
| fr1_room | rungb_causal_diverse_w4 | -8.3 [-14.7, -4.1] | yes | -7.0 [-12.4, -2.9] | yes | 79 [65, 94] | -0.383 [-0.568, -0.202] |
| fr1_room | edge_src2_causal_diverse_w4 | -6.4 [-13.9, -2.3] | yes | -4.1 [-11.0, -1.9] | yes | 74 [61, 94] | -0.264 [-0.439, -0.116] |
| fr1_room | edge_src5_causal_diverse_w4 | -4.7 [-13.4, -0.3] | yes | -3.0 [-10.4, 0.6] | no | 65 [49, 93] | -0.165 [-0.406, -0.011] |
| fr1_room | edge_tgt3_causal_diverse_w4 | -3.8 [-12.8, 1.3] | no | -2.5 [-10.8, 2.1] | no | 61 [41, 90] | -0.116 [-0.387, 0.063] |
| fr1_room | edge_src2_tgt3_causal_diverse_w4 | -3.2 [-12.7, 2.4] | no | -1.8 [-10.3, 3.7] | no | 57 [34, 90] | -0.074 [-0.379, 0.117] |
| fr1_room | rungb_centred_adjacent_w4 | -4.0 [-7.2, -1.0] | yes | -3.2 [-7.3, -0.5] | yes | 75 [55, 87] | -0.261 [-0.512, -0.040] |
| fr1_room | edge_src2_centred_adjacent_w4 | -2.7 [-4.9, -0.6] | yes | -2.6 [-5.2, -0.3] | yes | 70 [54, 85] | -0.177 [-0.349, -0.027] |
| fr1_room | edge_src5_centred_adjacent_w4 | -2.6 [-4.6, -0.8] | yes | -2.7 [-4.6, -0.2] | yes | 70 [55, 85] | -0.168 [-0.329, -0.034] |
| fr1_room | edge_tgt3_centred_adjacent_w4 | -2.9 [-5.2, -0.6] | yes | -2.6 [-4.9, -0.1] | yes | 69 [54, 82] | -0.187 [-0.374, -0.028] |
| fr1_room | edge_src2_tgt3_centred_adjacent_w4 | -2.7 [-4.9, -0.6] | yes | -2.6 [-4.6, -0.1] | yes | 69 [54, 82] | -0.176 [-0.357, -0.027] |
| fr1_room | rungb_causal_adjacent_w4 | -5.1 [-8.3, -2.5] | yes | -3.5 [-6.8, -1.0] | yes | 76 [62, 92] | -0.273 [-0.462, -0.113] |
| fr1_room | edge_src2_causal_adjacent_w4 | -3.6 [-6.5, -1.6] | yes | -2.2 [-5.3, -0.7] | yes | 69 [56, 86] | -0.174 [-0.278, -0.084] |
| fr1_room | edge_src5_causal_adjacent_w4 | -2.5 [-5.6, -0.5] | yes | -1.8 [-4.2, 0.5] | no | 62 [46, 84] | -0.106 [-0.197, -0.026] |
| fr1_room | edge_tgt3_causal_adjacent_w4 | -2.2 [-5.3, 0.0] | no | -1.7 [-3.7, 1.0] | no | 59 [43, 80] | -0.095 [-0.186, -0.004] |
| fr1_room | edge_src2_tgt3_causal_adjacent_w4 | -2.0 [-5.2, 0.2] | no | -1.6 [-3.6, 0.9] | no | 59 [43, 80] | -0.083 [-0.172, 0.004] |
| fr1_room | gt_ceiling_causal_adjacent_w1_noself | -77.4 [-80.7, -74.8] | yes | -77.9 [-81.9, -76.2] | yes | 100 [100, 100] | -3.637 [-4.921, -2.409] |
| fr1_room | gt_ceiling_causal_diverse_w4_noself_src2_tgt3 | -70.4 [-74.5, -66.3] | yes | -70.1 [-76.6, -65.6] | yes | 100 [100, 100] | -3.273 [-4.294, -2.215] |
| fr1_360 | rungb_causal_diverse_w4 | 4.0 [-11.8, 27.3] | no | -6.8 [-9.6, -2.4] | yes | 72 [57, 91] | -0.363 [-0.675, -0.088] |
| fr1_360 | edge_src2_causal_diverse_w4 | 7.6 [-8.9, 30.2] | no | -2.5 [-7.8, 1.5] | no | 58 [42, 81] | -0.173 [-0.516, 0.083] |
| fr1_360 | edge_src5_causal_diverse_w4 | 8.8 [-8.1, 31.0] | no | -0.9 [-7.5, 4.4] | no | 54 [38, 79] | -0.103 [-0.470, 0.162] |
| fr1_360 | edge_tgt3_causal_diverse_w4 | 7.3 [-8.0, 29.7] | no | -1.9 [-7.0, 1.4] | no | 60 [43, 82] | -0.168 [-0.456, 0.085] |
| fr1_360 | edge_src2_tgt3_causal_diverse_w4 | 8.1 [-7.5, 30.7] | no | -1.0 [-6.9, 3.1] | no | 56 [37, 81] | -0.128 [-0.430, 0.133] |
| fr1_360 | rungb_centred_adjacent_w4 | -6.9 [-9.0, -4.5] | yes | -5.5 [-8.2, -4.0] | yes | 80 [73, 86] | -0.340 [-0.461, -0.234] |
| fr1_360 | edge_src2_centred_adjacent_w4 | -3.4 [-7.1, -0.5] | yes | -3.3 [-6.4, -0.9] | yes | 69 [59, 80] | -0.157 [-0.329, -0.034] |
| fr1_360 | edge_src5_centred_adjacent_w4 | -3.3 [-6.9, -0.4] | yes | -2.5 [-6.6, -0.3] | yes | 67 [54, 80] | -0.151 [-0.334, -0.026] |
| fr1_360 | edge_tgt3_centred_adjacent_w4 | -4.6 [-7.0, -2.0] | yes | -3.4 [-6.3, -2.0] | yes | 72 [64, 80] | -0.217 [-0.329, -0.122] |
| fr1_360 | edge_src2_tgt3_centred_adjacent_w4 | -4.5 [-6.8, -1.9] | yes | -3.3 [-5.9, -2.0] | yes | 72 [64, 80] | -0.209 [-0.313, -0.119] |
| fr1_360 | rungb_causal_adjacent_w4 | -2.6 [-8.9, 5.1] | no | -6.9 [-8.9, -3.1] | yes | 75 [64, 88] | -0.307 [-0.495, -0.151] |
| fr1_360 | edge_src2_causal_adjacent_w4 | 0.1 [-6.7, 7.0] | no | -3.8 [-6.6, -1.5] | yes | 65 [54, 80] | -0.163 [-0.349, -0.030] |
| fr1_360 | edge_src5_causal_adjacent_w4 | 0.2 [-6.2, 7.0] | no | -4.0 [-5.5, -1.5] | yes | 67 [56, 82] | -0.151 [-0.313, -0.025] |
| fr1_360 | edge_tgt3_causal_adjacent_w4 | 0.4 [-5.8, 7.3] | no | -3.3 [-4.7, -1.3] | yes | 67 [56, 81] | -0.134 [-0.286, -0.007] |
| fr1_360 | edge_src2_tgt3_causal_adjacent_w4 | 0.6 [-5.7, 7.4] | no | -2.9 [-4.5, -1.2] | yes | 67 [56, 81] | -0.127 [-0.280, -0.000] |
| fr1_360 | gt_ceiling_causal_adjacent_w1_noself | -72.3 [-75.8, -67.1] | yes | -75.2 [-77.2, -72.0] | yes | 100 [100, 100] | -3.604 [-4.324, -2.967] |
| fr1_360 | gt_ceiling_causal_diverse_w4_noself_src2_tgt3 | -52.2 [-73.0, -26.2] | yes | -63.6 [-79.9, -53.5] | yes | 93 [81, 100] | -3.116 [-4.016, -2.311] |
| fr3 | rungb_causal_diverse_w4 | -5.1 [-11.5, -0.8] | yes | -6.6 [-13.1, -0.6] | yes | 68 [52, 92] | -0.271 [-0.554, -0.092] |
| fr3 | edge_src2_causal_diverse_w4 | 4.3 [-2.0, 9.3] | no | 2.9 [-4.6, 6.5] | no | 39 [23, 67] | 0.102 [-0.138, 0.285] |
| fr3 | edge_src5_causal_diverse_w4 | 3.9 [-1.9, 8.5] | no | 2.7 [-4.2, 7.2] | no | 38 [22, 66] | 0.092 [-0.122, 0.258] |
| fr3 | edge_tgt3_causal_diverse_w4 | 1.3 [-3.9, 5.0] | no | 0.8 [-5.3, 5.1] | no | 45 [29, 72] | -0.009 [-0.235, 0.142] |
| fr3 | edge_src2_tgt3_causal_diverse_w4 | 1.2 [-3.9, 4.9] | no | 0.9 [-5.1, 5.1] | no | 45 [29, 72] | -0.011 [-0.234, 0.136] |
| fr3 | rungb_centred_adjacent_w4 | -6.0 [-10.2, -3.0] | yes | -5.2 [-7.5, -4.0] | yes | 75 [67, 84] | -0.291 [-0.537, -0.140] |
| fr3 | edge_src2_centred_adjacent_w4 | 1.9 [-0.4, 3.5] | no | 2.6 [1.4, 5.1] | yes | 29 [20, 38] | 0.020 [-0.142, 0.112] |
| fr3 | edge_src5_centred_adjacent_w4 | -1.5 [-4.4, 0.3] | no | -0.3 [-0.9, 0.3] | no | 53 [48, 60] | -0.112 [-0.313, 0.004] |
| fr3 | edge_tgt3_centred_adjacent_w4 | -1.8 [-5.3, 0.3] | no | -0.3 [-1.0, 0.1] | no | 55 [50, 62] | -0.126 [-0.349, 0.005] |
| fr3 | edge_src2_tgt3_centred_adjacent_w4 | -1.9 [-5.3, 0.3] | no | -0.3 [-1.0, 0.1] | no | 55 [50, 62] | -0.127 [-0.350, 0.004] |
| fr3 | rungb_causal_adjacent_w4 | -5.4 [-9.7, -3.1] | yes | -5.5 [-7.5, -2.6] | yes | 75 [69, 88] | -0.270 [-0.495, -0.157] |
| fr3 | edge_src2_causal_adjacent_w4 | 0.5 [-2.4, 2.4] | no | 0.9 [-1.3, 2.4] | no | 41 [30, 59] | -0.036 [-0.192, 0.055] |
| fr3 | edge_src5_causal_adjacent_w4 | -0.4 [-3.1, 1.3] | no | 0.4 [-1.7, 1.5] | no | 47 [36, 66] | -0.067 [-0.224, 0.019] |
| fr3 | edge_tgt3_causal_adjacent_w4 | -0.5 [-3.7, 1.4] | no | 0.2 [-1.7, 1.3] | no | 49 [38, 66] | -0.074 [-0.251, 0.021] |
| fr3 | edge_src2_tgt3_causal_adjacent_w4 | -0.6 [-3.7, 1.4] | no | 0.1 [-1.8, 1.2] | no | 48 [37, 66] | -0.074 [-0.250, 0.020] |
| fr3 | gt_ceiling_causal_adjacent_w1_noself | -71.4 [-75.8, -66.5] | yes | -74.6 [-78.0, -71.3] | yes | 100 [100, 100] | -2.935 [-3.408, -2.708] |
| fr3 | gt_ceiling_causal_diverse_w4_noself_src2_tgt3 | -86.8 [-88.7, -85.0] | yes | -88.3 [-89.8, -86.5] | yes | 100 [100, 100] | -3.559 [-4.001, -3.318] |

## Joint solved scale

| Sequence | Config | Mean change % [CI] | Mean robust | Median change % [CI] | Median robust | Improved % [CI] | Mean abs diff [CI] |
|---|---|---|---|---|---|---|---|
| fr1_desk | joint_causal_adjacent_w4 | -2.4 [-7.7, 4.2] | no | -4.5 [-8.1, 4.5] | no | 64 [47, 78] | -0.129 [-0.409, 0.186] |
| fr1_desk | joint_causal_diverse_w4 | 5.4 [-4.5, 17.5] | no | -0.3 [-7.0, 11.1] | no | 51 [33, 67] | 0.206 [-0.341, 0.851] |
| fr1_desk | joint_centred_adjacent_w4 | -1.8 [-4.7, 2.7] | no | -2.6 [-5.4, -0.9] | yes | 65 [55, 73] | -0.139 [-0.314, 0.089] |
| fr1_desk | nofusion_joint_causal_adjacent_w4 | 0.0 [-0.0, 0.0] | no | 0.0 [0.0, 0.0] | no | 24 [19, 33] | -0.000 [-0.000, 0.000] |
| fr1_desk | shuffled_poses_joint_causal_adjacent_w4 | 145.8 [82.0, 195.6] | yes | 107.3 [54.9, 196.9] | yes | 3 [0, 8] | 6.025 [4.249, 7.430] |
| fr1_room | joint_causal_adjacent_w4 | -4.5 [-7.9, -1.5] | yes | -2.3 [-6.5, -0.6] | yes | 74 [58, 89] | -0.256 [-0.520, -0.060] |
| fr1_room | joint_causal_diverse_w4 | -10.7 [-15.7, -5.1] | yes | -8.6 [-16.8, -3.8] | yes | 83 [67, 95] | -0.540 [-0.879, -0.223] |
| fr1_room | joint_centred_adjacent_w4 | -2.7 [-4.8, -0.6] | yes | -1.8 [-4.0, -0.3] | yes | 68 [52, 81] | -0.173 [-0.341, -0.026] |
| fr1_room | nofusion_joint_causal_adjacent_w4 | 0.0 [-0.0, 0.0] | no | 0.0 [0.0, 0.0] | no | 23 [15, 31] | 0.000 [-0.000, 0.000] |
| fr1_room | shuffled_poses_joint_causal_adjacent_w4 | 57.8 [31.6, 79.0] | yes | 46.4 [22.1, 70.9] | yes | 10 [4, 19] | 2.609 [1.112, 4.295] |
| fr1_360 | joint_causal_adjacent_w4 | -5.9 [-11.6, -1.0] | yes | -5.8 [-11.3, -3.3] | yes | 76 [66, 87] | -0.398 [-0.690, -0.160] |
| fr1_360 | joint_causal_diverse_w4 | 3.2 [-12.2, 16.1] | no | -1.9 [-13.3, 12.6] | no | 54 [27, 80] | -0.013 [-0.718, 0.811] |
| fr1_360 | joint_centred_adjacent_w4 | -1.0 [-3.6, 2.5] | no | -0.6 [-3.8, 1.5] | no | 54 [36, 69] | -0.139 [-0.253, 0.000] |
| fr1_360 | nofusion_joint_causal_adjacent_w4 | -0.0 [-0.0, 0.0] | no | 0.0 [0.0, 0.0] | no | 23 [16, 32] | -0.000 [-0.000, 0.000] |
| fr1_360 | shuffled_poses_joint_causal_adjacent_w4 | 76.9 [21.9, 109.4] | yes | 42.0 [14.4, 70.9] | yes | 7 [2, 17] | 2.897 [1.199, 4.473] |
| fr3 | joint_causal_adjacent_w4 | -4.3 [-8.7, -1.6] | yes | -3.8 [-6.9, -2.0] | yes | 71 [61, 89] | -0.220 [-0.452, -0.089] |
| fr3 | joint_causal_diverse_w4 | -3.2 [-9.2, 0.8] | no | -4.1 [-8.6, 0.4] | no | 64 [49, 88] | -0.185 [-0.453, -0.019] |
| fr3 | joint_centred_adjacent_w4 | -6.0 [-10.2, -3.2] | yes | -5.2 [-6.7, -3.9] | yes | 79 [71, 89] | -0.295 [-0.541, -0.150] |
| fr3 | nofusion_joint_causal_adjacent_w4 | 0.0 [0.0, 0.0] | yes | 0.0 [0.0, 0.0] | no | 24 [17, 34] | 0.000 [0.000, 0.000] |
| fr3 | shuffled_poses_joint_causal_adjacent_w4 | 193.1 [146.0, 243.0] | yes | 147.3 [123.2, 206.7] | yes | 1 [0, 3] | 7.550 [6.203, 9.201] |
