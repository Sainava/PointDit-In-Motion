# Sampler initialisation comparison

## Runs

| run | init | seed | seed mode |
|---|---|---|---|
| zeros_rerun | zeros | None | per_call |
| noise_seed0 | noise | 0 | per_frame |
| noise_seed1 | noise | 1 | per_frame |
| noise_seed2 | noise | 2 | per_frame |

| sequence | frames |
|---|---|
| rgbd_dataset_freiburg1_360 | 200-299 |
| rgbd_dataset_freiburg3_long_office_household | 740-839 |

## Seed disagreement (x100, per-frame median relative difference)

| sequence | comparison (reference-aligned) | frames | median of medians [95% CI] | median of p90 | max of medians |
|---|---|---|---|---|---|
| rgbd_dataset_freiburg1_360 | noise_seed0-noise_seed1 | 100 | 0.333 [0.304, 0.404] | 1.324 | 0.682 |
| rgbd_dataset_freiburg1_360 | noise_seed0-noise_seed2 | 100 | 0.330 [0.282, 0.421] | 1.382 | 0.768 |
| rgbd_dataset_freiburg1_360 | noise_seed1-noise_seed2 | 100 | 0.311 [0.283, 0.430] | 1.344 | 0.880 |
| rgbd_dataset_freiburg1_360 | zeros_rerun-noise_seed0 | 100 | 0.262 [0.243, 0.300] | 0.973 | 0.592 |
| rgbd_dataset_freiburg1_360 | zeros_rerun-noise_seed1 | 100 | 0.248 [0.221, 0.292] | 0.952 | 0.508 |
| rgbd_dataset_freiburg1_360 | zeros_rerun-noise_seed2 | 100 | 0.254 [0.205, 0.343] | 1.007 | 0.600 |
| rgbd_dataset_freiburg3_long_office_household | noise_seed0-noise_seed1 | 100 | 0.285 [0.249, 0.328] | 0.940 | 0.693 |
| rgbd_dataset_freiburg3_long_office_household | noise_seed0-noise_seed2 | 100 | 0.265 [0.249, 0.284] | 0.958 | 0.534 |
| rgbd_dataset_freiburg3_long_office_household | noise_seed1-noise_seed2 | 100 | 0.287 [0.260, 0.316] | 0.955 | 0.535 |
| rgbd_dataset_freiburg3_long_office_household | zeros_rerun-noise_seed0 | 100 | 0.214 [0.198, 0.233] | 0.685 | 0.459 |
| rgbd_dataset_freiburg3_long_office_household | zeros_rerun-noise_seed1 | 100 | 0.205 [0.177, 0.232] | 0.697 | 0.507 |
| rgbd_dataset_freiburg3_long_office_household | zeros_rerun-noise_seed2 | 100 | 0.208 [0.180, 0.243] | 0.718 | 0.473 |

## Accuracy: median per-frame difference {noise} - zeros_rerun [95% CI]

| sequence | metric | zeros_rerun median | noise_seed0 | noise_seed1 | noise_seed2 |
|---|---|---|---|---|---|
| rgbd_dataset_freiburg1_360 | rel_d | 6.887 | 0.013 [-0.004, 0.045] | 0.017 [-0.018, 0.060] | 0.017 [-0.024, 0.043] |
| rgbd_dataset_freiburg1_360 | rel_p | 7.740 | 0.024 [-0.002, 0.029] | 0.015 [-0.010, 0.049] | 0.015 [-0.019, 0.042] |
| rgbd_dataset_freiburg1_360 | delta1_d | 94.422 | -0.023 [-0.054, -0.007] | -0.011 [-0.057, 0.006] | -0.017 [-0.075, 0.010] |
| rgbd_dataset_freiburg3_long_office_household | rel_d | 4.041 | 0.033 [0.019, 0.045] | 0.016 [0.003, 0.030] | 0.027 [0.020, 0.041] |
| rgbd_dataset_freiburg3_long_office_household | rel_p | 4.915 | 0.041 [0.020, 0.054] | 0.018 [0.002, 0.039] | 0.027 [0.008, 0.039] |
| rgbd_dataset_freiburg3_long_office_household | delta1_d | 98.041 | 0.002 [-0.006, 0.008] | 0.008 [-0.005, 0.020] | -0.010 [-0.021, -0.004] |

## Consistency: median over pairs of per-pair median x100, visible pixels [95% CI]

| sequence | gap (ms) | run | pairs | M1 | M2 | M2 / M2(zeros_rerun) |
|---|---|---|---|---|---|---|
| rgbd_dataset_freiburg1_360 | 33 | zeros_rerun | 99 | 0.914 [0.762, 1.100] | 0.657 [0.631, 0.748] | - |
| rgbd_dataset_freiburg1_360 | 33 | noise_seed0 | 99 | 0.884 [0.796, 1.171] | 0.750 [0.652, 0.908] | 1.141 [1.033, 1.262] |
| rgbd_dataset_freiburg1_360 | 33 | noise_seed1 | 99 | 0.955 [0.884, 1.142] | 0.724 [0.686, 0.844] | 1.101 [1.051, 1.186] |
| rgbd_dataset_freiburg1_360 | 33 | noise_seed2 | 99 | 0.963 [0.863, 1.275] | 0.800 [0.706, 0.945] | 1.217 [1.115, 1.316] |
| rgbd_dataset_freiburg1_360 | 100 | zeros_rerun | 97 | 1.500 [1.243, 1.829] | 1.059 [0.923, 1.249] | - |
| rgbd_dataset_freiburg1_360 | 100 | noise_seed0 | 97 | 1.509 [1.417, 1.921] | 1.098 [0.979, 1.327] | 1.037 [0.957, 1.119] |
| rgbd_dataset_freiburg1_360 | 100 | noise_seed1 | 97 | 1.582 [1.369, 1.963] | 1.120 [0.979, 1.308] | 1.058 [0.960, 1.123] |
| rgbd_dataset_freiburg1_360 | 100 | noise_seed2 | 97 | 1.580 [1.378, 1.829] | 1.121 [0.991, 1.298] | 1.059 [0.968, 1.102] |
| rgbd_dataset_freiburg1_360 | 333 | zeros_rerun | 90 | 3.259 [2.466, 4.000] | 2.046 [1.727, 2.344] | - |
| rgbd_dataset_freiburg1_360 | 333 | noise_seed0 | 90 | 3.197 [2.523, 4.138] | 2.039 [1.768, 2.296] | 0.997 [0.943, 1.064] |
| rgbd_dataset_freiburg1_360 | 333 | noise_seed1 | 90 | 3.194 [2.428, 4.225] | 2.092 [1.788, 2.348] | 1.022 [0.954, 1.089] |
| rgbd_dataset_freiburg1_360 | 333 | noise_seed2 | 90 | 3.388 [2.502, 4.094] | 2.034 [1.742, 2.373] | 0.994 [0.941, 1.052] |
| rgbd_dataset_freiburg3_long_office_household | 33 | zeros_rerun | 95 | 0.544 [0.494, 0.673] | 0.419 [0.389, 0.518] | - |
| rgbd_dataset_freiburg3_long_office_household | 33 | noise_seed0 | 95 | 0.618 [0.562, 0.744] | 0.488 [0.424, 0.565] | 1.167 [1.035, 1.237] |
| rgbd_dataset_freiburg3_long_office_household | 33 | noise_seed1 | 95 | 0.604 [0.559, 0.763] | 0.495 [0.464, 0.597] | 1.182 [1.098, 1.251] |
| rgbd_dataset_freiburg3_long_office_household | 33 | noise_seed2 | 95 | 0.611 [0.546, 0.747] | 0.478 [0.435, 0.522] | 1.142 [1.020, 1.210] |
| rgbd_dataset_freiburg3_long_office_household | 100 | zeros_rerun | 93 | 0.657 [0.616, 0.740] | 0.561 [0.517, 0.620] | - |
| rgbd_dataset_freiburg3_long_office_household | 100 | noise_seed0 | 93 | 0.723 [0.655, 0.845] | 0.603 [0.531, 0.650] | 1.074 [0.991, 1.113] |
| rgbd_dataset_freiburg3_long_office_household | 100 | noise_seed1 | 93 | 0.713 [0.656, 0.860] | 0.615 [0.552, 0.681] | 1.095 [1.019, 1.168] |
| rgbd_dataset_freiburg3_long_office_household | 100 | noise_seed2 | 93 | 0.774 [0.682, 0.874] | 0.609 [0.547, 0.670] | 1.085 [1.016, 1.183] |
| rgbd_dataset_freiburg3_long_office_household | 333 | zeros_rerun | 87 | 0.947 [0.822, 1.077] | 0.803 [0.638, 0.917] | - |
| rgbd_dataset_freiburg3_long_office_household | 333 | noise_seed0 | 87 | 1.014 [0.905, 1.085] | 0.795 [0.627, 0.966] | 0.990 [0.967, 1.116] |
| rgbd_dataset_freiburg3_long_office_household | 333 | noise_seed1 | 87 | 1.004 [0.871, 1.161] | 0.815 [0.687, 0.974] | 1.015 [1.003, 1.171] |
| rgbd_dataset_freiburg3_long_office_household | 333 | noise_seed2 | 87 | 1.012 [0.889, 1.130] | 0.783 [0.659, 0.992] | 0.975 [0.939, 1.136] |

## Regression: zero-init subset pairs vs the full baseline run

| sequence | pairs | baseline pairs in subset | max |diff| | failures |
|---|---|---|---|---|
| rgbd_dataset_freiburg1_360 | 286 | 286 | 0.00e+00 | 0 |
| rgbd_dataset_freiburg3_long_office_household | 275 | 275 | 0.00e+00 | 0 |
