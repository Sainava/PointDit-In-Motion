# Analysis summary

Inputs: `results/eval/baseline/per_frame.csv`, `results/covariates/covariates.csv`, `results/eval/baseline/focal_split.csv`.

Spearman rho with average ranks. Partial rho: correlation of rank residuals after linear regression on the controls' ranks. pooled_fr1: ranks and residuals computed within each fr1 sequence, then pooled; for fast vs slow, the mean of the per-sequence differences. 95% percentile CIs from a moving-block bootstrap within each sequence (block 15 frames, 2000 resamples, seed 20261010; subsets are applied after resampling). Fast / slow: fastest / slowest third of frames by angular_speed_dps. Subsets with n < 40 are skipped.

## Q1 Motion and blur

| unit | y | x | subset | controls | n | rho | 95% CI |
|---|---|---|---|---|---|---|---|
| fr1_desk | rel_d | angular_speed_dps | all | - | 300 | 0.153 | [-0.056, 0.412] |
| fr1_room | rel_d | angular_speed_dps | all | - | 300 | -0.047 | [-0.372, 0.335] |
| fr1_360 | rel_d | angular_speed_dps | all | - | 300 | 0.344 | [0.068, 0.578] |
| fr3_long_office_household | rel_d | angular_speed_dps | all | - | 300 | 0.181 | [-0.048, 0.420] |
| pooled_fr1 | rel_d | angular_speed_dps | all | - | 900 | 0.150 | [-0.006, 0.329] |
| fr1_desk | rel_d | sharpness_rel | all | - | 300 | -0.225 | [-0.398, -0.059] |
| fr1_room | rel_d | sharpness_rel | all | - | 300 | -0.213 | [-0.408, 0.019] |
| fr1_360 | rel_d | sharpness_rel | all | - | 300 | -0.057 | [-0.270, 0.140] |
| fr3_long_office_household | rel_d | sharpness_rel | all | - | 300 | -0.276 | [-0.419, -0.110] |
| pooled_fr1 | rel_d | sharpness_rel | all | - | 900 | -0.165 | [-0.278, -0.055] |

| unit | subset | n | n per third | median rel_d fast | median rel_d slow | fast − slow | 95% CI |
|---|---|---|---|---|---|---|---|
| fr1_desk | all | 300 | 100 | 4.38 | 3.85 | 0.53 | [-0.37, 1.67] |
| fr1_room | all | 300 | 100 | 5.32 | 6.12 | -0.80 | [-2.74, 2.38] |
| fr1_360 | all | 300 | 100 | 5.70 | 4.29 | 1.41 | [-0.09, 4.35] |
| fr3_long_office_household | all | 300 | 100 | 3.95 | 3.64 | 0.32 | [-0.27, 0.92] |
| pooled_fr1 | all | 900 | 300 | n/a | n/a | 0.38 | [-0.45, 1.93] |

## Q2 Blur vs GT timing

Frames with depth_offset_ms < 5: fr1_desk 32/300, fr1_room 3/300, fr1_360 272/300, fr3_long_office_household 300/300.

- Q2 rel_d vs angular_speed_dps [depth_offset_ms<5], fr1_desk: skipped, n = 32 < 40
- Q2 rel_d vs angular_speed_dps [depth_offset_ms<5], fr1_room: skipped, n = 3 < 40
- Q2 rel_d vs angular_speed_dps [depth_offset_ms<5], pooled_fr1: skipped, fewer than 2 fr1 sequences have n >= 40
- Q2 rel_d vs sharpness_rel [depth_offset_ms<5], fr1_desk: skipped, n = 32 < 40
- Q2 rel_d vs sharpness_rel [depth_offset_ms<5], fr1_room: skipped, n = 3 < 40
- Q2 rel_d vs sharpness_rel [depth_offset_ms<5], pooled_fr1: skipped, fewer than 2 fr1 sequences have n >= 40
- Q2 fast vs slow third [depth_offset_ms<5], fr1_desk: skipped, n = 32 < 40
- Q2 fast vs slow third [depth_offset_ms<5], fr1_room: skipped, n = 3 < 40
- Q2 fast vs slow third [depth_offset_ms<5], pooled_fr1: skipped, fewer than 2 fr1 sequences have n >= 40

| unit | y | x | subset | controls | n | rho | 95% CI |
|---|---|---|---|---|---|---|---|
| fr1_360 | rel_d | angular_speed_dps | depth_offset_ms<5 | - | 272 | 0.325 | [0.039, 0.564] |
| fr3_long_office_household | rel_d | angular_speed_dps | depth_offset_ms<5 | - | 300 | 0.181 | [-0.048, 0.420] |
| fr1_360 | rel_d | sharpness_rel | depth_offset_ms<5 | - | 272 | -0.033 | [-0.253, 0.169] |
| fr3_long_office_household | rel_d | sharpness_rel | depth_offset_ms<5 | - | 300 | -0.276 | [-0.419, -0.110] |
| fr1_desk | rel_d | sharpness_rel | all | expected_misalignment_px edge_density | 300 | -0.294 | [-0.424, -0.170] |
| fr1_room | rel_d | sharpness_rel | all | expected_misalignment_px edge_density | 300 | -0.275 | [-0.421, -0.085] |
| fr1_360 | rel_d | sharpness_rel | all | expected_misalignment_px edge_density | 300 | -0.229 | [-0.449, -0.050] |
| fr3_long_office_household | rel_d | sharpness_rel | all | expected_misalignment_px edge_density | 300 | -0.274 | [-0.391, -0.107] |
| pooled_fr1 | rel_d | sharpness_rel | all | expected_misalignment_px edge_density | 900 | -0.264 | [-0.357, -0.179] |

| unit | subset | n | n per third | median rel_d fast | median rel_d slow | fast − slow | 95% CI |
|---|---|---|---|---|---|---|---|
| fr1_360 | depth_offset_ms<5 | 272 | 90 | 5.68 | 4.30 | 1.38 | [-0.22, 4.35] |
| fr3_long_office_household | depth_offset_ms<5 | 300 | 100 | 3.95 | 3.64 | 0.32 | [-0.27, 0.92] |

## Q3 Scene content

| unit | y | x | subset | controls | n | rho | 95% CI |
|---|---|---|---|---|---|---|---|
| fr1_desk | rel_d | edge_density | all | - | 300 | 0.418 | [0.032, 0.658] |
| fr1_room | rel_d | edge_density | all | - | 300 | 0.568 | [0.228, 0.745] |
| fr1_360 | rel_d | edge_density | all | - | 300 | 0.861 | [0.720, 0.920] |
| fr3_long_office_household | rel_d | edge_density | all | - | 300 | 0.524 | [0.120, 0.763] |
| pooled_fr1 | rel_d | edge_density | all | - | 900 | 0.616 | [0.436, 0.708] |
| fr1_desk | rel_d | depth_range_ratio | all | - | 300 | 0.380 | [0.022, 0.655] |
| fr1_room | rel_d | depth_range_ratio | all | - | 300 | 0.512 | [0.094, 0.777] |
| fr1_360 | rel_d | depth_range_ratio | all | - | 300 | 0.730 | [0.516, 0.870] |
| fr3_long_office_household | rel_d | depth_range_ratio | all | - | 300 | 0.547 | [0.149, 0.821] |
| pooled_fr1 | rel_d | depth_range_ratio | all | - | 900 | 0.541 | [0.347, 0.684] |
| fr1_desk | rel_d | near_border_fraction | all | - | 300 | 0.119 | [-0.127, 0.352] |
| fr1_room | rel_d | near_border_fraction | all | - | 300 | 0.300 | [-0.140, 0.641] |
| fr1_360 | rel_d | near_border_fraction | all | - | 300 | 0.335 | [0.041, 0.602] |
| fr3_long_office_household | rel_d | near_border_fraction | all | - | 300 | 0.279 | [-0.065, 0.541] |
| pooled_fr1 | rel_d | near_border_fraction | all | - | 900 | 0.260 | [0.058, 0.433] |
| fr1_desk | rel_d_near | near_border_fraction | all | - | 300 | 0.217 | [-0.028, 0.449] |
| fr1_room | rel_d_near | near_border_fraction | all | - | 300 | 0.206 | [-0.268, 0.587] |
| fr1_360 | rel_d_near | near_border_fraction | all | - | 300 | 0.345 | [0.023, 0.617] |
| fr3_long_office_household | rel_d_near | near_border_fraction | all | - | 300 | -0.086 | [-0.466, 0.319] |
| pooled_fr1 | rel_d_near | near_border_fraction | all | - | 900 | 0.256 | [0.041, 0.438] |

## Q4 Implied focal length

From `results/eval/baseline/focal_split.csv`. Frames with focal_split_logdiff < ln(1.1): fr1_desk 35/300, fr1_room 9/300, fr1_360 51/300, fr3_long_office_household 26/300.

- Q4 rel_d vs abs_log_focal_ratio [focal_split_logdiff<ln(1.1)], fr1_desk: skipped, n = 35 < 40
- Q4 rel_d vs abs_log_focal_ratio [focal_split_logdiff<ln(1.1)], fr1_room: skipped, n = 9 < 40
- Q4 rel_d vs abs_log_focal_ratio [focal_split_logdiff<ln(1.1)], fr3_long_office_household: skipped, n = 26 < 40
- Q4 rel_d vs abs_log_focal_ratio [focal_split_logdiff<ln(1.1)], pooled_fr1: skipped, fewer than 2 fr1 sequences have n >= 40

| unit | y | x | subset | controls | n | rho | 95% CI |
|---|---|---|---|---|---|---|---|
| fr1_360 | rel_d | abs_log_focal_ratio | focal_split_logdiff<ln(1.1) | - | 51 | -0.907 | [-0.936, -0.646] |

## Q5 Temporal spikes

Spike: rel_d exceeds the median rel_d of the frames within ±5 positions (itself excluded) by more than max(1.5 points, 50%). Largest increase: the depth third whose rel_d exceeds its own neighbour median the most. Differences use spike labels fixed from the original series.

| sequence | frames | spikes | rate | largest increase near / mid / far | spike frames |
|---|---|---|---|---|---|
| fr1_desk | 300 | 2 | 0.7% | 0 / 1 / 1 | 337, 338 |
| fr1_room | 300 | 0 | 0.0% | 0 / 0 / 0 | - |
| fr1_360 | 300 | 1 | 0.3% | 1 / 0 / 0 | 134 |
| fr3_long_office_household | 300 | 2 | 0.7% | 0 / 0 / 2 | 758, 761 |

| sequence | covariate | median at spikes | median elsewhere | spikes − elsewhere | 95% CI |
|---|---|---|---|---|---|
| fr1_desk | angular_speed_dps | 75.379 | 27.526 | 47.853 | [37.487, 53.917] |
| fr1_desk | sharpness_rel | 0.694 | 1.000 | -0.306 | [-0.459, -0.161] |
| fr1_room | angular_speed_dps | n/a | 34.869 | n/a | [n/a, n/a] |
| fr1_room | sharpness_rel | n/a | 1.014 | n/a | [n/a, n/a] |
| fr1_360 | angular_speed_dps | 75.087 | 53.348 | 21.739 | [16.721, 27.243] |
| fr1_360 | sharpness_rel | 1.071 | 0.992 | 0.079 | [0.025, 0.134] |
| fr3_long_office_household | angular_speed_dps | 8.568 | 14.322 | -5.754 | [-8.392, -3.670] |
| fr3_long_office_household | sharpness_rel | 0.989 | 0.999 | -0.010 | [-0.199, 0.183] |

## Robustness to block length

CI position relative to zero at block lengths 8, 15, 30: 10 of 55 estimates differ from block 15 at block 8 or 30.

| estimate | identity | block 8 | block 15 | block 30 |
|---|---|---|---|---|
| correlation | Q1, pooled_fr1, rel_d, angular_speed_dps, all | > 0 | spans 0 | > 0 |
| correlation | Q1, fr1_room, rel_d, sharpness_rel, all | < 0 | spans 0 | spans 0 |
| correlation | Q3, fr1_desk, rel_d, edge_density, all | > 0 | > 0 | spans 0 |
| correlation | Q3, fr3_long_office_household, rel_d, edge_density, all | > 0 | > 0 | spans 0 |
| correlation | Q3, fr1_desk, rel_d, depth_range_ratio, all | > 0 | > 0 | spans 0 |
| correlation | Q3, fr1_room, rel_d, depth_range_ratio, all | > 0 | > 0 | spans 0 |
| correlation | Q3, fr1_360, rel_d, near_border_fraction, all | > 0 | > 0 | spans 0 |
| correlation | Q3, fr3_long_office_household, rel_d, near_border_fraction, all | > 0 | spans 0 | spans 0 |
| correlation | Q3, fr1_desk, rel_d_near, near_border_fraction, all | > 0 | spans 0 | spans 0 |
| fast − slow | Q1, fr1_360, all | spans 0 | spans 0 | > 0 |

