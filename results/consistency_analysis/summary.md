# Consistency analysis summary

Inputs: `results/consistency/baseline`, `results/covariates/covariates.csv`, `results/eval/baseline/per_frame.csv`.

Per-pair values are each pair's median over visible pixels (x100). Median over pairs with 95% percentile CIs from a moving-block bootstrap over pairs ordered by frame i, within each sequence and gap (block 15 shown, 2000 resamples, seed 20261013). 'All 8/15/30': the CI lies on the same side of the reference value (1 for ratios, 0 for correlations) at block lengths 8/15/30. Ratio: median of per-pair M2 / M2 floor. pooled_fr1: ranks and residuals within each fr1 sequence, then pooled.

## 1. Headline

| sequence | gap ms | M2 | M2 floor | M1 | GT floor | M2 / M2 floor | ratio CI excl. 1, all 8/15/30 |
|---|---|---|---|---|---|---|---|
| fr1_360 | 33 | 0.56 [0.48, 0.67] | 0.37 [0.32, 0.43] | 0.78 [0.65, 0.92] | 0.40 [0.35, 0.45] | 1.50 [1.34, 1.68] | yes |
| fr1_360 | 100 | 0.91 [0.82, 1.07] | 0.49 [0.40, 0.56] | 1.24 [1.11, 1.56] | 0.63 [0.52, 0.69] | 1.77 [1.43, 2.10] | yes |
| fr1_360 | 333 | 1.64 [1.28, 1.92] | 0.71 [0.56, 0.87] | 2.58 [2.13, 3.25] | 1.34 [0.98, 1.62] | 1.92 [1.59, 2.41] | yes |
| fr1_desk | 33 | 0.48 [0.38, 0.67] | 0.34 [0.28, 0.42] | 0.79 [0.65, 0.90] | 0.39 [0.37, 0.46] | 1.42 [1.26, 1.53] | yes |
| fr1_desk | 100 | 0.72 [0.58, 0.88] | 0.45 [0.37, 0.55] | 1.05 [0.89, 1.36] | 0.51 [0.45, 0.61] | 1.51 [1.33, 1.67] | yes |
| fr1_desk | 333 | 1.24 [1.04, 1.48] | 0.62 [0.53, 0.81] | 2.05 [1.64, 2.32] | 0.96 [0.80, 1.22] | 1.80 [1.50, 2.11] | yes |
| fr1_room | 33 | 0.61 [0.44, 0.73] | 0.33 [0.31, 0.35] | 0.82 [0.57, 0.99] | 0.32 [0.30, 0.34] | 1.64 [1.33, 2.09] | yes |
| fr1_room | 100 | 0.76 [0.60, 0.89] | 0.46 [0.42, 0.48] | 1.07 [0.87, 1.31] | 0.46 [0.41, 0.49] | 1.59 [1.34, 1.92] | yes |
| fr1_room | 333 | 1.20 [0.95, 1.41] | 0.69 [0.59, 0.75] | 2.02 [1.71, 2.27] | 0.78 [0.64, 0.88] | 1.57 [1.35, 2.08] | yes |
| fr3 | 33 | 0.34 [0.29, 0.40] | 0.25 [0.23, 0.27] | 0.44 [0.39, 0.51] | 0.26 [0.24, 0.29] | 1.34 [1.18, 1.47] | yes |
| fr3 | 100 | 0.46 [0.38, 0.52] | 0.28 [0.26, 0.30] | 0.57 [0.53, 0.62] | 0.30 [0.28, 0.32] | 1.50 [1.26, 1.67] | yes |
| fr3 | 333 | 0.65 [0.49, 0.81] | 0.36 [0.34, 0.39] | 1.05 [0.89, 1.19] | 0.41 [0.37, 0.45] | 1.70 [1.42, 1.93] | yes |

## 2. Persistence

Not available: no persistence results.

## 3. Drivers of excess inconsistency

Excess = M2 − M2 floor per pair. Covariates: mean over frames i and j; rel_d_j: per-frame Rel_d of frame j. Partial: controlling for edge_density and expected_misalignment_px.

Gap 33 ms.

| unit | x | n | rho | CI excl. 0, all 8/15/30 |
|---|---|---|---|---|
| fr1_360 | sharpness_rel | 299 | -0.08 [-0.25, 0.04] | no |
| fr1_360 | angular_speed_dps | 299 | 0.13 [-0.05, 0.30] | no |
| fr1_360 | edge_density | 299 | 0.25 [0.05, 0.42] | yes |
| fr1_360 | depth_range_ratio | 299 | 0.28 [0.10, 0.48] | yes |
| fr1_360 | expected_misalignment_px | 299 | -0.18 [-0.35, -0.03] | yes |
| fr1_360 | actual_gap_ms | 299 | -0.00 [-0.12, 0.11] | no |
| fr1_360 | rel_d_j | 299 | 0.30 [0.10, 0.46] | yes |
| fr1_360 | sharpness_rel (partial) | 299 | -0.10 [-0.29, 0.02] | no |
| fr1_desk | sharpness_rel | 299 | -0.04 [-0.21, 0.10] | no |
| fr1_desk | angular_speed_dps | 299 | 0.21 [-0.03, 0.42] | no |
| fr1_desk | edge_density | 299 | 0.17 [-0.05, 0.40] | no |
| fr1_desk | depth_range_ratio | 299 | 0.28 [0.07, 0.42] | yes |
| fr1_desk | expected_misalignment_px | 299 | 0.01 [-0.24, 0.25] | no |
| fr1_desk | actual_gap_ms | 299 | -0.01 [-0.10, 0.08] | no |
| fr1_desk | rel_d_j | 299 | 0.32 [0.13, 0.48] | yes |
| fr1_desk | sharpness_rel (partial) | 299 | -0.05 [-0.22, 0.10] | no |
| fr1_room | sharpness_rel | 299 | -0.04 [-0.25, 0.15] | no |
| fr1_room | angular_speed_dps | 299 | -0.08 [-0.36, 0.25] | no |
| fr1_room | edge_density | 299 | 0.53 [0.23, 0.73] | yes |
| fr1_room | depth_range_ratio | 299 | 0.40 [0.10, 0.61] | yes |
| fr1_room | expected_misalignment_px | 299 | 0.00 [-0.26, 0.26] | no |
| fr1_room | actual_gap_ms | 299 | -0.04 [-0.09, 0.06] | no |
| fr1_room | rel_d_j | 299 | 0.55 [0.37, 0.66] | yes |
| fr1_room | sharpness_rel (partial) | 299 | -0.09 [-0.23, 0.05] | no |
| fr3 | sharpness_rel | 294 | -0.14 [-0.24, 0.02] | no |
| fr3 | angular_speed_dps | 294 | 0.30 [0.09, 0.48] | yes |
| fr3 | edge_density | 294 | 0.40 [0.19, 0.55] | yes |
| fr3 | depth_range_ratio | 294 | 0.36 [0.13, 0.54] | yes |
| fr3 | expected_misalignment_px | 294 | -0.03 [-0.16, 0.14] | no |
| fr3 | actual_gap_ms | 294 | -0.10 [-0.19, -0.02] | yes |
| fr3 | rel_d_j | 294 | 0.21 [-0.02, 0.37] | no |
| fr3 | sharpness_rel (partial) | 294 | -0.11 [-0.22, 0.07] | no |
| pooled_fr1 | sharpness_rel | 897 | -0.05 [-0.17, 0.03] | no |
| pooled_fr1 | angular_speed_dps | 897 | 0.09 [-0.05, 0.22] | no |
| pooled_fr1 | edge_density | 897 | 0.32 [0.18, 0.44] | yes |
| pooled_fr1 | depth_range_ratio | 897 | 0.32 [0.19, 0.44] | yes |
| pooled_fr1 | expected_misalignment_px | 897 | -0.06 [-0.19, 0.08] | no |
| pooled_fr1 | actual_gap_ms | 897 | -0.02 [-0.06, 0.04] | no |
| pooled_fr1 | rel_d_j | 897 | 0.39 [0.27, 0.48] | yes |
| pooled_fr1 | sharpness_rel (partial) | 897 | -0.08 [-0.19, -0.01] | no |

Gap 100 ms.

| unit | x | n | rho | CI excl. 0, all 8/15/30 |
|---|---|---|---|---|
| fr1_360 | sharpness_rel | 297 | -0.16 [-0.34, -0.00] | no |
| fr1_360 | angular_speed_dps | 297 | 0.27 [0.04, 0.45] | yes |
| fr1_360 | edge_density | 297 | 0.12 [-0.11, 0.34] | no |
| fr1_360 | depth_range_ratio | 297 | 0.18 [-0.03, 0.42] | no |
| fr1_360 | expected_misalignment_px | 297 | -0.14 [-0.28, 0.07] | no |
| fr1_360 | actual_gap_ms | 297 | -0.01 [-0.10, 0.06] | no |
| fr1_360 | rel_d_j | 297 | 0.23 [0.01, 0.44] | yes |
| fr1_360 | sharpness_rel (partial) | 297 | -0.17 [-0.36, -0.01] | no |
| fr1_desk | sharpness_rel | 297 | 0.07 [-0.08, 0.22] | no |
| fr1_desk | angular_speed_dps | 297 | 0.20 [0.00, 0.39] | no |
| fr1_desk | edge_density | 297 | 0.21 [0.02, 0.40] | yes |
| fr1_desk | depth_range_ratio | 297 | 0.11 [-0.06, 0.29] | no |
| fr1_desk | expected_misalignment_px | 297 | 0.06 [-0.14, 0.26] | no |
| fr1_desk | actual_gap_ms | 297 | 0.10 [0.01, 0.21] | yes |
| fr1_desk | rel_d_j | 297 | 0.16 [-0.00, 0.29] | no |
| fr1_desk | sharpness_rel (partial) | 297 | 0.06 [-0.08, 0.20] | no |
| fr1_room | sharpness_rel | 297 | 0.04 [-0.18, 0.23] | no |
| fr1_room | angular_speed_dps | 297 | -0.08 [-0.37, 0.23] | no |
| fr1_room | edge_density | 297 | 0.36 [0.04, 0.57] | yes |
| fr1_room | depth_range_ratio | 297 | 0.27 [-0.01, 0.46] | no |
| fr1_room | expected_misalignment_px | 297 | -0.07 [-0.31, 0.21] | no |
| fr1_room | actual_gap_ms | 297 | 0.09 [-0.06, 0.20] | no |
| fr1_room | rel_d_j | 297 | 0.42 [0.21, 0.54] | yes |
| fr1_room | sharpness_rel (partial) | 297 | -0.02 [-0.20, 0.16] | no |
| fr3 | sharpness_rel | 292 | -0.01 [-0.11, 0.12] | no |
| fr3 | angular_speed_dps | 292 | 0.11 [-0.12, 0.30] | no |
| fr3 | edge_density | 292 | 0.22 [-0.05, 0.46] | no |
| fr3 | depth_range_ratio | 292 | 0.21 [-0.10, 0.45] | no |
| fr3 | expected_misalignment_px | 292 | -0.02 [-0.16, 0.11] | no |
| fr3 | actual_gap_ms | 292 | -0.01 [-0.12, 0.08] | no |
| fr3 | rel_d_j | 292 | 0.00 [-0.25, 0.22] | no |
| fr3 | sharpness_rel (partial) | 292 | 0.02 [-0.09, 0.15] | no |
| pooled_fr1 | sharpness_rel | 891 | -0.02 [-0.12, 0.08] | no |
| pooled_fr1 | angular_speed_dps | 891 | 0.13 [-0.01, 0.26] | no |
| pooled_fr1 | edge_density | 891 | 0.23 [0.08, 0.35] | yes |
| pooled_fr1 | depth_range_ratio | 891 | 0.19 [0.06, 0.31] | yes |
| pooled_fr1 | expected_misalignment_px | 891 | -0.05 [-0.16, 0.08] | no |
| pooled_fr1 | actual_gap_ms | 891 | 0.06 [-0.01, 0.12] | no |
| pooled_fr1 | rel_d_j | 891 | 0.27 [0.16, 0.36] | yes |
| pooled_fr1 | sharpness_rel (partial) | 891 | -0.05 [-0.15, 0.04] | no |

## 4. M2 by depth third

| sequence | gap ms | M2 near | M2 mid | M2 far | floor near | floor mid | floor far |
|---|---|---|---|---|---|---|---|
| fr1_360 | 33 | 0.50 [0.41, 0.59] | 0.54 [0.49, 0.62] | 0.61 [0.51, 0.84] | 0.32 | 0.37 | 0.44 |
| fr1_360 | 100 | 0.76 [0.65, 0.93] | 0.89 [0.81, 1.01] | 1.03 [0.88, 1.23] | 0.41 | 0.47 | 0.56 |
| fr1_360 | 333 | 1.48 [1.12, 1.77] | 1.55 [1.25, 1.93] | 1.66 [1.21, 2.09] | 0.60 | 0.66 | 0.84 |
| fr1_desk | 33 | 0.37 [0.32, 0.49] | 0.46 [0.36, 0.64] | 0.64 [0.54, 0.84] | 0.23 | 0.34 | 0.54 |
| fr1_desk | 100 | 0.53 [0.43, 0.68] | 0.72 [0.60, 0.91] | 0.93 [0.77, 1.21] | 0.33 | 0.45 | 0.67 |
| fr1_desk | 333 | 0.99 [0.71, 1.21] | 1.41 [1.14, 1.69] | 1.51 [1.26, 1.92] | 0.48 | 0.66 | 0.85 |
| fr1_room | 33 | 0.54 [0.38, 0.70] | 0.54 [0.43, 0.65] | 0.68 [0.48, 0.86] | 0.27 | 0.33 | 0.42 |
| fr1_room | 100 | 0.69 [0.51, 0.87] | 0.73 [0.57, 0.87] | 0.91 [0.65, 1.13] | 0.37 | 0.44 | 0.59 |
| fr1_room | 333 | 1.00 [0.81, 1.23] | 1.21 [0.95, 1.42] | 1.30 [0.92, 1.62] | 0.53 | 0.69 | 0.89 |
| fr3 | 33 | 0.31 [0.27, 0.35] | 0.33 [0.29, 0.38] | 0.37 [0.32, 0.48] | 0.20 | 0.26 | 0.30 |
| fr3 | 100 | 0.41 [0.34, 0.44] | 0.45 [0.36, 0.50] | 0.48 [0.40, 0.57] | 0.23 | 0.29 | 0.34 |
| fr3 | 333 | 0.55 [0.44, 0.64] | 0.70 [0.50, 0.90] | 0.71 [0.57, 0.93] | 0.29 | 0.37 | 0.45 |

Median of per-pair M2 / M2 floor within each third; yes / no: CI excludes 1 at all of 8/15/30.

| sequence | gap ms | near | mid | far |
|---|---|---|---|---|
| fr1_360 | 33 | 1.63 [1.34, 1.85] yes | 1.40 [1.22, 1.79] yes | 1.36 [1.23, 1.63] yes |
| fr1_360 | 100 | 1.77 [1.43, 2.21] yes | 1.79 [1.51, 2.32] yes | 1.56 [1.35, 1.85] yes |
| fr1_360 | 333 | 2.22 [1.75, 2.60] yes | 1.94 [1.51, 2.56] yes | 1.68 [1.31, 2.09] yes |
| fr1_desk | 33 | 1.50 [1.35, 1.72] yes | 1.29 [1.14, 1.47] yes | 1.31 [1.18, 1.51] yes |
| fr1_desk | 100 | 1.58 [1.39, 1.81] yes | 1.50 [1.26, 1.76] yes | 1.40 [1.28, 1.62] yes |
| fr1_desk | 333 | 1.83 [1.49, 2.19] yes | 1.91 [1.45, 2.38] yes | 1.68 [1.42, 2.09] yes |
| fr1_room | 33 | 1.82 [1.37, 2.50] yes | 1.52 [1.24, 1.81] yes | 1.57 [1.19, 1.79] yes |
| fr1_room | 100 | 1.83 [1.41, 2.24] yes | 1.47 [1.21, 1.78] yes | 1.47 [1.08, 1.72] no |
| fr1_room | 333 | 1.82 [1.36, 2.29] yes | 1.48 [1.26, 1.96] yes | 1.36 [1.07, 1.82] no |
| fr3 | 33 | 1.46 [1.25, 1.63] yes | 1.23 [1.14, 1.38] yes | 1.25 [1.13, 1.47] yes |
| fr3 | 100 | 1.64 [1.36, 1.80] yes | 1.42 [1.22, 1.59] yes | 1.36 [1.22, 1.55] yes |
| fr3 | 333 | 1.72 [1.50, 2.01] yes | 1.71 [1.43, 2.15] yes | 1.62 [1.33, 1.94] yes |

## 5. Region-level flicker events at 33 ms

Event: a third's per-pair M2 median exceeds the median of the ±5 neighbouring pairs (ordered by time, itself excluded) by more than max(0.5 points, 100%). Floor control: the same rule on the M2 floor of the same third; a model event coincides when a floor event lies on the same pair or within ±1 pair.

| sequence | third | pairs | model events | floor events | model events coinciding | model events not explained by floor |
|---|---|---|---|---|---|---|
| fr1_360 | near | 299 | 34 (11.4%) | 2 | 0 | 34 |
| fr1_360 | mid | 299 | 30 (10.0%) | 0 | 0 | 30 |
| fr1_360 | far | 299 | 35 (11.7%) | 1 | 1 | 34 |
| fr1_desk | near | 299 | 9 (3.0%) | 2 | 0 | 9 |
| fr1_desk | mid | 299 | 14 (4.7%) | 14 | 2 | 12 |
| fr1_desk | far | 299 | 27 (9.0%) | 10 | 3 | 24 |
| fr1_room | near | 299 | 28 (9.4%) | 0 | 0 | 28 |
| fr1_room | mid | 299 | 30 (10.0%) | 3 | 3 | 27 |
| fr1_room | far | 299 | 31 (10.4%) | 0 | 0 | 31 |
| fr3 | near | 294 | 2 (0.7%) | 2 | 0 | 2 |
| fr3 | mid | 294 | 2 (0.7%) | 3 | 1 | 1 |
| fr3 | far | 294 | 21 (7.1%) | 1 | 2 | 19 |

Model event pairs; * coincides with a floor event.

| sequence | third | event pairs (i→j) |
|---|---|---|
| fr1_360 | near | 22→23, 34→35, 66→67, 70→71, 75→76, 81→82, 82→83, 83→84, 91→92, 92→93, 111→112, 112→113, 126→127, 132→133, 133→134, 134→135, 143→144, 150→151, 159→160, 160→161, 171→172, 187→188, 196→197, 198→199, 202→203, 210→211, 237→238, 239→240, 269→270, 270→271, 272→273, 277→278, 284→285, 289→290 |
| fr1_360 | mid | 12→13, 16→17, 22→23, 32→33, 66→67, 82→83, 83→84, 91→92, 92→93, 94→95, 111→112, 112→113, 125→126, 137→138, 139→140, 143→144, 178→179, 187→188, 193→194, 196→197, 198→199, 208→209, 210→211, 218→219, 228→229, 230→231, 233→234, 239→240, 273→274, 274→275 |
| fr1_360 | far | 16→17, 20→21*, 22→23, 61→62, 66→67, 70→71, 81→82, 82→83, 83→84, 91→92, 92→93, 94→95, 111→112, 112→113, 134→135, 135→136, 136→137, 154→155, 160→161, 165→166, 166→167, 178→179, 179→180, 182→183, 196→197, 198→199, 199→200, 206→207, 208→209, 209→210, 210→211, 220→221, 228→229, 260→261, 289→290 |
| fr1_desk | near | 315→316, 318→319, 322→323, 323→324, 363→364, 365→366, 484→485, 485→486, 498→499 |
| fr1_desk | mid | 303→304, 318→319, 322→323, 323→324, 363→364*, 365→366*, 389→390, 395→396, 397→398, 432→433, 482→483, 484→485, 485→486, 498→499 |
| fr1_desk | far | 263→264, 272→273*, 274→275, 281→282, 303→304, 317→318, 322→323, 324→325, 336→337, 348→349, 350→351, 358→359, 365→366, 379→380, 395→396, 397→398, 415→416*, 417→418, 418→419, 431→432, 442→443, 447→448, 468→469, 482→483, 497→498, 498→499, 504→505* |
| fr1_room | near | 373→374, 377→378, 387→388, 395→396, 401→402, 406→407, 408→409, 412→413, 413→414, 420→421, 426→427, 427→428, 438→439, 450→451, 451→452, 457→458, 458→459, 484→485, 542→543, 543→544, 547→548, 550→551, 553→554, 580→581, 587→588, 603→604, 660→661, 663→664 |
| fr1_room | mid | 367→368*, 369→370*, 370→371, 373→374*, 377→378, 399→400, 409→410, 413→414, 420→421, 428→429, 431→432, 432→433, 438→439, 441→442, 450→451, 461→462, 477→478, 534→535, 546→547, 547→548, 550→551, 551→552, 553→554, 557→558, 570→571, 579→580, 589→590, 596→597, 650→651, 663→664 |
| fr1_room | far | 370→371, 378→379, 380→381, 397→398, 408→409, 412→413, 416→417, 420→421, 428→429, 434→435, 435→436, 461→462, 541→542, 542→543, 545→546, 547→548, 548→549, 564→565, 567→568, 571→572, 582→583, 583→584, 584→585, 590→591, 591→592, 593→594, 594→595, 606→607, 623→624, 632→633, 664→665 |
| fr3 | near | 643→644, 690→691 |
| fr3 | mid | 640→641, 768→769* |
| fr3 | far | 650→651, 652→653, 659→660, 715→716, 726→727, 737→738, 740→741, 747→748, 749→750, 758→759, 766→767, 767→768, 768→769, 770→771, 775→776*, 776→777*, 779→780, 800→801, 809→810, 840→841, 853→854 |

fr3 events with frame i or j in 758–776; earlier per-frame far-third spike frames: 758, 761, 767, 768, 770, 776.

| third | event pairs (i→j) | spike frames in an event pair |
|---|---|---|
| near | - | - |
| mid | 768→769* | 768 |
| far | 758→759, 766→767, 767→768, 768→769, 770→771, 775→776*, 776→777* | 758, 767, 768, 770, 776 |
