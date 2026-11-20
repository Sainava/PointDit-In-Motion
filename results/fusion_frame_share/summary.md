# Fusion frame share

| scope | value |
|---|---|
| quantity | per frame rel_d change in percent of raw, negative means fusion helps |
| poses | oracle poses |
| scale | joint solved scale |
| frames | has_fused true only |
| equal | rel_d_fused within 1e-06 of rel_d_raw |
| bootstrap | moving block over frames in index order, blocks 8, 15, 30, 2000 resamples, seed 0, widest 95 percent interval over blocks |
| worse run | consecutive fused frames in index order |

| sequence | fused frames | lower | equal | higher |
|---|---|---|---|---|
| fr1_360 | 296 | 74.0 % | 0.0 % | 26.0 % |
| fr1_desk | 296 | 59.5 % | 0.0 % | 40.5 % |

| sequence | median change % | mean change % | p10 change % | p90 change % |
|---|---|---|---|---|
| fr1_360 | -5.64 | -4.05 | -21.02 | +11.56 |
| fr1_desk | -2.42 | -2.23 | -17.25 | +12.88 |

| sequence | mean change % [95 % CI] | share lower [95 % CI] |
|---|---|---|
| fr1_360 | -4.05 [-7.57, +0.21] | 74.0 % [64.2 %, 81.4 %] |
| fr1_desk | -2.23 [-4.86, +0.09] | 59.5 % [52.7 %, 68.6 %] |

| sequence | worst indices | worst change % | best indices | best change % |
|---|---|---|---|---|
| fr1_360 | 93 207 95 94 104 | 116.831 80.372 66.871 63.463 62.571 | 114 113 211 115 212 | -53.742 -51.191 -39.616 -38.118 -35.442 |
| fr1_desk | 365 342 333 291 282 | 45.275 44.901 44.200 40.972 38.084 | 327 328 252 293 239 | -50.120 -44.343 -34.841 -30.392 -26.589 |

| sequence | longest worse run | run start index | run end index |
|---|---|---|---|
| fr1_360 | 9 | 275 | 283 |
| fr1_desk | 9 | 337 | 345 |

| sequence | frame | has fused | rank from worst | in worst 5 |
|---|---|---|---|---|
| fr1_360 | 23 | yes | 222 | no |
