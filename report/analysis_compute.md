# Analysis and compute

## 1. Where the error is: error maps

**Method.** Per-pixel relative depth error |aligned − GT| / GT (fixed 0–0.25 scale) and an edge breakdown: Rel_d = band_px × rel_band + (1 − band_px) × rel_out, where the band is all valid pixels within 5 px of a GT depth discontinuity or hole. Comparing each worst frame with its sequence's median frame shows where the extra error comes from.

| Frame vs median | Extra error on edges | Extra error elsewhere | Excess mainly from |
|---|---|---|---|
| fr1_room 373 vs 637 | +3.6 | +5.1 | regions (59%) |
| fr1_desk 337 vs 426 | +2.2 | +4.1 | regions (66%) |
| fr1_360 242 vs 146 | +4.5 | +2.4 | edges (65%) |
| fr3 648 vs 867 | +1.7 | +0.6 | edges (small) |

**Findings.**
- In median frames, error concentrates on depth edges (ratio band_err / band_px 1.6–2.6). In the worst fr1_room and fr1_desk frames the ratio drops and rel_out roughly triples: failures spread into whole regions.
- Error maps show the failing regions are distinct surfaces placed at the wrong depth relative to the rest of the scene, a non-affine error that one scale + shift cannot fix:
  - fr1_360 242: door-frame edges, plus the far room seen through the doorway (~15–20%).
  - fr1_desk 337: the far background beyond the monitors, plus an error gradient across the desk surface (a tilted plane).
  - fr1_room 373: mainly a near, heavily blurred object cut off at the bottom image border (≥25% over its whole area), plus the serrated edge of a wooden rack.
- fr1_room 373 has a narrow GT depth range (0.62–2.91 m), so the earlier hypothesis that fr1_room is harder because of a wide depth range is **not supported** by its worst frame.
- fr1_room 370–374: depth–RGB offset does not track error within the cluster (e.g. frame 379: 17.8 ms, Rel_d 10.2; frame 372: 13.3 ms, Rel_d 14.1), and the dominant error is a whole region, which timing misalignment cannot produce. **Interpretation: model error.** The edge component is to be confirmed with the expected-misalignment covariate.
- fr1_desk 362–363 (visibly blurred) are region-dominated too, so blur should not be assumed to cause only edge errors.

**Candidate failure modes for further analysis:** near objects cut off by the image border; far regions seen through openings; tilted large planes (possibly a wrong implied field of view).


## 2. Per-frame covariates

Computed for all 1,200 window frames (`results/covariates/covariates.csv`): camera angular and linear speed (from ground truth, ±20 ms around each RGB timestamp), image-motion estimate, expected depth–RGB misalignment in pixels, sharpness (Laplacian variance) and sharpness relative to ±0.5 s neighbours, GT depth percentiles, depth-edge density and a near-border-object heuristic.

| Sequence | Angular speed mean / max (°/s) | Linear mean (m/s) | Median sharpness | Mean expected misalignment (px) |
|---|---|---|---|---|
| fr1_desk | 31.9 / 96.6 | 0.54 | 0.00234 | 5.58 |
| fr1_room | 36.8 / 87.4 | 0.33 | 0.00208 | 5.27 |
| fr1_360 | 53.5 / 122.3 | 0.28 | 0.00087 | 1.40 |
| fr3_long_office | 15.2 / 36.8 | 0.20 | 0.00386 | 0.02 |

**Findings.**
- Speed ordering matches TUM's published averages (fr1_360 fastest, fr3 slowest); absolute values are 30–50% higher, consistent with motion-capture jitter amplified by the short differencing window. Speeds are used as relative covariates only.
- fr1_room 366–380: expected misalignment is 4–9 px, comparable to the 5 px edge band, so GT timing can explain part of the edge error, and some may fall just outside the band. It cannot produce region-wide errors (e.g. the foreground object in frame 373) and tracks Rel_d only loosely (frame 372: Rel_d 14.1 at 5.7 px). **Revised interpretation: region errors are model errors; edge errors are partly timing.**
- fr1_360: mean expected misalignment is 1.4 px, so its edge-dominated errors (e.g. frames 233–243) are not timing; blur or scene content remain.

## 3. Error by distance and implied focal length

**Error by depth tertile** (rel_d over the near / mid / far third of each frame's GT depth) shows that failing regions differ by sequence:
- fr1_room 366–420: the near third is consistently worst (370–374: near 20.7–24.8 vs mid 8.4–9.8, far 7.3–10.5), quantitatively confirming the near-foreground failure seen in the error map of frame 373.
- fr1_desk 337: far third worst (13.4 vs near 7.6), matching the far-background error.
- fr1_360 241–242: near and mid elevated together.
- fr3: isolated single-frame spikes (758, 761, 767, 768, 770, 776) where the far third jumps to ~11–12 while neighbours stay at ~3–4 and δ1 drops to ~89. With slow motion, synchronized depth and deterministic inference, this is frame-to-frame instability in far geometry (**temporal flicker**), relevant to the consistency analysis.

**Implied focal length** (`focal_ratio`) is plausible and smooth on fr3 (0.78–1.29) but unreliable on parts of fr1 (up to 4.7 on fr1_room; ~0.7 on fr1_360 frames with Rel_d < 2, where a 30% FOV error is not credible). Likely ill-conditioned when one camera-facing plane dominates the view. Not used until filtered by a reliability check (split-half agreement).

## 4. What explains the error: per-frame analysis

**Method.** All 1,200 frames, per sequence and pooled over fr1 (ranked within sequence). Spearman and partial Spearman correlations; 95% CIs from a moving-block bootstrap (block 15, 2,000 resamples); claims only where the CI excludes zero at block lengths 8, 15 and 30. Inputs and full tables: `results/analysis/`; forest plot: `report/figures/analysis/correlations_forest.png`.

**Image sharpness (key result).** Lower relative sharpness predicts higher depth error independently of GT timing and scene content: partial Spearman of rel_d vs sharpness_rel, controlling for expected misalignment and edge density, is −0.23 to −0.29 in all four sequences (all CIs exclude zero) and −0.26 [−0.36, −0.18] pooled over fr1, robust to block length. It also holds on fr3, where expected misalignment is ~0.02 px, so it is not a GT-timing artifact. It is described as an image-sharpness effect, not specifically motion blur: fr3 moves slowly yet shows the same effect.

**Camera speed** is a weak, inconsistent predictor (clear only on fr1_360, ρ 0.34); fast-vs-slow differences in median rel_d all have CIs spanning zero. Sharpness, the property of the image the model sees, matters more than speed.

**GT timing** could not be tested by restriction: only 32 fr1_desk and 3 fr1_room frames have depth offset < 5 ms. It was controlled through expected misalignment instead (above).

**Scene content** is the strongest factor: rel_d rises with depth-edge density (pooled ρ 0.62; fr1_360 0.86) and depth range (pooled 0.54). Within-sequence content varies slowly, so several per-sequence intervals lose significance at block 30; only the pooled results and fr1_360 are robust. Edge density is computed from GT and includes hole boundaries, so part of this effect may reflect GT quality around edges.

**Near-border objects** are at most a weak factor (pooled ρ 0.26, borderline); the fr1_room 373 case is not a general failure mode.

**Temporal spikes** are rare (≤ 0.7% of frames) and of two kinds: fr1_desk 337/338 occur at fast, blurred moments (75 °/s vs 28 typical, sharpness_rel 0.69), while fr3 758/761 occur at slower-than-typical motion with normal sharpness, i.e. not motion-driven: near-identical inputs produce different far geometry. The spike rule misses region-level flicker that barely moves overall rel_d (fr3 767–776).

**Implied focal length: inconclusive.** A split-half check (fitting left and right image halves separately) passes only 121 of 1,200 frames. The halves disagree systematically in opposite directions even on fr3, suggesting the fit lacks a term for horizontal offset (principal point or small yaw) rather than random noise. The only testable subset (fr1_360, n = 51, ρ −0.91) consists mostly of plane-dominated frames where both halves are equally ill-conditioned, so the result is an artifact. Not used in conclusions.

**Summary.** PointDiT's per-frame error on TUM is driven mainly by scene content (depth edges, depth range) and secondarily by image sharpness, which has an effect independent of GT timing. Camera speed matters only through sharpness. A small number of frames show non-motion-driven instability in far geometry, which motivates the temporal-consistency analysis.

**Limitations.** One 300-frame window per sequence; strongly autocorrelated frames; correlational analysis only; Kinect GT has noise and holes at edges; fr1 intrinsics are approximate.


## Compute sweep (single T4, one process, 200 frames from fr1_360 and fr3)

Latency is model.generate only. The GPU clock fell from 1590 MHz to 585 MHz during the first config and stayed there, so latencies are relative to each other under a sustained throttled clock. Accuracy is the median per frame Rel_d difference against the bf16 baseline, with a block bootstrap interval (block 15).

| Config | Latency ms | Rel_d diff [95% CI] |
|---|---|---|
| baseline bf16, 3 steps | 1593 | 0 |
| fp16 | 341 | +0.007 [0.004, 0.015] |
| steps 1 | 797 | +0.107 [0.081, 0.123] |
| steps 2 | 1201 | +0.020 [0.003, 0.036] |
| steps 4 | 2008 | +0.004 [-0.006, 0.016] |
| tokens 512 | 744 | +0.254 [0.144, 0.437] |
| tokens 2048 | 3715 | +0.216 [0.092, 0.286] |
| model B | 495 | +0.796 [0.520, 1.007] |
| input downscale 2 | 1601 | -0.017 [-0.056, 0.014] |

Findings. fp16 is 4.7 times faster than bf16 on the T4 at a negligible accuracy cost. Extra steps and tokens do not help. Model B is clearly less accurate. Limits are two sequences, autocorrelated frames and a single block length.