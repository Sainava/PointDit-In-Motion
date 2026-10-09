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