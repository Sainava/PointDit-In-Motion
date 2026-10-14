# Temporal consistency

## Pipeline validation (development checks, not results)

- Synthetic tests: warping recovers geometry (median error < 0.5%), self-pairs give 0 for GT floor, M1, M2 and M2 floor; occlusion, z-buffer, M2 affine invariance and time-based pairing verified.
- Warp checks: RGB of frame i warped into frame j with GT depth aligns with RGB_j (0–1 px residual on fr3; a few px on some fr1 pairs from motion-capture pose jitter).
- GT-only sweep over the 300-frame windows (depth-timestamp poses), median of per-pair medians ×100: GT floor 0.26–0.41 on fr3 and 0.32–1.34 on fr1, growing with time gap; M2 floor at or below the GT floor, except near-ties on fr1_room at 33/100 ms.
- 20-frame prediction samples (first look, not representative): on fr3, M1 is ~2–3× the GT floor and M2 ~2× the M2 floor at all gaps; on fr1_360 frames 0–19, M2 is slightly above the M2 floor at 33 and 100 ms and below it at 333 ms (only 10 pairs).
- The comparison of GT floors computed with depth- vs RGB-timestamp poses is deferred to the full Kaggle run.

## Full evaluation (all 1,200 frames; 33 / 100 / 333 ms gaps)

Results: `results/consistency/baseline/` (summary.csv, pairs_*.csv, consistency.json). Kaggle CPU, ~2.4 h, peak ~2.4 GB per worker. Regression check against the development values: 114/114 within 1e-3, pair counts exact.

Median over pairs of each pair's median relative depth error ×100, visible pixels:

| Sequence | Gap (ms) | GT floor (depth poses) | GT floor (RGB poses) | M1 | M2 floor | M2 | M2 / M2 floor |
|---|---|---|---|---|---|---|---|
| fr1_desk | 33 / 100 / 333 | 0.39 / 0.51 / 0.96 | 0.46 / 0.62 / 1.25 | 0.79 / 1.05 / 2.05 | 0.34 / 0.45 / 0.62 | 0.48 / 0.72 / 1.24 | 1.42 / 1.51 / 1.80 |
| fr1_room | 33 / 100 / 333 | 0.32 / 0.46 / 0.78 | 0.35 / 0.53 / 0.99 | 0.82 / 1.07 / 2.02 | 0.33 / 0.46 / 0.69 | 0.61 / 0.76 / 1.20 | 1.64 / 1.59 / 1.57 |
| fr1_360 | 33 / 100 / 333 | 0.40 / 0.63 / 1.34 | 0.39 / 0.64 / 1.36 | 0.78 / 1.24 / 2.58 | 0.37 / 0.49 / 0.71 | 0.56 / 0.91 / 1.64 | 1.50 / 1.77 / 1.92 |
| fr3_long_office | 33 / 100 / 333 | 0.26 / 0.30 / 0.41 | 0.26 / 0.30 / 0.41 | 0.44 / 0.57 / 1.05 | 0.25 / 0.28 / 0.36 | 0.34 / 0.46 / 0.65 | 1.34 / 1.50 / 1.70 |

**Findings.**
- PointDiT's frame-to-frame inconsistency (M2) exceeds its floor in every sequence at every gap (1.3–1.9×); M1 exceeds the GT floor by 1.6–2.8×.
- The excess grows with time gap in three of four sequences (largest on fr1_360, 1.92× at 333 ms): predicted geometry is partly view-dependent.
- Frame-to-frame disagreement (~0.3–0.6% at 33 ms) is far smaller than per-frame error (~4–6% Rel_d), so most error is persistent across nearby viewpoints rather than flicker. To be confirmed with like-for-like statistics (per-frame median vs per-pair median).
- GT floor timing: RGB-timestamp poses raise the floor on fr1_desk (+19% / +30% at 33 / 333 ms) and fr1_room (+7% / +27%), but not on fr1_360 or fr3, which have small depth offsets. Depth-timestamp poses are used for the GT floor.
- Interquartile ranges are of the same size as the medians: a minority of pairs are much less consistent than typical. Drivers to be analysed.

## Sampling noise vs zero initialisation

Subsets: fr1_360 frames 200–299, fr3_long_office_household 740–839. Runs: zero init (default) and noise init with three seeds (per-frame seeds, reproducible). Results: `results/sampling/`.

**Validation.** The zero-init rerun reproduced the cached baseline exactly (200/200 frames, max difference 0); zero-init consistency on the subsets matched the full baseline run exactly (0 failures).

**Findings.**
- Noise init increases frame-to-frame inconsistency (M2) at 33 ms by 10–22% on fr1_360 and 14–18% on fr3 (most CIs exclude no change); by ~4–9% at 100 ms (significant on fr3 only); not at 333 ms (ratios 0.98–1.02). Sampling noise adds short-term flicker; at larger gaps, view-dependent inconsistency dominates.
- Same-frame disagreement between noise seeds is ~0.27–0.33% (median per pixel), 40–70% of the zero-init frame-to-frame inconsistency at 33 ms (0.66% fr1_360, 0.42% fr3 on these subsets).
- Accuracy is essentially unchanged: noise minus zeros in per-frame Rel_d is +0.01 to +0.04 points (baseline 4–7%); on fr3 noise is very slightly worse (CIs exclude zero).
- Zero init vs a noise seed (~0.21–0.26) disagrees less than seed vs seed (~0.27–0.33), by about √2, consistent with the zero-init prediction lying near the centre of the noisy predictions.

**Conclusion.** For moving-camera use, deterministic zero initialisation reduces short-term flicker at no accuracy cost.


## Analysis of inconsistency

Results: `results/consistency_analysis/`, `results/persistence/`; figures in `report/figures/consistency/`. Statistics as in the per-frame analysis (block-bootstrap CIs; claims only where the CI excludes the reference value at block lengths 8, 15 and 30).

**Headline.** M2 / M2 floor is 1.34–1.92 in every sequence and gap, CI excluding 1 at all block lengths. It grows with gap on three of four sequences (fr1_360: 1.50 → 1.92; fr3: 1.34 → 1.70; fr1_desk: 1.42 → 1.80; fr1_room is flat, 1.64 → 1.57).

**What drives excess inconsistency (33 ms).** Scene content and the frame's own accuracy, not blur: edge density (ρ 0.25–0.53; pooled fr1 0.32), depth range (0.28–0.40; pooled 0.32) and the frame's own Rel_d (pooled fr1 0.39) have robust positive correlations. Image sharpness has no robust effect on inconsistency (pooled partial ρ −0.08, not robust to block length), in contrast to its effect on per-frame accuracy. Camera speed matters only on fr3 (ρ 0.30) at 33 ms and fr1_360 (0.27) at 100 ms. Effects are weaker at 100 ms. Frames that are wrong tend to be the ones that are unstable.

**Which region.** Relative to its floor, the near third is least consistent at 33 ms in every sequence (ratio near / mid / far: fr1_360 1.63 / 1.40 / 1.36; fr1_desk 1.50 / 1.29 / 1.31; fr1_room 1.82 / 1.52 / 1.57; fr3 1.46 / 1.23 / 1.25). In absolute terms the far third flickers most, but its floor is also highest (Kinect noise grows with distance). The differences between thirds narrow at 333 ms.

**Flicker events (33 ms, per depth third).** 98 of 99 (fr1_360), 86 of 89 (fr1_room), 45 of 50 (fr1_desk) and 22 of 25 (fr3) model events have no matching event in the GT floor, so they are model behaviour, not data artifacts. On fr3, flicker is concentrated in the far third (21 events, 7.1% of pairs, against 0.7% in near and mid): near-identical frames from a slow, synchronized camera receive different far geometry, even with deterministic inference. The event list covers earlier spike frames 758, 767, 768, 770; frame 776 coincides with a floor event and is likely partly a ground-truth artifact. Several fr1_360 event pairs occur in all three thirds at once (e.g. 66→67, 82→83, 91→92, 111→112): whole-frame jumps, not region-specific.

**Persistence.** On the same pixels, with GT depth as common reference, the median per-frame error is 2.0–2.5%, while frame-to-frame disagreement at 33 ms is 0.44–0.82% (2.6–5.2× smaller). The persistent fraction P is 0.90–0.94 at 33 ms, 0.79–0.90 at 100 ms and 0.47–0.75 at 333 ms (lowest on fr1_360, highest on fr3). Most error is therefore shared between neighbouring frames, and the view-dependent part grows with viewpoint change. Caveats: shared GT noise in the reference slightly inflates P; pose jitter lowers P at 333 ms on fr1, so those values are lower bounds.

**Limitations.** One window per sequence; strongly autocorrelated pairs; correlational analysis; Kinect GT noise and fr1 pose jitter limit the floors; the flicker-event rule is a threshold choice.