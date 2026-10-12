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