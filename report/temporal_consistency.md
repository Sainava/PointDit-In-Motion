# Temporal consistency

## Pipeline validation (development checks, not results)

- Synthetic tests: warping recovers geometry (median error < 0.5%), self-pairs give 0 for GT floor, M1, M2 and M2 floor; occlusion, z-buffer, M2 affine invariance and time-based pairing verified.
- Warp checks: RGB of frame i warped into frame j with GT depth aligns with RGB_j (0–1 px residual on fr3; a few px on some fr1 pairs from motion-capture pose jitter).
- GT-only sweep over the 300-frame windows (depth-timestamp poses), median of per-pair medians ×100: GT floor 0.26–0.41 on fr3 and 0.32–1.34 on fr1, growing with time gap; M2 floor at or below the GT floor, except near-ties on fr1_room at 33/100 ms.
- 20-frame prediction samples (first look, not representative): on fr3, M1 is ~2–3× the GT floor and M2 ~2× the M2 floor at all gaps; on fr1_360 frames 0–19, M2 is slightly above the M2 floor at 33 and 100 ms and below it at 333 ms (only 10 pairs).
- The comparison of GT floors computed with depth- vs RGB-timestamp poses is deferred to the full Kaggle run.