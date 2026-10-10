# Notes: data, protocol and caveats

Running reference of facts and caveats that affect how results are produced and interpreted. Results themselves are in `report/data_metrics.md`.

## TUM RGB-D data

### fr1 depth lags the RGB image
fr1 depth frames are paired with RGB frames up to ~20 ms apart (mean 7–11 ms); fr3 is essentially synchronised (~0.1 ms). During fast motion, fr1 GT depth is therefore slightly misaligned with the image. This is a GT error, not a model error, and it occurs on exactly the fast frames used to study motion blur, so **blur and GT misalignment are confounded on fr1**.
- Keep `depth_offset_ms` as a per-frame column (done in `per_frame.csv`).
- Report blur results also restricted to frames with a small offset (e.g. < 5 ms).

### Dropped frames ("holes")
Association drops RGB frames without a depth frame within 20 ms, so consecutive kept frames are sometimes 64–100 ms apart instead of ~33 ms.
- Consistency metrics (Week 3) must define frame gaps by **time** (or original `rgb.txt` index), never by position in the kept-frame list.
- Windows were chosen as the 300-frame stretch with the fewest holes. fr1 windows have none; fr3 has no hole-free stretch, and its window has 5 single-frame holes (61–68 ms vs a 32 ms median), which is acceptable.

### Intrinsics confound (fr1 vs fr3)
fr1 uses TUM's approximate ROS-default intrinsics (TUM's recommendation for its pre-registered depth); fr3 uses calibrated intrinsics. fr1 GT point maps are therefore slightly distorted, which inflates Rel_p more than Rel_d (Rel_p − Rel_d ≈ 2.1 on fr1_room vs ≈ 1.0 on fr3).
- **Use Rel_d for fr1 vs fr3 comparisons.**

### Ground-truth properties
- Invalid GT pixels (depth 0) are excluded from all metrics. They reduce coverage but **cannot add error**, so GT holes never explain a high-error frame.
- Kinect artefacts visible in GT point clouds: occlusion shadows (missing depth next to object edges) and depth quantisation that grows with distance (layered far surfaces). Both limit fine-detail comparison at edges and at range.
- fr1_360 trajectory length is 6.64 m vs TUM's published ~5.8 m, most likely ground-truth jitter accumulating on a mostly rotational path. Not investigated further.

## Evaluation protocol

### Upstream (fork `664736a`)
Preprocessing (`dataloader/eval_depth.py`, wired in `main.py:569-596`):
- Shortest side resized to 512 (PIL bilinear), centre crop 512×512, intrinsics adjusted (`:222-272`). GT depth resized with nearest-neighbour.
- GT depth decoded from MoGe's log-encoded PNG (`:34-72`): 0 → NaN, 65535 → inf.
- GT points from `depth_to_pointmap` (`:75-108`): X = (u − cx) z / fx, u = column index, no +0.5 offset.
- GT mask is `isfinite(depth)` only (`:281`); no depth range limits.

Inference (`engine.py`):
- `--eval_num_tokens` defaults to 0 (`main.py:301`), so a 512×512 input is not resized (`engine.py:498-534`).
- Generation under bf16 autocast, then `.float()` (`:687-708`); no resize back.

Masks:
- Sky removal is computed (`engine.py:729-735`) but never applied to metrics; only the GT mask reaches `compute_metrics` (`:775-783`, confirmed by the flag's help text, `main.py:316-323`).
- Images with fewer than 100 valid GT pixels are skipped (`:765-773`).

Alignment (entry point `metrics/compute_metrics.py:86`), fitted on at most 64×64 pixels sampled by `masked_nearest_resize` (`:121`, `masked_resize.py:29`) and applied to the full image:
- **Points:** one scale + 3D shift, `align_points_scale_xyz_shift` (`alignment.py:320-370`, called at `compute_metrics.py:286-288`). Weighted L1 objective Σᵢ wᵢ‖s·pᵢ + t − gᵢ‖₁ with wᵢ = 1/‖gᵢ‖, solved exactly: for each candidate anchor pixel the shift makes that pixel match and a weighted median gives the scale (`align`, `:67-97`); the lowest-loss anchor wins. `max_iters`/`eps` are unused.
- **Depth:** separate scale + shift on predicted z, `align_depth_affine` (`alignment.py:178-229`), weights 1/g, same exact L1 method (`compute_metrics.py:174-176`).

Metrics (`metrics/metrics.py`), averaged over each image's masked pixels:

| Metric | Definition | Lines |
|---|---|---|
| Rel_p | mean(‖p − g‖ / (‖g‖ + 1e-6)) | `:30-34` |
| δ1_p | mean(‖p − g‖ < 0.25 · min(‖g‖, ‖p‖)), a distance threshold, not a 1.25 ratio | `:37-43` |
| Rel_d | mean(\|d − g\| / (g + 1e-6)) | `:20-22` |
| δ1_d | mean(max(g/d, d/g) < 1.25) | `:25-27` |

Dataset score = plain mean of per-image values, ×100 (`engine.py:790-796`, `:1477-1483`).

### This project's protocol (differences from upstream)
- **Full 640×480 TUM frames**, aspect-preserving resize to the default token budget (448×592), prediction resized back to 640×480; no 512 resize-and-crop.
- **Metrics:** upstream's own alignment and metric functions via `evaluate_frame`, GT mask only (`pred_valid=None`), verified identical to upstream on Booster.
- Frames with fewer than 100 valid GT pixels are recorded with `skipped = true` and empty metrics instead of being silently dropped.
- Evaluation runs on CPU; per-frame values can differ from GPU by ~1e-6 (float32 reduction order). Negligible.

**Consequence: TUM numbers are not directly comparable to the paper's tables** (different protocol and data). Never present them as a like-for-like comparison.

## Compute and latency

- Baseline predictions ran as two parallel processes (one per T4). Their latencies (1,674–1,844 ms/frame) are 25–40% above single-process runs (~1,330 ms), likely from CPU contention (4 shared cores; preprocessing and compression partly inside the timed region) and T4 clock throttling under sustained load. Accuracy is unaffected.
- **Never report latency from parallel runs.** Compute-sweep latency must come from a single process on one GPU with nothing else running.
- The T4 has no native bf16 support, while upstream hard-codes bf16 autocast; fp16 may be much faster on T4. To test, checking numerical agreement first.

## Analysis caveats

- One window per sequence, and neighbouring frames are highly correlated, so **300 frames are not 300 independent samples**. Differences between sequences are hypotheses until supported across conditions.
- Preview panels normalise each depth map to its own 2nd–98th percentile, which hides global scale/shift differences and moderate shape errors. **"Layout looks right" does not mean low error.** Use per-pixel error maps for failure analysis.



- Camera speeds use ±20 ms differencing of motion-capture poses, which amplifies jitter: absolute speeds are 30–50% above TUM's published averages. Use them for ranking and correlation, not as absolute values.
- `sharpness_rel` is less reliable near window edges, where the ±0.5 s neighbourhood is one-sided (e.g. fr1_room frame 660: 3.81).
- `expected_misalignment_px` is an order-of-magnitude estimate (it ignores motion direction).
- Implied focal length (`focal_ratio`) is recovered by a least-absolute-deviations fit of X = a·(u−cx)·Z + b·(u−cx) + c (focal = 1/a), which absorbs the unknown shift. A median of (u−cx)·Z/X on aligned points fails (returned 1.01 for a true ratio of 1.2), because the alignment's shift absorbs focal errors. Uses fx only (fr3 has a 0.7% fx/fy difference). Ill-conditioned for frames dominated by one camera-facing plane.

- Statistics (scripts/analyze.py): Spearman with average ranks; partial Spearman via rank residuals after linear regression on the controls' ranks; pooled fr1 results rank within each sequence first. 95% CIs from a moving-block bootstrap within each sequence (block 15, 2000 resamples, fixed seed); resamples always cover the whole sequence and subset filters are applied afterwards. Claims are made only where the CI excludes zero at block lengths 8, 15 and 30.
- The depth_offset_ms < 5 restriction is unusable on fr1_desk (32 frames) and fr1_room (3); fr1 depth is almost always offset. Timing is controlled through expected_misalignment_px in partial correlations instead.
- Spike rule: rel_d above the ±5-frame neighbour median by more than max(1.5 points, 50%). It misses region-level flicker that barely moves overall rel_d (fr3 767–776); the consistency analysis needs a per-region measure. Spike CIs are near-meaningless with 1–2 spikes per sequence.
- edge_density is computed from GT and includes hole boundaries, so it partly reflects GT quality around edges.

- Upstream's exact L1 alignment compares every subsampled pixel with every other: at 4,096 pixels a single point-alignment fit builds 4,096 × 4,096 × 3 arrays (~3.5 GB peak per M1 frame, ~1 GB per M2 fit). This caused two kernel panics (watchdog timeouts from memory thrashing) on the 8 GB development Mac. The code asserts that no fit receives more than 4,096 pixels; full consistency runs happen on Kaggle, and local runs are limited to tests and small samples, single process, 2 threads.

- gt_floor_rgb (GT floor with RGB-timestamp poses) is scored on the same visible pixels as all other metrics (visibility from depth-timestamp poses), so the two GT floors are compared pixel for pixel. It therefore differs slightly from a sweep run entirely with RGB-timestamp poses.