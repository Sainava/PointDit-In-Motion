A. fr1 ground truth lags the image slightly. fr1 depth frames are up to ~20 ms away from their RGB frames. During fast motion, the GT depth is then slightly misaligned with the image. That's a GT error, not a model error, and it happens exactly on the fast frames you'll study for motion blur. So in Week 2, blur and GT misalignment are confounded. The fix then: also report blur results restricted to frames with a small depth-RGB offset (e.g. < 5 ms), and keep the offset as a column in the per-frame CSV.

B. Kept frames have holes. Dropped frames mean consecutive kept frames are sometimes 64–100 ms apart instead of ~33 ms. For Week 3's consistency metric, define frame gaps by time (or by the original rgb.txt index), not by position in the kept list. When choosing ~300-frame windows in subtask 4, prefer stretches with few holes.

The fr1_360 trajectory length (6.64 m against TUM's ~5.8 m) is most likely ground-truth jitter on a mostly rotational path.

Evaluation protocol (upstream, fork 664736a)
(How upstream evaluates (fork at 664736a))

Preprocessing (dataloader/eval_depth.py, wired up in main.py:569-596):
- Image: the shortest side is resized to 512 (PIL bilinear) and the image is centre-cropped to 512×512, with intrinsics adjusted (:222-272). GT depth is resized with nearest-neighbour.
- GT depth is decoded from MoGe's log-encoded PNG (:34-72): 0 becomes NaN, 65535 becomes inf.
- GT points come from depth_to_pointmap (:75-108): X = (u−cx)z/fx, with u the column and no +0.5 offset.
- GT mask is isfinite(depth) only (:281). There are no depth range limits.

Inference (engine.py):
- --eval_num_tokens defaults to 0 (main.py:301), so a 512×512 input is never resized (engine.py:498-534).
- Then generate runs under bf16 autocast, followed by .float() (:687-708). There is no resize back.

Masks:
- Sky removal is computed (engine.py:729-735) but never applied to the metrics. Only the GT mask goes into compute_metrics (:775-783), and the flag's help text confirms it (main.py:316-323).
- Images with fewer than 100 valid GT pixels are skipped (:765-773).

Alignment (metrics/compute_metrics.py:86): this is the entry point. The fit is solved on at most 64×64 pixels, picked by masked_nearest_resize (:121, masked_resize.py:29), and then applied to the full image.
- Points are fitted with one scale plus a 3D shift: align_points_scale_xyz_shift (alignment.py:320-370), called at compute_metrics.py:286-288.
  - Objective: weighted L1, Σᵢ wᵢ‖s·pᵢ + t − gᵢ‖₁ with wᵢ = 1/‖gᵢ‖.
  - It is solved exactly, not iteratively. For each candidate anchor pixel, the shift makes that pixel match and a weighted median gives the scale (align, :67-97); the anchor with the lowest loss wins. The max_iters/eps arguments are never used.
- Depth has its own, separate alignment: scale plus shift on the predicted z, using align_depth_affine (alignment.py:178-229) with weights 1/g_depth and the same exact L1 method (compute_metrics.py:174-176).

Metrics (metrics/metrics.py), each averaged over the masked pixels of one image:

┌────────┬────────────────────────────────────────────────────────────────────────┬────────┐
│ Metric │                               Definition                               │ Lines  │
├────────┼────────────────────────────────────────────────────────────────────────┼────────┤
│ Rel_p  │ mean(‖p−g‖ / (‖g‖+1e-6))                                               │ :30-34 │
├────────┼────────────────────────────────────────────────────────────────────────┼────────┤
│ δ1_p   │ mean(‖p−g‖ < 0.25·min(‖g‖, ‖p‖)), a distance threshold rather than a   │ :37-43 │
│        │ 1.25 ratio                                                             │        │
├────────┼────────────────────────────────────────────────────────────────────────┼────────┤
│ Rel_d  │ mean(|d−g| / (g+1e-6))                                                 │ :20-22 │
├────────┼────────────────────────────────────────────────────────────────────────┼────────┤
│ δ1_d   │ mean(max(g/d, d/g) < 1.25)                                             │ :25-27 │
└────────┴────────────────────────────────────────────────────────────────────────┴────────┘

The dataset score is the plain mean of the per-image values, ×100 (engine.py:790-796, :1477-1483).

***TUM frames are evaluated at full 640×480 rather than the benchmark's resize-and-crop to 512, so your numbers are not directly comparable to the paper's tables.***