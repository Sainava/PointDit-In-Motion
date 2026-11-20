# PointDiT in Motion — project guide

A map of everything built so far. It covers how data flows, how each number is computed, why each step exists, and what comes next. Facts come from the code unless a sentence is marked **(context)**, which means it is background I was given and cannot be checked in the repo. Paths are relative to the repo root.

---

## 1. One-page overview

**The question (context).** How reliable is PointDiT for a camera moving through a room? PointDiT is a pixel-space diffusion transformer that predicts a dense 3D point map from a single image, defined only up to a per-frame affine transform. It is evaluated zero-shot on four TUM RGB-D sequences (fr1_desk, fr1_room, fr1_360, fr3_long_office_household), using one 300-frame window from each. The work is an applied study meant for LinkedIn, for the PointDiT authors and for master's applications.

**Pipeline.**

```
 TUM folder (rgb.txt, depth.txt, groundtruth.txt, PNGs)
        |
        |  src/pim/data/tum.py        associate RGB<->depth, slerp poses, decode depth
        |  configs/sequences.yaml     intrinsics + 300-frame window
        v
 Frame list (RGB path, depth path, depth offset, pose at RGB time)
        |
        +--> scripts/predict.py  (Kaggle T4)  ---> predictions/<run>/<seq>/<idx>.npz  (float16 points)
        |                                                   |
        +--> scripts/compute_covariates.py ---> results/covariates/covariates.csv
        |                                                   |
        |    scripts/evaluate.py (Kaggle CPU) <-------------+
        |        GT backprojection + upstream alignment + metrics
        |        ---> results/eval/baseline/per_frame.csv  (+ focal_split.csv)
        |                                                   |
        |    scripts/consistency_eval.py (Kaggle CPU) <-----+   warp, visibility, GT floor, M1, M2, M2 floor
        |        ---> results/consistency/baseline/pairs_<seq>.csv, summary.csv
        |    scripts/persistence_eval.py (Kaggle CPU) <-----+   E_i, E_j, persistent fraction P
        |        ---> results/persistence/
        |    sampling runs (GPU) + scripts/sampling_compare.py (CPU)
        |        ---> results/sampling/
        v
 scripts/analyze.py              (laptop)  per-frame statistics  ---> results/analysis/, report/figures/analysis/
 scripts/analyze_consistency.py  (laptop)  pair statistics       ---> results/consistency_analysis/, report/figures/consistency/
```

**Phases (context for branch and tag names, code for contents).**

| Phase | Branch / tag | What it produced |
|---|---|---|
| 1 Pipeline and baseline | `feat/week1-data-metrics`, tag `week1` | TUM loader, window selection, cached inference, metric wrapper verified against upstream, `results/eval/baseline/per_frame.csv`, `report/data_metrics.md` |
| 2 What drives per-frame error | `feat/analysis-compute`, tag `analysis` | Error maps and edge band, depth tertiles, implied focal length, covariates, rank statistics with block bootstrap, `report/analysis_compute.md` |
| 3 Temporal consistency | `feat/temporal-consistency`, tag `consistency-analysis` (equal to the merge into `main`, commit `a8f6652`) | Warping, visibility, GT floor, M1, M2, M2 floor, persistence, sampling-noise experiment, flicker events, `report/temporal_consistency.md` |
| Still to do (context) | — | Temporal fusion fix, visual deliverables, compute sweep, final report and outreach |

`feat/temporal-consistency` is one commit ahead of `main`. That commit (`1ab329a`) only points the notebooks back at the feature branch.

---

## 2. Where code and data live

### Directory map

| Path | What it holds |
|---|---|
| `src/pim/data/tum.py` | Sequence loading, association, pose interpolation, depth decoding, windows |
| `src/pim/data/windows.py` | Hole counting and fewest-holes window selection |
| `src/pim/model/runner.py` | `PointDiTRunner`, which wraps the fork's model for single-image inference |
| `src/pim/metrics/geometry.py` | Back-projection, projection, `evaluate_frame` (upstream metrics), `implied_focal` |
| `src/pim/metrics/error_maps.py` | Per-pixel relative error, depth-edge band, edge breakdown, depth tertiles |
| `src/pim/evaluation.py` | Per-frame evaluation tasks and the `per_frame.csv` row format |
| `src/pim/analysis/covariates.py` | Motion, sharpness and scene-geometry covariates |
| `src/pim/analysis/stats.py` | Ranks, (partial) Spearman, moving-block bootstrap, spike detection |
| `src/pim/consistency/warp.py` | Relative pose, depth-timestamp pose, forward warp with z-buffer, pairing by time gap |
| `src/pim/consistency/metrics.py` | Visibility, GT floor, M1, M2, M2 floor, per-pair summaries |
| `src/pim/consistency/evaluation.py` | Per-sequence pair loop, frame cache, pair CSV |
| `src/pim/consistency/persistence.py` | E_i, E_j, persistent fraction, M1 regression check |
| `src/pim/sampling.py` | Seed disagreement, determinism check, bootstrap of medians and ratios |
| `src/pim/subsets.py`, `src/pim/runs.py`, `src/pim/paths.py` | Frame subsets, run records and provenance, repo paths and fork setup |
| `configs/` | `sequences.yaml` (intrinsics, windows), run configs (`baseline`, `zeros_rerun`, `noise_seed0-2`), `sampling_subsets.yaml`, `consistency_expected.json` (regression values) |
| `scripts/` | Command-line entry points (table below) |
| `notebooks/` | `kaggle_runner.ipynb` (GPU), `kaggle_evaluate.ipynb` (CPU) |
| `tests/` | 16 test files, about 190 tests, plus the smoke fixture `tests/fixtures/smoke_fr1360_664736a/` |
| `third_party/pointdit/` | Fork of PointDiT, submodule pinned at `664736a`. Read only |

### Scripts, where they run, what they write

| Script | Runs on | Reads | Writes |
|---|---|---|---|
| `setup_weights.py`, `smoke_test.py`, `check_sampler_init.py`, `check_booster_metrics.py` | Kaggle GPU (runner notebook) | Weights, one frame, Booster scenes | Smoke output and printed checks |
| `predict.py` | Kaggle GPU | TUM, run config | `/kaggle/working/predictions/<run>/<seq>/<idx>.npz`, `run.json` |
| `check_determinism.py` | Kaggle (runner and evaluate notebooks) | Two prediction runs | `results/sampling/determinism.csv` |
| `evaluate.py` | Kaggle CPU | Predictions, TUM | `results/eval/<run>/per_frame.csv`, `eval.json` |
| `compare_eval_csv.py`, `check_frame.py` | Kaggle CPU | Old and new CSV, one frame | Printed regression checks |
| `focal_reliability.py` | Kaggle CPU | Predictions, `per_frame.csv` | `results/eval/baseline/focal_split.csv` |
| `preview_predictions.py` | Kaggle (both notebooks) | Predictions, TUM | Preview PNGs and `error_breakdown.csv` (Kaggle output, not in `results/`) |
| `compute_covariates.py` | Not recorded (needs TUM images only, no predictions) | TUM | `results/covariates/covariates.csv` |
| `inspect_covariates.py`, `inspect_tum.py`, `plot_metrics.py`, `export_gt_ply.py`, `select_windows.py`, `check_reprojection.py` | Development checks | Various | `results/covariates/inspect/`, `report/figures/data_metrics/`, PLY files, `configs/sequences.yaml` windows |
| `consistency_eval.py` | Kaggle CPU | Predictions, TUM, `configs/consistency_expected.json` | `results/consistency/<run>/` |
| `persistence_eval.py` | Kaggle CPU | Predictions, TUM, `results/consistency/baseline/` | `results/persistence/` |
| `sampling_compare.py` | Kaggle CPU | Sampling runs, their eval and consistency outputs | `results/sampling/` |
| `analyze.py` | Laptop | `per_frame.csv`, `covariates.csv`, `focal_split.csv` | `results/analysis/`, `report/figures/analysis/` |
| `analyze_consistency.py` | Laptop | Pairs, covariates, `per_frame.csv`, persistence | `results/consistency_analysis/`, `report/figures/consistency/` |
| `scripts/figures.py` | — | — | Empty file |

**Machines (context).** GPU prediction runs on Kaggle T4 notebooks and CPU-heavy evaluation runs on a separate Kaggle CPU notebook. The laptop (8 GB) only runs tests and small CSV analyses. Notebooks clone this repo with the fork submodule at `664736a`. DINOv3 code is pinned separately (`6876159`), and DINOv3 weights are downloaded each session with an HF token and never committed.

### Notebook flags

| Notebook | Flag | Stage it runs |
|---|---|---|
| `kaggle_runner.ipynb` | `RUN_SAMPLING = False` (default) | Weights, smoke test, sampler and Booster checks, `predict.py` with `configs/baseline.yaml` split over two T4s, previews |
| `kaggle_runner.ipynb` | `RUN_SAMPLING = True` | `predict.py --subsets` for `noise_seed0-2` and `zeros_rerun` (two configs per T4), then `check_determinism.py` against the attached baseline |
| `kaggle_evaluate.ipynb` | `RUN_EVALUATION` | `evaluate.py`, then `compare_eval_csv.py` against an attached older CSV (tolerance 1e-6). If False, it reuses the attached `per_frame.csv` |
| `kaggle_evaluate.ipynb` | `RUN_FOCAL_SPLIT` | `focal_reliability.py` |
| `kaggle_evaluate.ipynb` | `RUN_CONSISTENCY` | `consistency_eval.py` with 4 workers, M2 maps for fr1_360 and fr1_desk at 33 ms, regression check |
| `kaggle_evaluate.ipynb` | `RUN_PERSISTENCE` | `persistence_eval.py` with M1 regression check |
| `kaggle_evaluate.ipynb` | `RUN_SAMPLING_EVAL` (must run alone) | `check_determinism.py`, `evaluate.py` and `consistency_eval.py` per sampling run, then `sampling_compare.py` |
| both | `PROJECT_BRANCH` | Branch to clone. Currently `feat/temporal-consistency` |

### Result files

| Path | Content |
|---|---|
| `results/eval/baseline/per_frame.csv` | 1,200 rows, one per frame (columns in §3.4) |
| `results/eval/baseline/eval.json` | Protocol text, commits, timings |
| `results/eval/baseline/focal_split.csv` | Implied focal length fitted on full, left and right halves |
| `results/covariates/covariates.csv` | 1,200 rows of covariates |
| `results/analysis/` | `correlations.csv`, `tertile_comparison.csv`, `spikes.csv`, `spike_comparison.csv`, `summary.md` |
| `results/consistency/baseline/` | `pairs_<seq>.csv` (one row per pair), `summary.csv`, `consistency.json` |
| `results/persistence/` | `persistence_<seq>.csv`, `persistence_summary.csv`, `persistence.json` |
| `results/sampling/` | `determinism.csv`, `eval/<run>/`, `consistency/<run>/`, `seed_disagreement*.csv`, `accuracy_*.csv`, `consistency_summary.csv`, `consistency_regression.csv`, `summary.md` |
| `results/consistency_analysis/` | `headline.csv`, `persistence.csv`, `drivers.csv`, `pair_drivers.csv`, `thirds.csv`, `flicker_events.csv`, `flicker_counts.csv`, `summary.md` |
| `report/*.md` | Phase write-ups (`data_metrics.md`, `analysis_compute.md`, `temporal_consistency.md`) and `notes.md` (protocol and caveats) |
| `report/figures/` | `data_metrics/`, `analysis/`, `consistency/` PNGs |

Predictions, M2 maps and previews live only in Kaggle outputs and are not committed.

---

## 3. How data travels, step by step

Each step lists input, output, code, and why it exists.

### 3.1 TUM loading and association

- **Input.** `rgb.txt`, `depth.txt`, `groundtruth.txt` and PNGs of one sequence, plus `configs/sequences.yaml`.
- **Output.** `TUMSequence` with `K`, `window` and a list of `Frame` objects (index, RGB timestamp, RGB path, depth path, `depth_offset` = t_depth − t_rgb, `T_world_cam` at the RGB timestamp).
- **Code.** `src/pim/data/tum.py`. Entry point is `load_sequence`, which calls `associate`, `read_groundtruth`, `build_frames` and `interpolate_pose`.
- **Why.** TUM streams RGB and depth separately. A frame needs an RGB image, a depth image close in time, and a GT pose.

Steps inside `load_sequence`.
1. `associate` pairs RGB and depth by timestamp using TUM's greedy rule (§4.1).
2. `build_frames` interpolates the GT pose at each RGB timestamp and drops frames with no pose (outside the GT range or across a GT gap over 50 ms).
3. The frame `index` is the position in this kept list. All later code calls this the kept-frame index.

**Depth.** `load_depth` decodes 16-bit PNG depth as metres (raw / 5000), with 0 meaning missing. **Intrinsics.** `load_camera_config` reads `fx, fy, cx, cy` by camera name. fr1 uses ROS defaults (525, 525, 319.5, 239.5) and fr3 uses calibrated values (535.4, 539.2, 320.1, 247.6). No undistortion is applied anywhere (comment in `configs/sequences.yaml`).

### 3.2 Windows and holes

- **Input.** Kept-frame timestamps of a whole sequence.
- **Output.** `{start, count}` written into `configs/sequences.yaml`.
- **Code.** `src/pim/data/windows.py` (`hole_gaps`, `select_window`), `scripts/select_windows.py`. `apply_window` in `tum.py` slices the frame list.
- **Why.** Association drops RGB frames without a depth match, which leaves holes in time. A window with few holes keeps frame spacing regular.

A hole is a gap larger than 1.5× the median gap. The chosen window has the fewest holes, with ties going to the earliest start. The resulting windows are fr1_desk 209–508, fr1_room 366–665, fr1_360 0–299 (no holes) and fr3 637–936 (5 single-frame holes).

### 3.3 Inference and cached predictions

- **Input.** RGB uint8 640×480, run config (`init`, `seed`, `seed_mode`).
- **Output.** `predictions/<run>/<sequence>/<index:05d>.npz` holding `points` (H×W×3 float16), `valid` (sky mask), `rgb_timestamp`, `rgb_file`, `latency_ms`, `peak_mem_gb`, and optionally `seed`. A `run.json` records config, windows, commits, GPU and invocations.
- **Code.** `src/pim/model/runner.py` (`PointDiTRunner.predict`), `scripts/predict.py` (`predict_sequence`, `save_prediction`, `RunConfig.frame_seed`).
- **Why.** Inference needs a GPU and evaluation does not, so predictions are cached once and evaluated many times on CPU.

What `predict` does.
1. It scales pixels to [−1, 1].
2. It resizes bilinearly to the token-budget size (`_inference_size`, a copy of upstream's wild-image branch). For 640×480 this gives 448×592 (§4.2).
3. It runs `model.generate` under bf16 autocast. The sampler is 3 Euler steps, starting from `generate_noise_scale · N(0, I)` (`third_party/pointdit/denoiser.py`, `Denoiser.generate`). Init "zeros" sets the scale to 0 and "noise" sets it to the training `noise_scale` (1.0) (`_noise_scale`).
4. It resizes the output back to 640×480 with **nearest** interpolation.
5. It computes the sky mask (point norm > 2.9, upstream default), saves it, and never uses it in the per-frame metrics.

`load_runner` warms up twice. Existing `.npz` files are skipped, so runs can resume. Seeds are `None` for zeros. For `per_frame` mode, frame i uses seed × 1,000,000 + i.

Upstream already starts from zeros, so default inference is deterministic **(context, confirmed by the comment in `Denoiser.generate`)**.

### 3.4 GT point map, alignment, per-frame metrics

- **Input.** One cached prediction and the frame's GT depth and `K`.
- **Output.** One row of `per_frame.csv`.
- **Code.** `src/pim/evaluation.py` (`evaluate_task`, `extra_metrics`, `frame_row`), `src/pim/metrics/geometry.py` (`backproject`, `evaluate_frame`), upstream `third_party/pointdit/metrics/compute_metrics.py` (`compute_metrics`), `metrics/alignment.py`, `metrics/metrics.py`.
- **Why.** PointDiT's output is affine-invariant, so it must be aligned to GT before any error is measured. Reusing upstream's exact functions keeps the metric identical to the paper.

Flow.
1. `backproject` turns GT depth into a GT point map (§4.1).
2. `evaluate_frame` uses the GT mask only (`pred_valid=None`) and skips frames with fewer than 100 valid pixels.
3. It calls upstream `compute_metrics` with `points_affine_invariant = pred` and `depth_affine_invariant = pred[..., 2]` (the predicted z).
4. Upstream fits two separate alignments on a 64×64 masked-nearest subsample and applies them to the full frame. One is scale plus 3D shift for points and the other is scale plus shift for depth (§4.3).
5. Rel and δ1 are computed over all GT-valid pixels and multiplied by 100.
6. `evaluate_frame` also returns `aligned_depth`. This GT-aligned depth is the input to every later step (error maps, tertiles, M1, M2, persistence).

`per_frame.csv` columns are `sequence, frame_index, rgb_timestamp, rgb_file, gap_ms, depth_offset_ms, gt_valid_fraction, num_valid, skipped, rel_p, delta1_p, rel_d, delta1_d, latency_ms, peak_mem_gb, rel_d_near, rel_d_mid, rel_d_far, focal_ratio`. `gap_ms` is the time since the previous window frame. `depth_offset_ms` is |t_depth − t_rgb|.

**Memory guard.** `_upstream_compute_metrics` wraps every upstream aligner with `_size_checked`, which raises if a fit receives more than 4,096 pixels (`MAX_ALIGNMENT_PIXELS`).

### 3.5 Covariates

- **Input.** Window frames (RGB, GT depth, GT trajectory, `fx`).
- **Output.** `results/covariates/covariates.csv`.
- **Code.** `src/pim/analysis/covariates.py` (`frame_covariates`, `add_relative_sharpness`), `scripts/compute_covariates.py`.
- **Why.** To test what explains per-frame error. The candidates are motion, blur, GT timing and scene content. Formulas are in §4.4.

Columns are `angular_speed_dps, linear_speed_mps, image_motion_px_per_s, expected_misalignment_px, sharpness, sharpness_rel, depth_p5, depth_p50, depth_p95, depth_range_ratio, edge_density, near_border_fraction`. `sharpness_rel` is computed over the window frames only, so it is one-sided near window edges.

### 3.6 Error maps, edge band, tertiles, implied focal length

- **Code.** `src/pim/metrics/error_maps.py` (`relative_depth_error`, `depth_edge_band`, `edge_breakdown`, `rel_d_by_depth_tertile`), `scripts/preview_predictions.py` (`error_analysis`), `src/pim/metrics/geometry.py` (`implied_focal`), `scripts/focal_reliability.py` (`focal_split`).
- **Why.** Rel_d is one number per frame. Error maps show *where* the error is. The edge breakdown separates boundary error (which GT timing or Kinect edges could cause) from region error (model). Tertiles separate near and far. Focal length tests a field-of-view hypothesis.

Outputs. `rel_d_near/mid/far` and `focal_ratio` go into `per_frame.csv`. The edge breakdown goes into the preview's `error_breakdown.csv` (Kaggle only), and the left and right focal fits go into `focal_split.csv`.

### 3.7 Warping, visibility, pairing

- **Input.** Two frames i and j of the same window, with GT depth, GT-aligned prediction and poses.
- **Output.** Masks `visible` and `interior` in frame j, and GT and predictions of i warped into j.
- **Code.** `src/pim/consistency/warp.py` (`relative_pose`, `depth_pose`, `warp_depth`, `_forward_warp`, `pairs_by_gap`), `src/pim/consistency/metrics.py` (`visibility`).
- **Why.** To compare two predictions, the same 3D surface has to be found in both images. Only pixels seen in both frames (not occluded, not out of view) can be compared.

Pairing (`pairs_by_gap`). For each frame i and each gap g in {33, 100, 333} ms, j is the frame whose timestamp is closest to t_i + g. The pair is kept only if that timestamp is within 10 ms. Pairing is by time, never by list position, so pairs across holes are dropped. That is why fr3 has 294 pairs at 33 ms and not 299.

Two sets of poses are used.
- **RGB-timestamp poses** (`Frame.T_world_cam`) drive M1, M2, the M2 floor and `gt_floor_rgb`. They match what the model saw.
- **Depth-timestamp poses** (`depth_pose`, interpolated at t_rgb + depth_offset) drive visibility and the GT floor. They match when the GT depth was captured.

### 3.8 Consistency metrics and floors

- **Input.** Pairs from §3.7, plus `FrameInputs` per frame (`frame_inputs` runs `evaluate_frame` once per frame).
- **Output.** `pairs_<seq>.csv`, with one row per pair. For each of `gt_floor_depth, gt_floor_rgb, m1, m2_floor, m2`, and for pixel sets `visible` and `interior`, it stores the mean, median, p90 and near/mid/far medians (×100). It also stores `frac_visible`, `frac_interior`, `too_few_pixels` and `actual_gap_ms`. `summary.csv` holds the median and IQR over pairs, plus ratio columns.
- **Code.** `src/pim/consistency/metrics.py` (`pair_consistency`, `_m2`, `align_depth_to_anchor`, `_pixel_set`), `src/pim/consistency/evaluation.py` (`evaluate_sequence`, `FrameStore`, `pair_row`), `scripts/consistency_eval.py` (`summarize`, `regression_checks`).
- **Why.** M1 and M2 measure how much predictions of the same surface disagree across frames. The floors measure how much *GT itself* disagrees under the same procedure (Kinect noise, pose jitter, warping), so the excess over the floor is attributable to the model.

`FrameStore` loads each frame once and frees it after its last pair, and pairs run in order of frame i. That keeps memory to a band of frames (about 2.4 GB per worker on Kaggle, per `report/temporal_consistency.md`). The script ends with a regression check against `configs/consistency_expected.json` (tolerance 1e-3).

### 3.9 Persistence

- **Input.** The same pairs, pixels and poses as M1.
- **Output.** `persistence_<seq>.csv` with, per region (`visible`, `interior`, `visible_near/mid/far`), `n`, the median of |E_i|, |E_j| and |E_i − E_j|, Pearson, Spearman and P.
- **Code.** `src/pim/consistency/persistence.py` (`pair_errors`, `persistence_stats`, `m1_regression`), `scripts/persistence_eval.py`.
- **Why.** M2 is small compared with Rel_d. Persistence asks how much of each frame's error is *shared* with its neighbour (persistent) and how much is *view-dependent*. Fusion can only fix the second part.

Regression check. Per pair, median |E_i − E_j| must equal the stored M1 median within 1e-6, because the two are the same quantity by construction (§4.6).

### 3.10 Sampling experiment

- **Input.** Four runs on the subsets fr1_360 200–299 and fr3 740–839 (`configs/sampling_subsets.yaml`). The runs are `zeros_rerun` and `noise_seed0/1/2` with per-frame seeds.
- **Output.** `results/sampling/`.
- **Code.** `scripts/check_determinism.py`, `scripts/sampling_compare.py` (`frame_disagreements`, `accuracy_tables`, `consistency_summary`, `regression`), `src/pim/sampling.py` (`disagreement`, `median_estimate`, `median_ratio_estimate`, `consistency_regression`).
- **Why.** To find out whether the deterministic zero start helps or hurts, and how large pure sampling noise is compared with frame-to-frame inconsistency.

Checks.
1. `zeros_rerun` must match the cached baseline (max |Δpoints| ≤ 2e-3, the float16 storage tolerance) on every frame.
2. Zero-init consistency on the subsets must match the full baseline pairs within 1e-3.
3. Pairs are planned over the whole window and then filtered to the subset (`plan_pairs` with `frame_range`), so subset rows equal the full-run rows exactly.

### 3.11 Statistics layer

- **Code.** `src/pim/analysis/stats.py`, `scripts/analyze.py` (per frame), `scripts/analyze_consistency.py` (per pair), `src/pim/sampling.py` (sampling).
- **Why.** Frames are strongly autocorrelated, so ordinary CIs would be far too narrow. Rank statistics resist outliers, and partial correlations separate confounded covariates (blur versus GT timing).

`analyze.py` joins `per_frame.csv`, `covariates.csv` and `focal_split.csv` on (sequence, frame_index) and checks that keys and timestamps agree. It then computes correlations (Q1 motion, Q2 blur versus timing, Q3 scene content, Q4 focal), fast-versus-slow medians and spikes. `analyze_consistency.py` builds one series per (sequence, gap) ordered by frame i. To each pair it attaches the mean of frames i and j for each covariate and Rel_d of frame j. It joins persistence, checks its medians against `summary.csv` (to 1e-9), and computes headline estimates, drivers, thirds and flicker events.

### 3.12 How the pieces connect

```
configs/sequences.yaml ─┬─> predict.py ──> predictions ─┬─> evaluate.py ──> per_frame.csv ─┬─> analyze.py
                        │                               │        └─> focal_reliability.py ─┘
                        ├─> compute_covariates.py ──────┼──────────> covariates.csv ───────┼─> analyze.py
                        │                               │                                  └─> analyze_consistency.py
                        │                               ├─> consistency_eval.py ─> pairs_*.csv ─┬─> analyze_consistency.py
                        │                               │                                       ├─> persistence_eval.py (regression reference)
                        │                               │                                       └─> sampling_compare.py (regression reference)
                        │                               └─> persistence_eval.py ─> persistence_*.csv ─> analyze_consistency.py
configs/noise_seed*.yaml, zeros_rerun.yaml + sampling_subsets.yaml
        └─> predict.py --subsets ─> check_determinism.py, evaluate.py, consistency_eval.py ─> sampling_compare.py
```

---

## 4. Every formula, in one place

Notation. Pixel (u, v) with u the column and v the row, both integers with no +0.5 offset. `K` holds fx, fy, cx, cy. z means depth. A hat means prediction. g means GT. "×100" means values are reported in percent.

### 4.1 Data and geometry

| Quantity | Formula | Code | Why |
|---|---|---|---|
| Depth decoding | z = raw / 5000 m if raw > 0, else invalid | `tum.py` `load_depth` | TUM PNG convention |
| Back-projection | X = (u − cx)·z / fx, Y = (v − cy)·z / fy, Z = z | `geometry.py` `backproject` | GT point map in camera coordinates, the same convention as upstream `depth_to_pointmap` |
| Projection | u = fx·X/Z + cx, v = fy·Y/Z + cy | `geometry.py` `project` | Inverse of the above, used for warping |
| Association | Candidates have \|t_rgb − t_depth\| ≤ 20 ms (+1e-6 float slack). Accept greedily from the smallest \|Δt\|, each image used at most once | `tum.py` `associate` | TUM's own `associate.py` rule |
| Pose interpolation | Bracketing GT samples k, k+1 with gap ≤ 50 ms, α = (t − t_k)/(t_{k+1} − t_k). Translation = (1 − α)·t_k + α·t_{k+1}. Rotation = slerp(q_k, q_{k+1}, α), sign-flipped to the short arc, with normalised lerp when q_k·q_{k+1} > 0.9995 | `tum.py` `interpolate_pose`, `slerp` | Pose at the exact RGB (or depth) timestamp |
| Hole | gap_k > 1.5 × median(all gaps) | `windows.py` `hole_gaps` | Detects dropped frames |

### 4.2 Inference size

target tokens T = (512 / 16)² = 1024 (`eval_wild_target_tokens` and `eval_num_tokens` are both 0). Resize factor r = sqrt(T·16² / (H·W)), new side = round(side·r / 16)·16. For 640×480, r = 0.924, which gives 448×592. Code is `runner.py` `_inference_size`. This keeps the token count near the 512² training budget while preserving aspect ratio.

### 4.3 Upstream alignment and metrics

**Subsample.** `masked_nearest_resize(mask, size=(64, 64))` picks, for each of the 64×64 target cells, the nearest GT-valid pixel in that cell's receptive field. That gives at most 4,096 pixels. The fit uses only these pixels, and the result is applied to the whole frame (`compute_metrics.py` `compute_metrics`).

| Alignment | Objective (solved on the subsample) | Solver | Code |
|---|---|---|---|
| Depth (scale s, shift t) | min Σ_i w_i·\|s·ẑ_i + t − g_i\|, w_i = 1/g_i | For each anchor pixel k, t = g_k − s·ẑ_k, and s is the weighted median of (g_i − g_k)/(ẑ_i − ẑ_k). The lowest-loss anchor wins. This is exact for this two-parameter L1 problem | `alignment.py` `align_depth_affine`, `align` |
| Points (scale s, shift t ∈ R³) | min Σ_i w_i·Σ_c \|s·p̂_ic + t_c − g_ic\|, w_i = 1/‖g_i‖ | For each anchor pixel k, t = g_k − s·p̂_k (all three coordinates matched), and s is the weighted median over the 3n coordinate ratios. The lowest-loss anchor wins | `alignment.py` `align_points_scale_xyz_shift` |

The anchor search compares every subsampled pixel with every other. At 4,096 pixels that builds 4,096 × 4,096 × 3 arrays, which is the source of the memory peaks (§6).

| Metric | Definition (mean over GT-valid pixels of one frame, then ×100) | Code |
|---|---|---|
| Rel_p | mean ‖p − g‖ / (‖g‖ + 1e-6), p = s·p̂ + t | `metrics.py` `rel_point` |
| δ1_p | fraction with ‖p − g‖ < 0.25·min(‖g‖, ‖p‖) (a distance threshold, not a 1.25 ratio) | `metrics.py` `delta1_point` |
| Rel_d | mean \|d − g\| / (g + 1e-6), d = s_d·ẑ + t_d | `metrics.py` `rel_depth` |
| δ1_d | fraction with max(g/d, d/g) < 1.25 | `metrics.py` `delta1_depth` |

Sequence score is the plain mean (or median) of per-frame values. The ×100 is applied in `evaluate_frame`. Rel_d uses the depth alignment, not the z channel of the point alignment.

### 4.4 Covariates and per-frame extras

| Quantity | Formula | Code | Why |
|---|---|---|---|
| Angular speed | θ = arccos((tr R_rel − 1)/2), with T_rel = T(t − h)⁻¹·T(t + h) and h = 20 ms. ω = θ / 2h in °/s | `covariates.py` `camera_motion`, `rotation_angle` | Motion as a blur cause |
| Linear speed | v = ‖translation of T_rel‖ / 2h in m/s | same | Same |
| Image motion | m = ω[rad/s]·fx + v·fx / depth_p50 (px/s) | `image_motion_px_per_s` | Rough pixel speed. Ignores direction (order of magnitude only) |
| Expected misalignment | e = m × \|t_depth − t_rgb\| (px) | `frame_covariates` | How far GT depth edges shift relative to RGB on fr1 |
| Sharpness | Var of the 4-neighbour Laplacian over interior pixels of gray = (0.299R + 0.587G + 0.114B)/255 | `sharpness` | Standard blur proxy |
| Relative sharpness | sharpness / median(sharpness of window frames within ±0.5 s, itself included) | `relative_sharpness` | Removes slow scene-texture changes so only sudden blur remains |
| Depth percentiles and range | p5, p50, p95 of valid GT depth. range ratio = p95 / p5 | `depth_geometry` | Scene depth extent |
| Edge band | Seed pixels where neighbouring valid depths differ by more than 4% of the smaller one, or where valid meets invalid. Dilate by 5 px (Chebyshev, i.e. a square) and keep valid pixels | `error_maps.py` `depth_edge_band` | Boundary region where timing or Kinect edges add error |
| Edge density | \|band\| / \|valid\| | `depth_geometry` | Amount of depth discontinuity in the view. Includes hole boundaries |
| Near-border fraction | Fraction of valid pixels in the outer 10% strip with depth < 0.6 × median frame depth | `near_border_fraction` | Heuristic for near objects cut off by the border |
| Relative error map | e(u, v) = \|d − g\| / g on valid pixels (unscaled) | `relative_depth_error` | Per-pixel Rel_d |
| Edge breakdown | Rel_d = f_band·rel_band + (1 − f_band)·rel_out, where f_band is the band pixel fraction. `band_error_fraction` = Σ_band e / Σ e | `edge_breakdown` | Boundary error versus region error |
| Depth tertiles | Per frame, cut at the 33.3 and 66.7 percentiles of valid GT depth. near < low, mid in [low, high), far ≥ high. Mean e ×100 per third | `rel_d_by_depth_tertile` | Which distance fails |
| Implied focal | L1 fit (IRLS, 30 iterations) of X̂ = a·(u − cx)·Ẑ + b·(u − cx) + c on the **raw** prediction over GT-valid pixels with \|u − cx\| > 50. f = 1/a and `focal_ratio` = f / fx. Scale cancels and b, c absorb the shift | `geometry.py` `implied_focal` | Tests whether the model's field of view is wrong |
| Focal split | Same fit on pixels left of cx − 50 and right of cx + 50. `focal_split_logdiff` = \|ln(f_left / f_right)\| | `focal_reliability.py` `focal_split` | Reliability check. "Reliable" means logdiff < ln 1.1 |

### 4.5 Warping and visibility

| Quantity | Formula | Code | Why |
|---|---|---|---|
| Relative pose | T_{j←i} = T_world_cam_j⁻¹ · T_world_cam_i | `warp.py` `relative_pose` | Maps points from camera i into camera j |
| Forward warp | Back-project valid depth of i, transform by T_{j←i}, keep Z > 0, project, round to the nearest pixel, keep pixels inside the image. Z-buffer keeps the smallest Z per target pixel. Unhit pixels get 0 | `warp.py` `_forward_warp` | Expresses frame i's surface in frame j's pixels |
| Visible | hit ∧ valid_j ∧ \|z_gt_{i→j} − z_gt_j\| ≤ 0.15·z_gt_j, with the GT warp using depth-timestamp poses | `metrics.py` `visibility` | Same surface in both views. A larger disagreement means occlusion or disocclusion |
| Interior | visible ∧ not edge band(GT_j) | same | Removes boundary pixels where small misregistration dominates |
| Too few | \|visible\| < 1000 makes the whole pair NaN. A pixel subset with < 1000 scored pixels is NaN | `_pixel_set` | Avoids noisy medians |

The predicted depth of frame i is warped with **GT** intrinsics K and only from GT-valid pixels of i (`warp_depth(aligned_depth_i, gt_valid_i, ...)`).

### 4.6 Consistency metrics

All errors are per-pixel relative errors. Each pair stores the median over the scored pixels ×100, which is the headline per-pair value. The sequence value is the median over pairs.

| Metric | Per-pixel error | Poses | Scored pixels | Code |
|---|---|---|---|---|
| GT floor | \|z_gt_{i→j} − z_gt_j\| / z_gt_j | depth-timestamp | visible | `pair_consistency` |
| GT floor (RGB poses) | same, but GT_i warped with RGB-timestamp poses | RGB | visible ∧ hit | same |
| M1 | \|ẑ_{i→j} − ẑ_j\| / z_gt_j, where ẑ is each frame's GT-aligned depth | RGB | visible ∧ anchor hit | same |
| M2 | \|(a·ẑ_j^raw + b) − ẑ_{i→j}\| / ẑ_{i→j}, where (a, b) fits frame j's **raw** z to frame i's GT-aligned depth warped into j (`align_depth_affine`, weights 1/ẑ_{i→j}, 64×64 subsample of scored pixels) | RGB | visible ∧ anchor hit | `_m2`, `align_depth_to_anchor` |
| M2 floor | M2 with GT in place of predictions, i.e. \|(a·z_gt_j + b) − z_gt_{i→j}\| / z_gt_{i→j} | RGB (for the warp), depth (for visibility) | visible ∧ hit | `_m2` |
| M2 / M2 floor | median over pairs of the per-pair ratio M2_k / M2floor_k | — | — | `consistency_eval.py` `ratio_median`, `analyze_consistency.py` `pair_series` |
| Excess | M2_k − M2floor_k (per pair) | — | — | `pair_series` |

Why each exists.
- **GT floor with depth-timestamp poses.** GT depth was captured at t_depth, up to about 20 ms after the RGB on fr1. Warping it with RGB-time poses adds the camera motion in between. On fr1_desk that raised the floor by 19–30%.
- **M1.** Each frame is aligned to its own GT, so M1 mixes model inconsistency with per-frame alignment differences and GT noise.
- **M2 aligned to the warped anchor, not to GT.** GT fixes only frame i's scale and shift. Frame j is fitted to frame i's prediction, so M2 measures the prediction's self-consistency up to one affine transform. Per-frame GT alignment can neither hide nor create disagreement.
- **M2 floor.** It is the M2 a perfect predictor would get if its output were the Kinect GT. It includes Kinect noise, pose jitter, warping and the affine fit. A ratio of 1 means "as consistent as GT".

### 4.7 Persistence

| Quantity | Formula | Code |
|---|---|---|
| E_i | (ẑ_{i→j} − z_gt_j) / z_gt_j (signed) on M1 pixels | `persistence.py` `pair_errors` |
| E_j | (ẑ_j − z_gt_j) / z_gt_j on the same pixels | same |
| Identity | \|E_i − E_j\| = \|ẑ_{i→j} − ẑ_j\| / z_gt_j = M1 per pixel | basis of `m1_regression` |
| Persistent fraction | P = 1 − Var(E_i − E_j) / (Var(E_i) + Var(E_j)) = 2·Cov(E_i, E_j) / (Var E_i + Var E_j) | `persistence_stats` |

P = 1 when the two frames' errors are identical (fully persistent), 0 when they are uncorrelated, and negative when they are anti-correlated. It is computed per pair and the median is taken over pairs. Shared GT noise in z_gt_j raises P slightly, and pose jitter lowers it.

### 4.8 Sampling comparison

| Quantity | Formula | Code |
|---|---|---|
| Seed disagreement | For one frame, the reference config's GT-aligned depth is d_ref. The other config's raw z is fitted to d_ref with `align_depth_to_anchor`. The value is the median over pixels of \|aligned − d_ref\| / d_ref ×100. Mask = GT-valid ∧ both sky masks ∧ d_ref > 0 | `sampling.py` `disagreement`, `sampling_compare.py` `frame_disagreements` |
| Pairs compared | Every noise-seed pair, plus zeros against each seed | `comparisons` |
| Accuracy change | Median over frames of (Rel_d of noise − Rel_d of zeros) per frame | `accuracy_tables` |
| M2 ratio to zeros | median_pairs(M2 noise) / median_pairs(M2 zeros), with pairs resampled jointly | `median_ratio_estimate` |

If zero init sat exactly at the centre of independent noisy predictions, the seed-versus-seed distance would be √2 times the zeros-versus-seed distance.

### 4.9 Statistics

| Quantity | Definition | Code |
|---|---|---|
| Spearman ρ | Pearson correlation of average ranks (ties share the mean rank) | `stats.py` `average_ranks`, `spearman` |
| Partial Spearman | r_y = rank(y)/(n+1) minus its least-squares fit on [1, ranks of controls], with r_x the same. ρ = Pearson(r_x, r_y) | `_rank_residuals`, `partial_spearman` |
| Pooled (pooled_fr1) | Ranks and residuals computed **within** each fr1 sequence, then concatenated before Pearson. Sequence-level differences cannot drive ρ | `pooled_partial_spearman` |
| Moving-block bootstrap | ⌈n/L⌉ blocks of L consecutive positions with starts uniform in [0, n − L], concatenated and truncated to n. 2,000 resamples, run separately within each sequence (and each gap for pairs). Subset filters apply after resampling. 95% percentile CI | `moving_block_indices`, `bootstrap`, `percentile_ci` |
| Robustness rule | A claim is made only if the CI is on the same side of the reference value (0 for correlations, 1 for ratios) at L = 8, 15 **and** 30. Used in `analyze.py` and `analyze_consistency.py`. `sampling_compare.py` uses L = 15 only | `analyze.py` `robustness`, `analyze_consistency.py` `Bootstrap` |
| Fast versus slow | median Rel_d of the fastest third (by angular speed) − median of the slowest third. Pooled value = mean of per-sequence differences | `fast_slow_median_difference` |
| Spike (per frame) | Rel_d_k − m_k > max(1.5, 0.5·m_k), where m_k is the median of Rel_d at positions k−5..k+5 excluding k | `detect_spikes`, `analyze.py` `find_spikes` |
| Flicker event (per pair, 33 ms) | The same rule on each third's per-pair M2 median, with threshold max(0.5, 1.0·m_k) | `analyze_consistency.py` `flicker_events` |
| Floor control | A model event is "explained" if a floor event (the same rule on the same third's M2 floor) lies within ±1 pair | `near_any`, `event_tables` |

Neighbours are counted in positions, not seconds. On fr3, ±5 positions can span slightly more time because of holes.

---

## 5. Results so far, and what each means

### Phase 1 — baseline accuracy (`report/data_metrics.md`, `results/eval/baseline/per_frame.csv`)

| Sequence | Rel_p mean | δ1_p mean | Rel_d mean / median | δ1_d mean |
|---|---|---|---|---|
| fr1_desk | 5.19 | 97.65 | 4.24 / 3.91 | 97.63 |
| fr1_room | 8.10 | 94.39 | 5.99 / 6.06 | 94.80 |
| fr1_360 | 6.27 | 96.26 | 5.09 / 5.02 | 96.59 |
| fr3_long_office | 4.65 | 98.18 | 3.66 / 3.79 | 98.22 |

Validation (context and `report/data_metrics.md`). The metric wrapper matches upstream on 10 Booster scenes (max |diff| 0, aggregates Rel_d 1.995, δ1_d 99.937). An independent numpy recomputation matches the CSV to about 2.4e-6. The zero-init rerun is bitwise stable.

**Meaning.** PointDiT transfers zero-shot to handheld Kinect footage with about 4–6% depth error. fr3 is easiest. fr1 numbers are inflated in Rel_p by approximate intrinsics, so Rel_d is the metric for comparing fr1 with fr3. Errors come in clustered events rather than isolated outliers.

### Phase 2 — what drives per-frame error (`report/analysis_compute.md`, `results/analysis/summary.md`)

| Finding | Key number | Status |
|---|---|---|
| Scene content predicts error | Rel_d vs edge density, pooled fr1 ρ 0.62 (fr1_360 0.86). Rel_d vs depth range, pooled 0.54 | Solid when pooled and on fr1_360. Several per-sequence CIs lose significance at L = 30 |
| Lower relative sharpness, higher error, independent of GT timing | Partial ρ (controls are misalignment and edge density) −0.23 to −0.29 in all four sequences, pooled −0.26 [−0.36, −0.18] | Solid. It holds on fr3, where misalignment ≈ 0.02 px |
| Camera speed | Clear only on fr1_360 (ρ 0.34). Fast-vs-slow CIs span 0 | Weak. Speed acts through sharpness |
| Near-border objects | Pooled ρ 0.26, borderline | Weak, not a general failure mode |
| Worst frames fail in regions, not edges | fr1_room 373 and fr1_desk 337 excess is 59–66% outside the edge band. fr1_360 242 is edge-dominated | Solid for those frames (case studies) |
| Region error is model error, edge error is partly timing | fr1_room expected misalignment 4–9 px during 366–380 | Hypothesis supported by covariates |
| Far-geometry instability on fr3 | Single-frame far-third jumps to about 11–12 at 758–776 | Led to Phase 3 |
| Implied focal length | Only 121 of 1,200 frames pass the split-half check, halves disagree systematically, fr1_360 ρ −0.91 judged an artifact | **Inconclusive, dropped** |
| Wide depth range explains fr1_room > fr1_360 | fr1_room 373 has a narrow range (0.62–2.91 m) | **Not supported** |

**Meaning.** Per-frame error is driven mostly by what is in view (depth discontinuities, depth extent) and secondarily by image sharpness. Speed matters only through blur. GT timing on fr1 explains part of the edge error but none of the region failures.

### Phase 3 — temporal consistency (`report/temporal_consistency.md`, `results/consistency_analysis/summary.md`)

Median over pairs (×100, visible pixels). Gaps are 33, 100 and 333 ms.

| Sequence | GT floor | M1 | M2 floor | M2 | M2 / M2 floor | P (persistent fraction) |
|---|---|---|---|---|---|---|
| fr1_desk | 0.39 / 0.51 / 0.96 | 0.79 / 1.05 / 2.05 | 0.34 / 0.45 / 0.62 | 0.48 / 0.72 / 1.24 | 1.42 / 1.51 / 1.80 | 0.90 / 0.79 / 0.50 |
| fr1_room | 0.32 / 0.46 / 0.78 | 0.82 / 1.07 / 2.02 | 0.33 / 0.46 / 0.69 | 0.61 / 0.76 / 1.20 | 1.64 / 1.59 / 1.57 | 0.94 / 0.88 / 0.66 |
| fr1_360 | 0.40 / 0.63 / 1.34 | 0.78 / 1.24 / 2.58 | 0.37 / 0.49 / 0.71 | 0.56 / 0.91 / 1.64 | 1.50 / 1.77 / 1.92 | 0.92 / 0.84 / 0.47 |
| fr3 | 0.26 / 0.30 / 0.41 | 0.44 / 0.57 / 1.05 | 0.25 / 0.28 / 0.36 | 0.34 / 0.46 / 0.65 | 1.34 / 1.50 / 1.70 | 0.93 / 0.90 / 0.75 |

Validation. The full run matched the development values (114/114 within 1e-3, exact pair counts). The persistence M1 regression passed at 1e-6. The zero-init subset rerun matched baseline pairs with max diff 0.

| Finding | Key number | Status |
|---|---|---|
| PointDiT is less consistent than its GT floor | M2 / M2 floor 1.34–1.92, CI excludes 1 at L = 8/15/30 in all 12 cells | Solid |
| Excess grows with gap (view dependence) | fr1_360 1.50 → 1.92, fr3 1.34 → 1.70, fr1_desk 1.42 → 1.80. fr1_room flat | Solid for 3 of 4 sequences |
| Most error persists across neighbouring frames | P 0.90–0.94 at 33 ms, 0.79–0.90 at 100 ms, 0.47–0.75 at 333 ms. median \|E_j\| 2.0–2.5 vs median \|E_i − E_j\| 0.44–0.82 at 33 ms | Solid. The 333 ms values on fr1 are lower bounds (pose jitter) |
| Drivers of excess at 33 ms | Pooled fr1 robust ρ for edge density 0.32, depth range 0.32, Rel_d of frame j 0.39. Sharpness has no robust effect (pooled partial −0.08) | Solid pooled. Per sequence, fr1_desk edge density (0.17) is not robust |
| Near third least consistent relative to floor | 33 ms near / mid / far ratios, e.g. fr1_360 1.63 / 1.40 / 1.36, fr3 1.46 / 1.23 / 1.25 | Solid at 33 ms, narrows at 333 ms |
| Flicker events are model behaviour | 98/99, 86/89, 45/50, 22/25 events without a matching floor event (fr1_360, fr1_room, fr1_desk, fr3). fr3 events concentrate in the far third (21) | Solid, but the threshold is a choice |
| GT floor needs depth-timestamp poses | RGB poses raise the fr1_desk floor by 19% / 30% (33 / 333 ms) | Solid |
| Noise init adds short-term flicker | M2 ratio noise/zeros at 33 ms 1.10–1.22 (fr1_360) and 1.14–1.18 (fr3). About 1.04–1.10 at 100 ms. About 1 at 333 ms | Solid at 33 ms (L = 15 only) |
| Noise init leaves accuracy unchanged | ΔRel_d +0.01 to +0.04 points. On fr3 slightly worse, CIs exclude 0 | Solid, negligible size |
| Seed-to-seed disagreement | 0.27–0.33 (×100), about 47–68% of zero-init M2 at 33 ms | Solid |

**Meaning.** PointDiT is temporally stable in absolute terms (about 0.3–0.6% frame-to-frame at 33 ms against about 4–6% per-frame error), but measurably less stable than its own GT under the same procedure. Most error is a persistent, view-dependent bias, and its non-shared part grows as the viewpoint changes. Frames that are wrong are also the unstable ones. Deterministic zero init is the right default for video.

### Main limitations

- One 300-frame window per sequence, and strongly autocorrelated frames and pairs. All results are correlational.
- fr1 has approximate intrinsics, a depth lag of up to about 20 ms, and motion-capture jitter that inflates speeds by 30–50% and lowers P at 333 ms.
- Kinect GT has holes and edge shadows, and noise grows with distance. Edge density includes hole boundaries.
- Full-frame 640×480 protocol, not comparable to the paper's 512 crop.
- Event and spike rules are threshold choices. The bootstrap CIs for spikes are close to meaningless with 1–2 spikes.
- Latencies in `per_frame.csv` come from two parallel processes and must not be reported (`report/notes.md`).

---

## 6. Decisions and pitfalls

### Non-obvious decisions

| Decision | Reason |
|---|---|
| Use upstream's `compute_metrics` unchanged, GT mask only | It is the paper's exact metric, verified on Booster. The sky mask is computed but ignored, as upstream does |
| Full frames at the default token budget, resized back with nearest | It keeps the whole TUM view. The cost is that the paper's numbers are not comparable |
| Store predictions as float16 | Disk and transfer size. The determinism tolerance is 2e-3 because of this |
| Keep the 64×64 alignment subsample, also for M2 | It matches upstream. Exact L1 alignment is quadratic in pixels, and above 4,096 pixels it exhausts memory (asserted by `check_alignment_size`) |
| Pair frames by time with a 10 ms tolerance | Holes make list positions unequal in time, and pairs across holes are dropped |
| Depth-timestamp poses for visibility and the GT floor, RGB-timestamp poses for predictions | Each warp uses the pose at which its data was captured |
| M2 aligned to frame i's warped prediction | It measures self-consistency rather than per-frame fit to GT |
| Report medians and ratios of medians | Per-pixel and per-pair distributions are heavy-tailed (IQR about equal to the median) |
| Per-frame seeds for noise runs (seed × 1e6 + i) | `per_call` would give every frame the same starting noise and correlate them artificially |
| Rel_d, not Rel_p, for fr1 vs fr3 | Approximate fr1 intrinsics distort GT point maps, which inflates Rel_p (Rel_p − Rel_d ≈ 2.1 on fr1_room vs 1.0 on fr3) |
| Moving-block bootstrap with the L = 8/15/30 rule | Autocorrelation. The claim must not depend on the block length |
| Partial correlations with expected misalignment as control | The depth-offset < 5 ms subset is too small (32 and 3 frames on fr1_desk and fr1_room) |
| Covariates averaged over i and j for pair drivers | A pair's inconsistency depends on both frames |

### What went wrong and how it was fixed

| Problem | Fix |
|---|---|
| Smoke fixture recorded on the wrong sequence (context) | Commit `fa7b2a5` re-recorded it on an fr1_360 frame and made sequence order deterministic (`find_sequences` sorts). The fixture directory is `tests/fixtures/smoke_fr1360_664736a/` |
| Stale Kaggle clones running old code (context) | `kaggle_evaluate.ipynb` now fetches, checks out and fast-forwards an existing clone. Both notebooks clone only when `.git` is missing and print the project and fork commits |
| Kernel panics on the 8 GB Mac from memory thrashing (context and `report/notes.md`) | Diagnosed as 4,096 × 4,096 × 3 alignment arrays (about 3.5 GB peak per M1 frame). Added the 4,096-pixel assertion. Full runs moved to Kaggle. Local runs are tests and small samples only, one process, 2 threads |
| Early misreadings of failure clusters | The fr1_room 370–374 cluster was revised from GT timing to model error after error maps. The wide-depth-range hypothesis was rejected. Implied focal length was dropped after the split-half check |
| Parallel-run latency inflated 25–40% | Recorded as not reportable. The sweep must use a single process |
| fr1_360 trajectory 6.64 m vs TUM's ~5.8 m | Attributed to GT jitter, not investigated further |

---

## 7. What is next

### Before starting (housekeeping found in the repo)

1. **Branch names.** `main` equals the `consistency-analysis` merge, and `feat/temporal-consistency` is one commit ahead (notebooks point at it). If fusion gets its own branch, set `PROJECT_BRANCH` in both notebooks to it, and back to `main` on merge.
2. **Stale provenance.** `report/data_metrics.md` lists evaluation commit `584cd16` and 4.47 s/frame. The committed `results/eval/baseline/eval.json` is from `5be07dd` at 4.23 s/frame (the Phase 2 re-evaluation with tertiles and focal). Notebook versions are still `<n>` placeholders.
3. **Stray files.** `peak_memory/metrics/{consistency,geometry,motion}.py` and `scripts/figures.py` are empty. The fixture file is named `" meta.json"` with a leading space.
4. **Runner notebook clone.** `kaggle_runner.ipynb` does not pull an existing clone (only the evaluate notebook does). This is low risk in fresh sessions.
5. **Untested pieces.** `runner.py` (GPU only), `preview_predictions.py`, `compute_covariates.py` as a script, plotting functions, `check_booster_metrics.py` and `smoke_test.py` have no unit tests.
6. **Not in `results/`.** M2 maps (fr1_360 and fr1_desk, 33 ms), previews and `error_breakdown.csv` exist only in Kaggle outputs. The fusion and video steps will need them attached.
7. **Report wording to tighten.** See the list at the end of this section.

### Remaining steps in order (context for the list, repo for inputs)

| Step | Needs | Produces | Validate by |
|---|---|---|---|
| 1 Temporal fusion fix | Cached baseline predictions, poses, the warp and M2 alignment code (`warp_depth`, `align_depth_to_anchor`) | A fused depth per frame and a new run or eval directory | Synthetic tests (identical inputs give the input back, and noise is reduced), a window-size-zero run reproducing baseline exactly, then the same per-frame metrics, M2 / M2 floor and P. Success means M2 / M2 floor moves toward 1 without Rel_d getting worse |
| 2 Visual deliverables | Predictions, M2 maps, GT poses, `export_gt_ply.py` and the fork's PLY viewer | Hero video on fr1_360 (fr1_desk if time allows), fused 3D maps, figures | Visual checks against GT overlays. `scripts/figures.py` must be written |
| 3 Compute sweep | Runner changes (below), a single T4 with nothing else running | Latency, memory and accuracy per setting (steps, token budget, bf16 vs fp16, L vs B) | Numerical agreement with baseline on the 200 subset frames first, then `model_latency_ms` from single-process runs |
| 4 Final deliverables | All of the above | Report, README, LinkedIn post, email to the authors | Every number traced to a file in `results/` |

**Fusion design is not in the repo.** No fusion code, config or test exists yet. Two points to settle first.
- Aligning every frame to its own GT (as M1 and persistence do) leaks GT scale into a "fix". A deployable fusion should align frames to each other, like M2, with GT used at most once for evaluation.
- Using GT poses for warping is an oracle. It should be stated as such, or replaced with estimated poses.

Persistence suggests the ceiling. At 33 ms about 90% of error is shared between neighbours, so short-window fusion can only remove the remaining view-dependent part and flicker. Larger windows remove more, but bring in more pose and occlusion error.

**Compute sweep needs code changes.** `runner.py` hard-codes `torch.bfloat16` autocast, the `PointDiT-L/16` flags in `DEMO_L_512_ARGS`, the L checkpoint in `paths.py`, and the step count (`--num_sampling_steps 3`). `predict.py` `load_run_config` accepts only `init`, `seed` and `seed_mode`, so new config keys (dtype, steps, tokens, model) need parsing, `run.json` support and tests. Note that the T4 has no native bf16 (`report/notes.md`).

### Places where the reports and code disagree slightly

- Sampling, 33 ms. `temporal_consistency.md` says "most CIs exclude no change". All six do.
- Sampling, 100 ms. "Significant on fr3 only" holds for 2 of 3 fr3 seeds (seed 0 CI [0.991, 1.113]).
- Sampling, 333 ms. "Not at 333 ms" has one exception, fr3 `noise_seed1` 1.015 [1.003, 1.171]. Sampling uses only L = 15, not the 8/15/30 rule.
- The "about √2" ratio between seed-seed and zeros-seed disagreement is about 1.25–1.35 in `results/sampling/summary.md`.
- Drivers. "Edge density ρ 0.25–0.53" covers three sequences. fr1_desk is 0.17 and not robust. fr1_360 also shows a robust negative ρ with expected misalignment (−0.18), which the report does not mention.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| Point map | An H×W×3 array giving a 3D point (X, Y, Z) in camera coordinates for every pixel |
| Affine-invariant | The prediction is correct only up to an unknown global scale and shift. Points use one scale and a 3D shift, depth uses a scale and a shift. It must be aligned before comparison |
| Alignment | Solving for that scale and shift against a reference (GT or another frame), here by weighted L1 on a 64×64 subsample |
| Rel_d, Rel_p | Mean relative depth or point-distance error after alignment, ×100 |
| δ1 | Fraction of pixels within a tolerance (ratio 1.25 for depth, a 25% distance for points), ×100 |
| Kept-frame index | Position of a frame in the associated, pose-filtered list of a sequence |
| Hole | A gap in the kept frames more than 1.5× the median spacing (a dropped frame) |
| Depth offset | t_depth − t_rgb for a frame. Up to about 20 ms on fr1, about 0.1 ms on fr3 |
| Depth-timestamp pose | GT pose interpolated at the depth image's timestamp rather than the RGB's |
| Forward warp | Moving frame i's depth pixels into frame j via 3D points and the relative pose, keeping the nearest per pixel (z-buffer) |
| Visible pixels | Pixels of frame j hit by frame i's warped GT whose depth agrees with GT_j within 15% (same surface, not occluded) |
| Interior pixels | Visible pixels away from GT depth edges (outside the 5 px edge band) |
| Edge band | Valid pixels within 5 px of a GT depth jump (> 4%) or of a GT hole |
| GT floor | Disagreement of GT depth with itself across two frames under the same warp. It measures Kinect noise, pose error and warping error |
| M1 | Disagreement between two frames' predictions, each aligned to its own GT, measured on visible pixels |
| M2 | Disagreement between frame j's prediction fitted to frame i's warped prediction and that warped prediction. This is pure self-consistency |
| M2 floor | M2 computed with GT in place of predictions. It is the best achievable M2 given the data |
| M2 / M2 floor | How many times less consistent than GT the model is. 1 means as consistent as GT |
| Excess | M2 − M2 floor for one pair |
| E_i, E_j | Signed relative errors against GT_j of frame i's warped prediction and of frame j's own prediction |
| Persistent fraction P | Share of error variance common to both frames. 1 means the same error in both, 0 means unrelated |
| Flicker event | A pair whose per-third M2 jumps well above its neighbouring pairs |
| Floor control | Checking whether a GT-floor event happens at the same place. If not, the event is attributed to the model |
| Spike | A frame whose Rel_d jumps well above its ±5 neighbours |
| Spearman ρ | Correlation of ranks, robust to outliers and monotone transforms |
| Partial Spearman | Spearman after removing the linear effect of control variables' ranks |
| Pooled fr1 | The three fr1 sequences combined after ranking within each, so sequence differences do not create correlation |
| Moving-block bootstrap | Resampling contiguous blocks of frames (or pairs) to keep autocorrelation, which gives honest CIs |
| Zero init / noise init | Starting the diffusion sampler from zeros (deterministic) or from Gaussian noise (seeded) |
| Seed disagreement | Same-frame difference between two predictions that differ only in starting noise |
| Expected misalignment | Estimated pixel shift between RGB and depth caused by camera motion during the depth offset |
| Relative sharpness | Laplacian variance divided by its local median over ±0.5 s. Below 1 means blurrier than nearby frames |
