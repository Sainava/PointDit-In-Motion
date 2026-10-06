# Week 1: Data pipeline, metrics and baseline evaluation

## Setup

- **Model:** PointDiT-L/16, 512 checkpoint (`pointditl-512-mixdata-nodinov3-240c1a4f.pth`), EMA weights, frozen DINOv3 ViT-L/16 encoder.
- **Inference:** upstream defaults — 3 sampling steps, bf16 autocast, default token budget (TUM frames run at 448×592), zero initialisation (upstream default, deterministic).
- **Frames:** full 640×480 frames, aspect-preserving resize to the token budget; predictions resized back to 640×480.
- **Data:** TUM RGB-D, four sequences, one 300-frame window each, selected automatically as the stretch with the fewest dropped frames.

| Sequence | Window (kept-frame indices) | Holes in window | Holes in sequence |
|---|---|---|---|
| fr1_desk | 209–508 | 0 | 23 |
| fr1_room | 366–665 | 0 | 9 |
| fr1_360 | 0–299 | 0 | 12 |
| fr3_long_office_household | 637–936 | 5 | 111 |

fr3 has no hole-free 300-frame stretch; its 5 holes are single dropped frames (61–68 ms gaps vs a 32 ms median).

- **Intrinsics:** TUM's ROS defaults (525, 525, 319.5, 239.5) for fr1, calibrated values (535.4, 539.2, 320.1, 247.6) for fr3, both without undistortion, following TUM's recommendation for its pre-registered depth.

### Provenance

| Component | Commit |
|---|---|
| Project (predictions) | `5959d67` |
| Project (evaluation) | `<eval commit>` |
| PointDiT fork (Sainava/Sai-pointdit) | `664736a` |
| DINOv3 code | `6876159` |

- Environment: torch 2.11.0+cu128, CUDA 12.8, Kaggle Tesla T4.
- Predictions: Kaggle runner notebook, version `<n>` (1,200 frames, two T4s in parallel).
- Evaluation: Kaggle CPU evaluation notebook, version `<n>` (4.47 s/frame).

## Validation

1. **Sampler determinism.** The runner's default reproduces the original smoke test (max diff 2e-8, identical masks). Zero init run twice: identical. Noise init with seeds 0 vs 1: max diff 0.92. Same seed twice: identical.
2. **Metrics match the paper's evaluation.** On 10 Booster scenes, the project's metric wrapper and upstream's evaluation give identical per-scene values (max |diff| = 0), and the aggregates reproduce the original run exactly: Rel_p 3.001, δ1_p 99.939, Rel_d 1.995, δ1_d 99.937.
3. **Per-frame bookkeeping.** For one frame per sequence, an independent numpy recomputation of all four metrics matches the CSV within ~2.4e-6, with identical valid-pixel counts.
4. **GT geometry.** Back-projected TUM GT point clouds look correct: flat desk and floor planes, objects with correct thickness. RGB–depth registration checked with depth-edge overlays: tight on fr3, a few pixels off on fast fr1 frames (consistent with the fr1 depth–RGB offset).

## Results

Metrics follow upstream's protocol: per-frame affine alignment (scale + 3D shift for points, scale + shift for depth, weighted L1), GT mask only, ×100.

| Sequence | Rel_p mean / median | δ1_p mean / median | Rel_d mean / median | δ1_d mean / median |
|---|---|---|---|---|
| fr1_desk | 5.19 / 4.76 | 97.65 / 97.90 | 4.24 / 3.91 | 97.63 / 97.86 |
| fr1_room | 8.10 / 7.52 | 94.39 / 95.40 | 5.99 / 6.06 | 94.80 / 95.52 |
| fr1_360 | 6.27 / 6.25 | 96.26 / 96.87 | 5.09 / 5.02 | 96.59 / 97.02 |
| fr3_long_office | 4.65 / 4.60 | 98.18 / 98.65 | 3.66 / 3.79 | 98.22 / 98.76 |

No frames were skipped. Means and medians are close everywhere, so no heavy outliers dominate the averages.

**Not comparable to the paper's tables.** The paper reports Rel_d 3.09 / δ1_d 98.25 averaged over 7 benchmarks, using a 512 centre crop. These numbers use full frames on different, real Kinect data.

**Reading the table:**
- PointDiT transfers well zero-shot to TUM: roughly 4–6% depth error on real handheld Kinect footage.
- fr3 is easiest, consistent with slow motion, synchronised depth (~0.1 ms offset) and calibrated intrinsics.
- fr1_room is worse than fr1_360, although fr1_360 has the fastest rotation. Motion is therefore not the only factor; fr1_room's larger space and wider depth range is a candidate (hypothesis).
- fr1 uses approximate intrinsics, which distorts GT point maps and inflates Rel_p more than Rel_d (Rel_p − Rel_d ≈ 2.1 on fr1_room vs ≈ 1.0 on fr3). **Rel_d is the cleaner metric for fr1 vs fr3 comparisons.**

## Error over time

Figures: `report/figures/data_metrics/<sequence>_metrics.png`.

- **fr1_360:** several event-like rises; the largest sustained event around 7–9 s, Rel_d up to ~12%. Rel_p follows the same pattern.
- **fr1_desk:** jagged, with short spike clusters rather than drift. Main events around 3.5–5.5 s and 8.3–9 s, up to ~10%.
- **fr1_room:** highest error at the start of the window (~12–15%), declining gradually, then later events, notably around 5.5–6.2 s.
- **fr3_long_office:** stable, mostly 3–5%, short fluctuations around 4–6 s, maximum ~6%. Serves as the control sequence.

**Errors occur as clustered events, not isolated outliers.** Consecutive frames fail together.

## Failure clusters

Each cluster records what was observed in the preview panels (RGB | predicted depth | GT depth), then the current interpretation. Interpretations are hypotheses until tested quantitatively.

**General correction:** GT invalid (black) pixels are excluded from the metrics, so they remove pixels from the average rather than adding error. All worst frames have ordinary valid fractions (0.75–0.86), so GT holes explain none of the clusters. GT noise at edges and far range is real but affects all frames roughly equally, so it cannot explain spikes.

**Limitation of the previews:** each depth panel is normalised to its own 2nd–98th percentile, which hides global scale and shift differences and makes moderate shape errors hard to see. "The layout looks right" and "Rel_d is 14%" can both be true. An error-map panel is needed for reliable failure analysis.

### fr1_room, frames 370–374 (Rel_d 12.6–14.7%, worst overall)

**Observation:** RGB reasonably sharp, no strong smearing. Main structure correctly placed (table left, central vertical structure, wooden structure right). Prediction smoother than GT; some depth boundaries don't line up. Depth–RGB offset 13–17 ms. Rel_p well above Rel_d (~19 vs ~14).

**Interpretation:** first read as GT/timing misalignment. **Revised: likely model error in relative depth.** A 15 ms offset shifts edges by a few pixels, affecting a thin band around boundaries; a mean Rel_d of 14% over ~250k pixels means large regions are wrong. The large Rel_p − Rel_d gap and the gradual decline at the start of the window both suggest a difficult viewpoint where the relative depth between scene parts (e.g. near furniture vs far wall) is off. **To confirm with error maps:** error spread over regions → model; concentrated on edges → timing.

### fr1_360, frames 233–243 (Rel_d 10.4–11.9%)

**Observation:** clear motion blur, especially frames 233 and 238; doorway, shelf and object edges smeared. Broad structure correct; fine structures and thin objects smoothed away. Depth–RGB offset 0 ms.

**Interpretation:** **the strongest candidate for a blur-driven failure.** Timing cannot explain it (0 ms offset). Caveats: five frames judged by eye is anecdotal, and fast rotation also brings new viewpoints with large depth discontinuities, so blur and scene content are confounded. Needs quantitative testing.

### fr1_desk, frames 337–338 (Rel_d 9.7–10.2%)

**Observation:** RGB reasonably clear, slight softness only. Broad desk geometry correct; prediction too smooth, misses or merges thin monitor, cable and object structures.

**Interpretation:** mostly model limitation in fine geometric detail, not blur.

### fr1_desk, frames 362–363 (Rel_d 8.2–8.4%)

**Observation:** RGB clearly softer than 337–338, especially around monitors and objects. Broad arrangement correct; fine structure smoothed away.

**Interpretation:** blur plus model limitation. Contrasts usefully with 337–338: similar scene, different blur level.

### fr3_long_office, frames 648–776 (Rel_d ~5.9–6.1%)

**Observation:** frame 648 slightly soft; frames 761–776 clear, with predictions stable across consecutive frames. Prediction smoother than GT, no geometry failure.

**Interpretation:** normal model limitation (fine-detail smoothing). No blur or timing event.

### Summary

| Cluster | Current interpretation |
|---|---|
| fr1_room 370–374 | Likely model error in relative depth (revised from GT/timing); confirm with error maps |
| fr1_360 233–243 | Blur-dominant, plus fine-detail loss; best blur candidate |
| fr1_desk 337–338 | Model fine-detail limitation |
| fr1_desk 362–363 | Blur + model limitation |
| fr3 648–776 | Normal model limitation |

Three error sources are present and entangled in fr1: GT timing (depth–RGB offset), motion blur, and model limits (fine detail, relative depth). They need to be separated statistically rather than by eye.

## Open questions for Week 2

1. Add a per-pixel error panel (|aligned prediction − GT| / GT) to the previews; re-examine fr1_room 370–374 with it.
2. Quantify blur: per-frame camera speed (from GT poses) and image sharpness (Laplacian variance); relate them to Rel_d within each sequence.
3. Separate timing from blur: repeat the blur analysis restricted to frames with a small depth–RGB offset (e.g. < 5 ms).
4. Explain fr1_room > fr1_360: test scene depth range as a factor.
5. Compute sweep (model size, steps, token budget, bf16 vs fp16): measure latency with a single process on one GPU. The baseline run's latencies (1,674–1,844 ms) are inflated by two concurrent processes and must not be reported.