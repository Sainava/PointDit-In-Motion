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