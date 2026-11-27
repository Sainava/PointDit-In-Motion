# Oracle scale fusion, first readout (provisional)

**Status.** The first two sections are provisional medians and are kept as written for the record. Statements marked *[Superseded]* or *[Refuted]* are replaced by the bootstrap analysis and the edge follow up further down. Every fusion number in this file uses ground truth motion capture poses and the 100 targets at window positions 100 to 199 of each sequence, so it is an upper bound for a system with estimated poses.

Run. Kaggle CPU, 100 targets per sequence, 36 configurations, scale = oracle GT alignment, so every number is an upper bound. Results are in results/fusion/oracle_eval/baseline/. The regression against per_frame.csv passed on 400 frames (maximum difference 1.1e-6, tolerance 1e-4). Values are medians over targets and are not yet bootstrapped.

Findings
- Gate 1 passes provisionally. *["Gate 1" is not defined in any file of the repository.]* fr3 reaches -3.5% (causal adjacent w4) to -6.6% (causal diverse w4, median aggregator). fr1_desk reaches -5.8% only for centred adjacent w4 with the front layer.
- Controls. Shuffled poses are 35% to 140% worse. Pose noise hurts monotonically, and the fr3 gain is lost at about 10 mm or 0.5 degrees. *[Superseded. Bootstrap means put shuffled poses at 60 to 212 percent worse (results/fusion_analysis/controls.csv). Pose noise was tested only on the oracle causal_diverse_w4_rung_d configuration, whose fr3 gain is itself not robust (mean -1.7 [-7.8, 2.4]). Against that configuration, fr3 gets worse by 6.9 percent at 10 mm and 11.2 percent at 0.5 degrees. Pose noise was not tested on the deployable joint solved configuration.]*
- Short windows match the pre-registered forecast on fr3 and fr1_room. Distant sources reach 30% to 55% of the forecast, and fr1_desk gets worse as causal windows reach further back. *[Superseded. The 30 to 55 percent figure holds only for fr3 and fr1_room, see the bootstrap section.]*
- GT ceilings (noself) show the cause. Warp error is 0.5% to 1.4% in the interior and 2.5% to 5.5% in the depth edge band, comparable to the model's own error. Fusion helps in the interior and adds noise at edges. *[Refuted by the edge aware follow up below. The gain sits mainly at depth edges on fr3 and fr1_room.]*
- The median aggregator beats the mean on all four sequences. The front layer is inconsistent across sequences. *[Superseded. This holds for median changes. For mean changes the median aggregator is worse on fr1_360 for both tested windows (causal diverse w4 +4.0 vs +3.9 percent, centred adjacent w4 -6.9 vs -7.7 percent, results/fusion_analysis/summary.md).]*

Limits. One 100 target region per sequence, autocorrelated targets, oracle scale, GT poses.

## Edge aware follow up (correction)

Hypothesis. Fusion helps interior pixels and adds noise at depth edges, so masking source edges or falling back to the target's own depth at edges should raise the gain.

Result. Refuted. Every edge variant is worse than the plain median in all four sequences. Interior and edge diagnostics show the gain sits mainly at depth edges on fr3 and fr1_room (edge error -10 to -14 percent, interior -1 to -3 percent), where the model's error is large and view dependent. Interior error is small and shared between frames. On fr1_desk, distant sources worsen the interior by about 6 percent. The earlier statement that fusion adds noise at edges came from reading the GT ceiling rows without comparing them to the model's own edge error, and it was wrong.

Best deployable candidate so far. Causal adjacent window 4 with the median aggregator changes median Rel_d by -2.2, -3.5, -6.9 and -5.5 percent on fr1_desk, fr1_room, fr1_360 and fr3 (oracle scale, 100 targets).

The one frame GT ceiling (0.4 to 0.6 percent interior, 2.2 to 2.7 percent edge) shows the splat is not the limiting factor.


## Deployable result, joint solved scale (provisional)

Setup. Scale is estimated from the window itself (robust parallax solve, no ground truth). Poses remain the ground truth motion capture poses. Results are median per target changes in Rel_d over 100 targets, scored after the standard per frame alignment of the final depth, so they are compared with baseline_self and not with the oracle rows, which were scored under a different protocol.

Findings
- Causal adjacent window 4 changes Rel_d by -4.5, -2.3, -5.8 and -3.8 percent on fr1_desk, fr1_room, fr1_360 and fr3, with 64 to 76 percent of targets improved. *[Superseded. These are medians. The bootstrap means are -2.4, -4.5, -5.9 and -4.3 percent, see below.]*
- Controls. Shuffled poses are 42 to 147 percent worse. *[Medians. Bootstrap means are 58 to 193 percent worse.]* The no fusion control equals baseline within 1.5e-6.
- The solver recovers depth only to about 8 to 12 percent and scale correlations with the oracle are low. On fr1_360, 66 of 100 windows are rotation dominated and use the shared fallback fit. *[Corrected. For causal adjacent w4 the median target depth error is 9.8, 12.6 and 8.4 percent on fr1_desk, fr1_room and fr3, and about 400 percent on fr1_360. Scale correlations are 0.10, 0.55, -0.37 and 0.19 (results/fusion/joint_eval/baseline/joint_fusion_summary.csv). The fused depth is affine aligned to the target GT before scoring, which absorbs most of this scale error.]*
- Distant sources (diverse windows) help on fr3 and fr1_room but not reliably on fr1.

Limits. Medians, no bootstrap yet, oracle poses, 100 autocorrelated targets per sequence.


## Bootstrap analysis (supersedes the provisional medians above)

Method. Per target percent change in rel_d against baseline_self, mean and median, moving block bootstrap over targets (2000 resamples, blocks 8, 15 and 30). A claim is robust only when all three intervals exclude zero. Results are in results/fusion_analysis/.

Deployable result (joint solved scale, causal adjacent window 4, ground truth poses). Pooled mean change -4.3 percent [-6.6, -2.1], robust. Per sequence mean change fr1_room -4.5 (robust), fr1_360 -5.9 (robust), fr3 -4.3 (robust), fr1_desk -2.4 (not robust, interval -7.7 to 4.2). These are relative changes on the 100 middle targets of each window. In Rel_d points the pooled change is -0.25 [-0.37, -0.13].

Measured versus forecast. Short adjacent windows match the forecast on fr3, fr1_room and fr1_360. Longer and more distant windows fall short, and on fr1_desk and fr1_360 the diverse windows give no gain or a loss in point estimate. The forecast assumed warp error stays small. It does not on fr1_desk, which has the largest warp error and fastest motion.

Corrections. The earlier remark that joint solved beat the oracle on fr1_desk is not meaningful, since neither effect is robust there. The earlier claim that distant sources reach 30 to 55 percent of the forecast holds only for fr3 and fr1_room.

## Per frame share (full 300 frame windows, oracle poses, joint solved scale)

Fusion lowers rel_d on 74.0 percent of frames on fr1_360 and 59.5 percent on fr1_desk (median change -5.6 and -2.4 percent). Mean change is -4.05 [-7.57, +0.21] and -2.23 [-4.86, +0.09] percent, so over the full window the mean effect is small and not robust on either sequence. The 10th and 90th percentile changes are -21 and +12 percent on fr1_360 and -17 and +13 percent on fr1_desk. The longest run of consecutive worse frames is 9 on both sequences. Frame 23 on fr1_360, which showed higher disagreement in the flicker overlay, has a lower rel_d under fusion (rank 222 of 296 from worst).
