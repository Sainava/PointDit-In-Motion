# Oracle scale fusion, first readout (provisional)

Run. Kaggle CPU, 100 targets per sequence, 36 configurations, scale = oracle GT alignment, so every number is an upper bound. Results are in results/fusion/oracle_eval/baseline/. The regression against per_frame.csv passed on 400 frames (maximum difference 1.1e-6, tolerance 1e-4). Values are medians over targets and are not yet bootstrapped.

Findings
- Gate 1 passes provisionally. fr3 reaches -3.5% (causal adjacent w4) to -6.6% (causal diverse w4, median aggregator). fr1_desk reaches -5.8% only for centred adjacent w4 with the front layer.
- Controls. Shuffled poses are 35% to 140% worse. Pose noise hurts monotonically, and the fr3 gain is lost at about 10 mm or 0.5 degrees.
- Short windows match the pre-registered forecast on fr3 and fr1_room. Distant sources reach 30% to 55% of the forecast, and fr1_desk gets worse as causal windows reach further back.
- GT ceilings (noself) show the cause. Warp error is 0.5% to 1.4% in the interior and 2.5% to 5.5% in the depth edge band, comparable to the model's own error. Fusion helps in the interior and adds noise at edges.
- The median aggregator beats the mean on all four sequences. The front layer is inconsistent across sequences.

Limits. One 100 target region per sequence, autocorrelated targets, oracle scale, GT poses.


cat >> report/fusion_oracle_readout.md <<'EOF'

## Edge aware follow up (correction)

Hypothesis. Fusion helps interior pixels and adds noise at depth edges, so masking source edges or falling back to the target's own depth at edges should raise the gain.

Result. Refuted. Every edge variant is worse than the plain median in all four sequences. Interior and edge diagnostics show the gain sits mainly at depth edges on fr3 and fr1_room (edge error -10 to -14 percent, interior -1 to -3 percent), where the model's error is large and view dependent. Interior error is small and shared between frames. On fr1_desk, distant sources worsen the interior by about 6 percent. The earlier statement that fusion adds noise at edges came from reading the GT ceiling rows without comparing them to the model's own edge error, and it was wrong.

Best deployable candidate so far. Causal adjacent window 4 with the median aggregator changes median Rel_d by -2.2, -3.5, -6.9 and -5.5 percent on fr1_desk, fr1_room, fr1_360 and fr3 (oracle scale, 100 targets).

The one frame GT ceiling (0.4 to 0.6 percent interior, 2.2 to 2.7 percent edge) shows the splat is not the limiting factor.
EOF
git add report/fusion_oracle_readout.md results/fusion/oracle_eval_edge
git commit -m "docs - correct edge hypothesis in oracle fusion readout"
git push origin feat/temporal-fusion