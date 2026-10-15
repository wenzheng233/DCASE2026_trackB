# Next clean metric push

The current evidence says the best immediate path is not another large
architecture change. The 2023 same-input model gives a large gain even in
`keep-score` mode, so the current bottleneck is mostly category/count ranking
and calibration on top of usable v16 proposals.

Recommended next experiments, in priority order:

1. Sweep lightweight count/score calibration on cached predictions.
   - Keep the v16 masks fixed.
   - Use only 2023 same-input prior outputs and train-only logic where possible.
   - Try per-class score thresholds after `score1`, but avoid dev-test-specific
     manual class routes in reportable claims.

2. Distill the same-input 2023 teacher into v16 classification.
   - Keep localization/mask heads mostly frozen.
   - Use weak-class balanced sampling.
   - Use low-weight teacher class KL or pseudo-label targets for matched ROIs.
   - Goal: recover the inference-time prior gain inside the model.

3. Continue clean v16 only if the validation proxy improves early.
   - Do not run long unfreeze training unless a small cached/proxy run shows a
     real gain.
   - Previous full unfreeze continuations were not better than v16 epoch1.

4. Keep the compact per-class count candidate as the packaging-safe backup.
   - It has the best micro mAP.
   - It keeps all JSON files far below the 20 MB risk point.

