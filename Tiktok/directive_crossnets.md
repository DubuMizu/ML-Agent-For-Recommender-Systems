PRIORITY TASK: settle whether explicit feature crossing beats FM on this benchmark.

The journal contains a negative on `deepfm` (iterations 5 and 6, 0.5948 and
0.5983, both below the FM baseline). Do not trust it. Both runs peaked at
best_epoch=1 on every seed and used p=0.2, l2=1e-06 — they were measured
entirely inside the overfitting-limited regime that was only diagnosed at
iteration 11 and fixed at iteration 12. That result tells you the MLP overfits
fastest; it does not tell you that explicit crossing is worthless here.

Two function classes in that family have never been tried at all. Settle them.

STEP 1 — RESEARCH BEFORE YOU WRITE
  Use WebSearch/WebFetch to get the actual formulations of:
    * DCNv2 / CrossNetV2 (Wang et al., 2021) — the explicit cross layer
      x_{l+1} = x_0 * (W x_l + b) + x_l, including the low-rank variant.
    * xDeepFM / CIN (Lian et al., 2018) — the compressed interaction network,
      vector-wise Hadamard products against x_0 followed by sum pooling.
  Both have details that are easy to get subtly wrong from memory — CrossNetV2's
  low-rank projection, CIN's feature-map compression and its pooling step. A
  wrong implementation produces a negative result you would then wrongly trust,
  which is the exact failure mode this task exists to correct. Read first.

STEP 2 — IMPLEMENT
  Add both to agent_kit/models.py with @register, following the existing FM and
  DeepFM classes for the constructor convention and the forward(X) contract.
  Expose their structural knobs as config keys (cross depth, CIN layer sizes,
  low-rank dimension) so they can be searched.

STEP 3 — VERIFY THEY TRAIN, CHEAPLY
  Before spending search budget, run ONE short run_experiment per architecture
  (small epochs, 1-2 seeds) purely to confirm it trains, produces finite loss,
  and beats random. A syntax error or a shape bug found here costs one cheap
  iteration; found inside a tune sweep it costs the whole sweep. Fix and re-run
  as needed — a failure here is expected and is not a verdict.

STEP 4 — SEARCH EACH ONE PROPERLY, WITH BO + ASHA
  Use the `tune` tool, not a hand-stepped sequence of run_experiment calls.
  `tune` is Bayesian optimisation (GP + expected improvement, observation noise
  fixed at the measured seed SD) stacked with ASHA pruning, which is exactly
  what this comparison needs: these architectures have more knobs than FM, and
  hand-stepping them would spend the run's whole budget and still under-explore.

  For each architecture, tune over BOTH its structural knobs and the
  regularisation that the run has already shown to be the binding constraint —
  train.lr, model.p, train.l2 at minimum. Start from the iteration-22 recipe
  (p≈0.34, l2≈1.2e-05, lr≈0.00116, group_size=4, listwise softmax loss), not
  from deepfm's original under-regularised settings. Give each architecture a
  comparable trial count so the comparison is fair.

STEP 5 — REPORT A VERDICT
  Compare each architecture's best tuned result against the fm_softmax family's
  best single model already in the journal (~0.6040), which was produced by the
  same tune/BO+ASHA machinery, so the comparison is like-for-like. The noise
  floor is ~0.0012: a difference smaller than that is a TIE, and you must say so
  rather than declaring a winner.

  Record the verdict with record_finding, tagged 'crossnets', stating for each
  architecture: better / tie / worse, the numbers, and what the unbiased log did.
  A clean negative is a complete answer to this task and is worth recording
  properly — the operator asked whether these are better, and "no, and here is
  the evidence" is a real result.

STEP 6 — THEN USE WHAT YOU LEARNED
  If either architecture is competitive (within or above the noise floor of the
  FM family), add it as a FAMILY in your next ensemble_search. Remember the
  documented lesson from this run: din_dropout scored identically to fm_dropout
  alone and was still among the most valuable ensemble members, because its
  errors were decorrelated. An architecture that merely ties on its own can
  still earn weight in the portfolio, so a tie is a reason to include it, not a
  reason to discard it.

BUDGET
  This is the opening of the run, not the whole run. Aim to have the verdict
  recorded within roughly the first 25 iterations. If both architectures are
  clearly worse after a fair search, stop, record the negative, and go back to
  your own agenda — do not keep rescuing a losing direction.
