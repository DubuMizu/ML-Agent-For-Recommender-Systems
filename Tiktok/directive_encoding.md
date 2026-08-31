PRIORITY TASK: settle whether a better ENCODING beats a better model.

Every direction this run has explored so far changed the model or the objective,
and all of them landed inside the noise floor. The encoding itself was never a
variable — it was hard-coded and out of reach. It now is a variable, and there
is measured evidence that the default throws information away.

WHAT WAS MEASURED (validation, signals scored ALONE as rankers)
  reference: random GAUC 0.4993 | item popularity 0.6387 | FM 0.6674

  duration-bucket rate,  10 buckets (the default)   GAUC 0.5319
  duration-bucket rate,  20 buckets                 GAUC 0.5341
  duration-bucket rate,  50 buckets                 GAUC 0.5443
  duration-bucket rate, 200 buckets                 GAUC 0.5582
  video long_view rate                              GAUC 0.6387
  video x tab rate                                  GAUC 0.6479

  long_view is defined against duration, so the 10-bucket discretisation is
  lossy; and conditioning the item prior on the surface it was shown in is worth
  +0.0092 standalone.

WHAT WAS ALSO MEASURED — READ THIS BEFORE YOU GET OPTIMISTIC
  A naive 2-epoch A/B of both options came out NEUTRAL to slightly NEGATIVE:
    baseline (5 fields)   valid 0.6032   seed-ensemble 0.6038
    dur_buckets=50        valid 0.6031   seed-ensemble 0.6035
    + video_id x tab      valid 0.6023   seed-ensemble 0.6028

  That is not a verdict, and it is not a reason to skip the task. Both options
  ADD PARAMETERS, and this regime is overfitting-limited — that is the single
  most robust finding in the whole journal. A new field at the old p/l2 is
  simply under-regularised, so the naive A/B measured the wrong thing. It is a
  warning that a fixed-hyper-parameter comparison here will mislead you.

STEP 1 — TUNE THE ENCODING JOINTLY WITH THE REGULARISATION
  Use `tune` (BO + ASHA), not hand-stepped run_experiment calls. Put the
  encoding knob IN THE SEARCH SPACE alongside lr, p and l2, so the optimiser can
  find the regularisation that a wider encoding needs:

    space = {"features.dur_buckets": ["cat", [10, 20, 50, 200]],
             "train.lr": ["float", 3e-4, 2.5e-3, true],
             "model.p":  ["float", 0.20, 0.75, false],
             "train.l2": ["float", 1e-6, 3e-3, true]}

  Both forms are verified to work: dotted keys route into the features block,
  and the categorical space round-trips correctly for bucket counts and for
  nested crosses like ["cat", [[["video_id","tab"]], [["author_id","tab"]]]].
  So put the encoding knob in the space directly — do not hand-step it.

STEP 2 — THE SAME FOR CROSSES
  Test crosses the same way: [["video_id","tab"]], and if that pays, also
  [["author_id","tab"]] and [["user_id","dur_bucket"]]. The last one is the only
  cross that is a genuine user x item-attribute conjunction, which is the one
  family of terms that survives within-user cancellation.

STEP 3 — JUDGE ON THE SEED ENSEMBLE
  run_experiment now reports SEED ENSEMBLE valid primary alongside the mean of
  per-seed metrics. The seed ensemble is what a submission of that config
  actually scores, and it is the less noisy of the two. Compare on it. The noise
  floor is ~0.0012; below that it is a tie and you must say so.

STEP 4 — REPORT A VERDICT
  record_finding tagged 'encoding', stating for each option: better / tie /
  worse, the tuned numbers, and what the unbiased log did. A clean negative is a
  complete answer — the encoding was never testable before, so settling it
  either way is worth the iterations.

STEP 5 — THEN USE IT
  If an encoding wins, pass it as the shared `features` argument to
  ensemble_search and rebuild the portfolio on it. The portfolio is still the
  only lever that has reliably paid (+0.0016 over the best single, reproduced
  twice), so the run should end with a portfolio built on the best encoding you
  found — not with a single model.

DEAD ENDS — DO NOT SPEND ITERATIONS RE-TESTING
  Measured on validation while preparing this task:
   * user x author affinity is UNESTIMABLE: only 3.4% of validation (user,
     author) pairs appear in that user's train history, and scored alone it
     ranks at GAUC 0.4982, i.e. random. user x video is 1.6%. This is also why
     DIN and the whole behaviour-sequence direction only tied — a user's history
     almost never contains the candidate's author. Do not revisit it.
   * Recency weighting of training rows does nothing: video-rate GAUC is 0.6370
     / 0.6389 / 0.6389 across half-lives versus 0.6387 flat. The train-to-valid
     shift is a level shift, and levels cancel under within-user ranking.
   * Blending priors as scores is worse than the best single prior
     (0.75*video + 0.25*duration = 0.6382 against 0.6387 for video alone). They
     have to enter as model features, not as blended predictions.

BUDGET
  Aim to have the encoding verdict recorded within roughly the first 30
  iterations, then spend what remains on the portfolio.
