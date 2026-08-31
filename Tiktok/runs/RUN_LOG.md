# Agent run log

Auto-generated from `runs/journal.jsonl`; one row per iteration.

- FM baseline (validation primary): **0.6016**
- Oracle ceiling (validation): **0.8484**
- Best so far: **0.6045** (+0.0029 vs FM) at iteration 139
- Iterations: **140** (23 recovered failures)

| # | status | valid primary | GAUC | nDCG@5 | Δ vs FM | unbiased | secs | hypothesis |
|---|--------|---------------|------|--------|---------|----------|------|------------|
| 1 | ok | 0.6017 | 0.6676 | 0.5359 | +0.0001 | 0.3642 | 50 | Reproduce the FM baseline in the torch harness (control). |
| 2 | ok | 0.6021 | 0.6685 | 0.5357 | +0.0005 | 0.3712 | 53 | Swap pointwise log-loss for within-user BPR: the metric ranks within a user, so put the gradient on within-user comparisons. |
| 3 | ok | 0.6023 | 0.6688 | 0.5357 | +0.0007 | 0.3742 | 157 | Listwise softmax with 8 sampled negatives per positive, same user. |
| 4 | ok | 0.6017 | 0.6678 | 0.5356 | +0.0001 | 0.3753 | 178 | LambdaRank weighting pairs by \|delta nDCG@5\|, truncated at k=5 like the metric. |
| 5 | timeout | 0.5948 | 0.6577 | 0.5319 | -0.0068 | 0.3658 | 434 | Replacing FM's purely bilinear user x item interaction with DeepFM (FM + MLP over the concatenated field embeddings), trained with the best-so-far within-user softmax listwise loss, will lift the primary metric above 0.6023 because the ranking signal is limited by the *form* of the interaction, not by its rank. |
| 6 | ok | 0.5983 | 0.6628 | 0.5338 | -0.0033 | 0.3684 | 158 | Same DeepFM-vs-FM function-class test as iteration 5, but with a small enough MLP and listwise group that training actually converges inside the 420s budget, so the comparison against 0.6023 is real. |
| 7 | note | – | – | – | – | – | – | NEGATIVE, clean: the deep-cross function class hurts. iter 6 (deepfm k=16, hidden 64-32, p=0.2, softmax, group_size=4) converged properly in 158s and scored 0.5983 +-0.0004 vs 0.6023 for plain FM+softmax -- a -0.0040 gap, ~8x the 0.0005 seed SE, so this is a result and not noise. The unbiased log agrees (0.3684 vs 0.3742), so it is not a logging-policy artifact either: the MLP is genuinely worse at ranking, not just worse on validation. Diagnostic: best_epoch=1 on all three seeds and the epoch curves fall monotonically after epoch 1 (0.5989 -> 0.5798 -> 0.5745 -> 0.5711). The MLP memorises the user_id x video_id pairs within a single pass and then destroys generalisation; dropout 0.2 did not contain it. This is consistent with the prior knowledge that 1.14M rows cannot support more parameters -- that constraint binds on ANY added capacity, not just on higher k. iter 5 was the same config with a 128-64 MLP and hit the 420s TIMEOUT at epoch 1; its 0.5948 measures compute starvation and should not be read as evidence about anything. DO NOT RETUNE THE MLP (wider/narrower/more dropout/lower lr) -- the failure mode is one-epoch memorisation, which more regularisation only slows down.

Where this leaves the search. Two axes are now closed. (a) The objective: bce 0.6017, bpr 0.6021, softmax 0.6023, lambdarank 0.6017 -- a 0.0006 spread inside the noise band, so the loss is no longer the binding constraint. (b) Free-parameter capacity: k=8/16/32 flat (given), and now the MLP cross negative. What is still untouched is INFORMATION, not capacity -- every model so far sees only the static id fields, and the two directions that add signal rather than parameters are still unopened: (3) user behaviour sequences / DIN target attention, and (4) multi-task auxiliary heads (is_click / is_like / is_forward / play_time_ms) that regularise the sparse long_view target with correlated dense labels. Direction 4 is the better next move: it adds supervision instead of parameters, which is exactly the axis that has not been shown to fail, and it attacks label sparsity, which is the plausible reason every function class saturates at ~0.602. Direction 3 needs a per-user history tensor reconstructed from time_ms, but forward(X) receives only (B,F) id fields, so it needs the history folded into X and is the more expensive build. Note also that the hidden test is a LATER time window, and an aux-head model regularised by dense labels should be more shift-robust than one fit harder to long_view. |
| 8 | failed | – | – | – | – | – | – | (agent turn failed) |
| 9 | failed | – | – | – | – | – | 1 | Adding a DIN-style target-attention head over each user's past long-viewed videos on top of the FM (trained with the best-so-far listwise softmax loss) beats the FM+softmax best of 0.6023. |
| 10 | timeout | 0.6019 | 0.6682 | 0.5357 | +0.0003 | 0.3762 | 656 | Adding a DIN-style target-attention head over each user's past long-viewed videos on top of the FM (trained with the best-so-far listwise softmax loss) beats the FM+softmax best of 0.6023. |
| 11 | ok | 0.5994 | 0.6645 | 0.5343 | -0.0022 | 0.3715 | 360 | With the attention window shortened to L=8 and lr doubled so training actually converges inside the 600s budget, the DIN head beats FM+softmax (0.6023) rather than stopping under-trained at epoch 1. |
| 12 | timeout | 0.6035 | 0.6702 | 0.5368 | +0.0019 | 0.3705 | 608 | Heavy regularisation (embedding dropout p=0.3 plus l2 raised 100x to 1e-4) lets FM+softmax train for many epochs instead of peaking at epoch 1, lifting the peak above 0.6023. |
| 13 | timeout | 0.6018 | 0.6677 | 0.5358 | +0.0002 | 0.3675 | 620 | Pushing further along the regularisation axis that just worked (dropout 0.3->0.5, l2 1e-4->3e-4) while doubling the batch to fit more epochs in budget lifts validation above 0.6035. |
| 14 | timeout | 0.6029 | 0.6693 | 0.5365 | +0.0013 | 0.3735 | 686 | The DIN history head, trained under iteration 12's proven regularisation recipe so it survives past epoch 1, beats the regularised FM's 0.6035 and keeps DIN's superior unbiased score. |
| 15 | ok | 0.6035 | 0.6702 | 0.5368 | +0.0019 | 0.3705 | 532 | REDO iter12: FM+softmax, dropout p=0.3 l2=1e-4, fair per-seed budget. |
| 16 | ok | 0.6037 | 0.6704 | 0.5370 | +0.0021 | 0.3705 | 585 | REDO iter13: same but p=0.5 -- seed1 reached epoch 10 still improving when the clock died. |
| 17 | ok | 0.6036 | 0.6704 | 0.5368 | +0.0020 | 0.3723 | 929 | REDO iter14: DIN head under iteration 12's regularisation recipe. |
| 18 | ok | 0.6032 | 0.6694 | 0.5369 | +0.0016 | 0.3708 | 160 | [tune 1/14, init] BO over the regularisation family: regularisation is the only lever that has produced a real gain, so map that surface properly. |
| 19 | ok | 0.6011 | 0.6666 | 0.5357 | -0.0005 | 0.3633 | 893 | [tune 2/14, init] BO over the regularisation family: regularisation is the only lever that has produced a real gain, so map that surface properly. |
| 20 | ok | 0.6037 | 0.6703 | 0.5370 | +0.0021 | 0.3732 | 170 | [fm_softmax init] BO+ASHA trial, kept for the ensemble library |
| 21 | ok | 0.6026 | 0.6690 | 0.5363 | +0.0010 | 0.3678 | 552 | [fm_softmax init] BO+ASHA trial, kept for the ensemble library |
| 22 | ok | 0.6038 | 0.6703 | 0.5372 | +0.0022 | 0.3722 | 233 | [fm_softmax bayes] BO+ASHA trial, kept for the ensemble library |
| 23 | ok | 0.6037 | 0.6702 | 0.5371 | +0.0021 | 0.3722 | 183 | [fm_softmax bayes] BO+ASHA trial, kept for the ensemble library |
| 24 | ok | 0.6022 | 0.6685 | 0.5360 | +0.0006 | 0.3706 | 160 | [fm_bpr init] BO+ASHA trial, kept for the ensemble library |
| 25 | ok | 0.6029 | 0.6695 | 0.5364 | +0.0013 | 0.3689 | 122 | [fm_bpr init] BO+ASHA trial, kept for the ensemble library |
| 26 | ok | 0.6029 | 0.6695 | 0.5364 | +0.0013 | 0.3688 | 114 | [fm_bpr bayes] BO+ASHA trial, kept for the ensemble library |
| 27 | ok | 0.6031 | 0.6698 | 0.5364 | +0.0015 | 0.3713 | 103 | [fm_bpr bayes] BO+ASHA trial, kept for the ensemble library |
| 28 | ok | 0.5962 | 0.6600 | 0.5324 | -0.0054 | 0.3618 | 173 | [fm_lambda init] BO+ASHA trial, kept for the ensemble library |
| 29 | ok | 0.5997 | 0.6649 | 0.5345 | -0.0019 | 0.3640 | 419 | [fm_lambda init] BO+ASHA trial, kept for the ensemble library |
| 30 | ok | 0.5996 | 0.6648 | 0.5344 | -0.0020 | 0.3658 | 453 | [fm_lambda bayes] BO+ASHA trial, kept for the ensemble library |
| 31 | timeout | 0.6008 | 0.6664 | 0.5353 | -0.0008 | 0.3690 | 627 | [fm_lambda bayes] BO+ASHA trial, kept for the ensemble library |
| 32 | timeout | 0.6034 | 0.6698 | 0.5369 | +0.0018 | 0.3729 | 924 | [din_softmax init] BO+ASHA trial, kept for the ensemble library |
| 33 | timeout | 0.6018 | 0.6681 | 0.5355 | +0.0002 | 0.3739 | 895 | [din_softmax init] BO+ASHA trial, kept for the ensemble library |
| 34 | timeout | 0.6033 | 0.6698 | 0.5369 | +0.0017 | 0.3726 | 966 | [din_softmax bayes] BO+ASHA trial, kept for the ensemble library |
| 35 | timeout | 0.6035 | 0.6701 | 0.5368 | +0.0019 | 0.3703 | 837 | [din_softmax bayes] BO+ASHA trial, kept for the ensemble library |
| 36 | ok | 0.6029 | 0.6693 | 0.5366 | +0.0013 | 0.3727 | 132 | [fm_softmax init] BO+ASHA trial, kept for the ensemble library |
| 37 | ok | 0.6032 | 0.6697 | 0.5367 | +0.0016 | 0.3725 | 80 | [fm_softmax init] BO+ASHA trial, kept for the ensemble library |
| 38 | ok | 0.6030 | 0.6695 | 0.5365 | +0.0014 | 0.3702 | 204 | [fm_softmax init] BO+ASHA trial, kept for the ensemble library |
| 39 | ok | 0.6032 | 0.6697 | 0.5367 | +0.0016 | 0.3724 | 89 | [fm_softmax bayes] BO+ASHA trial, kept for the ensemble library |
| 40 | ok | 0.6030 | 0.6694 | 0.5367 | +0.0014 | 0.3724 | 93 | [fm_softmax bayes] BO+ASHA trial, kept for the ensemble library |
| 41 | ok | 0.6039 | 0.6703 | 0.5374 | +0.0023 | 0.3718 | 84 | [fm_bpr init] BO+ASHA trial, kept for the ensemble library |
| 42 | ok | 0.6015 | 0.6674 | 0.5356 | -0.0001 | 0.3717 | 52 | [fm_bpr init] BO+ASHA trial, kept for the ensemble library |
| 43 | ok | 0.6039 | 0.6704 | 0.5373 | +0.0023 | 0.3721 | 76 | [fm_bpr bayes] BO+ASHA trial, kept for the ensemble library |
| 44 | ok | 0.6039 | 0.6706 | 0.5373 | +0.0023 | 0.3720 | 79 | [fm_bpr bayes] BO+ASHA trial, kept for the ensemble library |
| 45 | ok | 0.5989 | 0.6633 | 0.5345 | -0.0027 | 0.3592 | 84 | [fm_bce init] BO+ASHA trial, kept for the ensemble library |
| 46 | ok | 0.6032 | 0.6694 | 0.5369 | +0.0016 | 0.3628 | 92 | [fm_bce init] BO+ASHA trial, kept for the ensemble library |
| 47 | ok | 0.6033 | 0.6697 | 0.5369 | +0.0017 | 0.3633 | 81 | [fm_bce bayes] BO+ASHA trial, kept for the ensemble library |
| 48 | ok | 0.6032 | 0.6695 | 0.5369 | +0.0016 | 0.3632 | 71 | [fm_bce bayes] BO+ASHA trial, kept for the ensemble library |
| 49 | ok | 0.6026 | 0.6689 | 0.5363 | +0.0010 | 0.3762 | 188 | [fm_lambda init] BO+ASHA trial, kept for the ensemble library |
| 50 | ok | 0.6017 | 0.6679 | 0.5355 | +0.0001 | 0.3767 | 172 | [fm_lambda init] BO+ASHA trial, kept for the ensemble library |
| 51 | ok | 0.6026 | 0.6690 | 0.5363 | +0.0010 | 0.3759 | 186 | [fm_lambda bayes] BO+ASHA trial, kept for the ensemble library |
| 52 | ok | 0.6027 | 0.6690 | 0.5364 | +0.0011 | 0.3760 | 186 | [fm_lambda bayes] BO+ASHA trial, kept for the ensemble library |
| 53 | ok | 0.6037 | 0.6705 | 0.5369 | +0.0021 | 0.3720 | 746 | [din_softmax init] BO+ASHA trial, kept for the ensemble library |
| 54 | ok | 0.6028 | 0.6693 | 0.5364 | +0.0012 | 0.3735 | 376 | [din_softmax init] BO+ASHA trial, kept for the ensemble library |
| 55 | ok | 0.6036 | 0.6704 | 0.5369 | +0.0020 | 0.3720 | 1449 | [din_softmax bayes] BO+ASHA trial, kept for the ensemble library |
| 56 | ok | 0.6037 | 0.6705 | 0.5369 | +0.0021 | 0.3721 | 745 | [din_softmax bayes] BO+ASHA trial, kept for the ensemble library |
| 57 | ensemble | – | – | – | – | – | – | Single-model search has converged at ~0.6038. The four objectives (softmax/BPR/BCE/LambdaRank) optimise different orderings of the same within-user impression list, and the DIN head uses a different input signal (user history) entirely, so their per-user rank errors should be substantially decorrelated. Greedy forward selection with replacement over a library spanning all five should beat the best single member by more than the residual single-model tuning gain. |
| 58 | failed | – | – | – | – | – | 7 | Injecting per-video behavioural priors (smoothed is_click/is_like/is_follow/is_comment/is_forward/is_hate rates, mean watch-time ratio, finish rate, log popularity, all computed on the train window only) as an item-side vector that is both globally weighted and crossed with the user embedding will beat the equivalent fm_dropout+softmax model, because the auxiliary engagement labels carry ordering information about a video that the sparse long_view label alone does not. |
| 59 | ok | 0.5988 | 0.6636 | 0.5339 | -0.0028 | 0.3772 | 124 | Per-video behavioural priors from the train window (smoothed engagement rates, mean watch-time ratio, finish rate, log popularity), entered globally and crossed with the user embedding, beat the equivalent fm_dropout+softmax model because the auxiliary labels carry ordering information the sparse long_view label does not. |
| 60 | ok | 0.6010 | 0.6667 | 0.5352 | -0.0006 | 0.3728 | 221 | [tune 1/14, init] The aux-prior term carries real signal (it produced the run's best unbiased score, 0.3772) but is trained far too aggressively -- every seed peaked at epoch 1 and decayed monotonically. Shrinking its initialisation, raising weight decay, lowering lr and increasing the empirical-Bayes pseudo-count should let the model reach a later, better validation peak; if no setting in this space clears 0.6038, the direction is validation-negative and I will stop building on it. |
| 61 | ok | 0.5987 | 0.6638 | 0.5337 | -0.0029 | 0.3761 | 121 | [tune 2/14, init] The aux-prior term carries real signal (it produced the run's best unbiased score, 0.3772) but is trained far too aggressively -- every seed peaked at epoch 1 and decayed monotonically. Shrinking its initialisation, raising weight decay, lowering lr and increasing the empirical-Bayes pseudo-count should let the model reach a later, better validation peak; if no setting in this space clears 0.6038, the direction is validation-negative and I will stop building on it. |
| 62 | ok | 0.6004 | 0.6665 | 0.5344 | -0.0012 | 0.3775 | 202 | [tune 3/14, init] The aux-prior term carries real signal (it produced the run's best unbiased score, 0.3772) but is trained far too aggressively -- every seed peaked at epoch 1 and decayed monotonically. Shrinking its initialisation, raising weight decay, lowering lr and increasing the empirical-Bayes pseudo-count should let the model reach a later, better validation peak; if no setting in this space clears 0.6038, the direction is validation-negative and I will stop building on it. |
| 63 | ok | 0.5988 | 0.6636 | 0.5339 | -0.0028 | 0.3801 | 129 | [tune 4/14, init] The aux-prior term carries real signal (it produced the run's best unbiased score, 0.3772) but is trained far too aggressively -- every seed peaked at epoch 1 and decayed monotonically. Shrinking its initialisation, raising weight decay, lowering lr and increasing the empirical-Bayes pseudo-count should let the model reach a later, better validation peak; if no setting in this space clears 0.6038, the direction is validation-negative and I will stop building on it. |
| 64 | ok | 0.6006 | 0.6664 | 0.5348 | -0.0010 | 0.3760 | 118 | [tune 5/14, init] The aux-prior term carries real signal (it produced the run's best unbiased score, 0.3772) but is trained far too aggressively -- every seed peaked at epoch 1 and decayed monotonically. Shrinking its initialisation, raising weight decay, lowering lr and increasing the empirical-Bayes pseudo-count should let the model reach a later, better validation peak; if no setting in this space clears 0.6038, the direction is validation-negative and I will stop building on it. |
| 65 | ok | 0.6010 | 0.6669 | 0.5351 | -0.0006 | 0.3757 | 208 | [tune 6/14, bayes] The aux-prior term carries real signal (it produced the run's best unbiased score, 0.3772) but is trained far too aggressively -- every seed peaked at epoch 1 and decayed monotonically. Shrinking its initialisation, raising weight decay, lowering lr and increasing the empirical-Bayes pseudo-count should let the model reach a later, better validation peak; if no setting in this space clears 0.6038, the direction is validation-negative and I will stop building on it. |
| 66 | ok | 0.6012 | 0.6671 | 0.5353 | -0.0004 | 0.3762 | 204 | [tune 7/14, bayes] The aux-prior term carries real signal (it produced the run's best unbiased score, 0.3772) but is trained far too aggressively -- every seed peaked at epoch 1 and decayed monotonically. Shrinking its initialisation, raising weight decay, lowering lr and increasing the empirical-Bayes pseudo-count should let the model reach a later, better validation peak; if no setting in this space clears 0.6038, the direction is validation-negative and I will stop building on it. |
| 67 | ok | 0.6012 | 0.6672 | 0.5353 | -0.0004 | 0.3761 | 207 | [tune 8/14, bayes] The aux-prior term carries real signal (it produced the run's best unbiased score, 0.3772) but is trained far too aggressively -- every seed peaked at epoch 1 and decayed monotonically. Shrinking its initialisation, raising weight decay, lowering lr and increasing the empirical-Bayes pseudo-count should let the model reach a later, better validation peak; if no setting in this space clears 0.6038, the direction is validation-negative and I will stop building on it. |
| 68 | ok | 0.6011 | 0.6671 | 0.5352 | -0.0005 | 0.3761 | 292 | [tune 9/14, bayes] The aux-prior term carries real signal (it produced the run's best unbiased score, 0.3772) but is trained far too aggressively -- every seed peaked at epoch 1 and decayed monotonically. Shrinking its initialisation, raising weight decay, lowering lr and increasing the empirical-Bayes pseudo-count should let the model reach a later, better validation peak; if no setting in this space clears 0.6038, the direction is validation-negative and I will stop building on it. |
| 69 | ok | 0.6011 | 0.6670 | 0.5352 | -0.0005 | 0.3762 | 410 | [tune 10/14, bayes] The aux-prior term carries real signal (it produced the run's best unbiased score, 0.3772) but is trained far too aggressively -- every seed peaked at epoch 1 and decayed monotonically. Shrinking its initialisation, raising weight decay, lowering lr and increasing the empirical-Bayes pseudo-count should let the model reach a later, better validation peak; if no setting in this space clears 0.6038, the direction is validation-negative and I will stop building on it. |
| 70 | ok | 0.6012 | 0.6671 | 0.5352 | -0.0004 | 0.3762 | 195 | [tune 11/14, bayes] The aux-prior term carries real signal (it produced the run's best unbiased score, 0.3772) but is trained far too aggressively -- every seed peaked at epoch 1 and decayed monotonically. Shrinking its initialisation, raising weight decay, lowering lr and increasing the empirical-Bayes pseudo-count should let the model reach a later, better validation peak; if no setting in this space clears 0.6038, the direction is validation-negative and I will stop building on it. |
| 71 | ok | 0.6011 | 0.6672 | 0.5351 | -0.0005 | 0.3739 | 229 | [tune 12/14, bayes] The aux-prior term carries real signal (it produced the run's best unbiased score, 0.3772) but is trained far too aggressively -- every seed peaked at epoch 1 and decayed monotonically. Shrinking its initialisation, raising weight decay, lowering lr and increasing the empirical-Bayes pseudo-count should let the model reach a later, better validation peak; if no setting in this space clears 0.6038, the direction is validation-negative and I will stop building on it. |
| 72 | ok | 0.6011 | 0.6672 | 0.5349 | -0.0005 | 0.3744 | 225 | [tune 13/14, bayes] The aux-prior term carries real signal (it produced the run's best unbiased score, 0.3772) but is trained far too aggressively -- every seed peaked at epoch 1 and decayed monotonically. Shrinking its initialisation, raising weight decay, lowering lr and increasing the empirical-Bayes pseudo-count should let the model reach a later, better validation peak; if no setting in this space clears 0.6038, the direction is validation-negative and I will stop building on it. |
| 73 | ok | 0.6016 | 0.6677 | 0.5354 | -0.0000 | 0.3766 | 207 | [tune 14/14, bayes] The aux-prior term carries real signal (it produced the run's best unbiased score, 0.3772) but is trained far too aggressively -- every seed peaked at epoch 1 and decayed monotonically. Shrinking its initialisation, raising weight decay, lowering lr and increasing the empirical-Bayes pseudo-count should let the model reach a later, better validation peak; if no setting in this space clears 0.6038, the direction is validation-negative and I will stop building on it. |
| 74 | note | – | – | – | – | – | – | NEGATIVE (single model), but informative. fm_aux = FM + per-video behavioural priors from the train window (smoothed is_click/is_like/is_follow/is_comment/is_forward/is_hate rates, mean watch-time ratio, finish rate, log popularity), entered both globally (g.a_v) and crossed with the user embedding (e_u^T P a_v). This is organizers' direction 4/5 -- a true joint multi-task head is NOT implementable because the batch sampler carries only field ids, never per-row aux labels, so the information can only be injected item-side.

Result: iter 59 valid 0.5988 (-0.0028 vs FM), every seed peaking at epoch 1 and decaying monotonically. A 14-trial BO over aux_scale/alpha/p/lr/l2 reached only 0.6016 -- exactly the FM baseline -- and drove lr to the lower bound (2e-4) with aux_scale shrinking toward 0, i.e. the search's preferred move is to turn the aux term OFF. Direction is validation-negative; do not re-test.

The genuinely interesting part: iter 59 scored unbiased 0.3772, the HIGHEST of the entire run (typical is 0.370-0.374, ensemble 0.3716). So the per-video engagement priors do carry ordering signal that generalises off-policy; what they cost is agreement with the logging policy that validation (and the hidden test, being a later window of the same logged stream) is scored against. Popularity/engagement priors are partly a model OF the logging policy, which is why the two signals move in opposite directions here. Kept as a decorrelated ensemble family rather than a single-model direction. |
| 75 | ok | 0.6029 | 0.6694 | 0.5364 | +0.0013 | 0.3726 | 93 | [fm_softmax init] BO+ASHA trial, kept for the ensemble library |
| 76 | ok | 0.6032 | 0.6696 | 0.5368 | +0.0016 | 0.3704 | 228 | [fm_softmax init] BO+ASHA trial, kept for the ensemble library |
| 77 | ok | 0.6032 | 0.6696 | 0.5368 | +0.0016 | 0.3705 | 218 | [fm_softmax bayes] BO+ASHA trial, kept for the ensemble library |
| 78 | ok | 0.6032 | 0.6696 | 0.5368 | +0.0016 | 0.3706 | 230 | [fm_softmax bayes] BO+ASHA trial, kept for the ensemble library |
| 79 | ok | 0.6038 | 0.6703 | 0.5374 | +0.0022 | 0.3716 | 85 | [fm_bpr init] BO+ASHA trial, kept for the ensemble library |
| 80 | ok | 0.6016 | 0.6678 | 0.5355 | +0.0000 | 0.3720 | 53 | [fm_bpr init] BO+ASHA trial, kept for the ensemble library |
| 81 | ok | 0.6038 | 0.6702 | 0.5374 | +0.0022 | 0.3717 | 85 | [fm_bpr bayes] BO+ASHA trial, kept for the ensemble library |
| 82 | ok | 0.6039 | 0.6704 | 0.5373 | +0.0023 | 0.3716 | 85 | [fm_bpr bayes] BO+ASHA trial, kept for the ensemble library |
| 83 | ok | 0.5988 | 0.6633 | 0.5343 | -0.0028 | 0.3585 | 94 | [fm_bce init] BO+ASHA trial, kept for the ensemble library |
| 84 | ok | 0.6034 | 0.6698 | 0.5371 | +0.0018 | 0.3628 | 79 | [fm_bce init] BO+ASHA trial, kept for the ensemble library |
| 85 | ok | 0.6035 | 0.6699 | 0.5370 | +0.0019 | 0.3627 | 79 | [fm_bce bayes] BO+ASHA trial, kept for the ensemble library |
| 86 | ok | 0.6035 | 0.6700 | 0.5370 | +0.0019 | 0.3628 | 61 | [fm_bce bayes] BO+ASHA trial, kept for the ensemble library |
| 87 | ok | 0.6038 | 0.6706 | 0.5370 | +0.0022 | 0.3718 | 596 | [din_softmax init] BO+ASHA trial, kept for the ensemble library |
| 88 | ok | 0.6030 | 0.6697 | 0.5364 | +0.0014 | 0.3739 | 265 | [din_softmax init] BO+ASHA trial, kept for the ensemble library |
| 89 | ok | 0.6038 | 0.6706 | 0.5370 | +0.0022 | 0.3718 | 534 | [din_softmax bayes] BO+ASHA trial, kept for the ensemble library |
| 90 | ok | 0.6038 | 0.6706 | 0.5370 | +0.0022 | 0.3718 | 534 | [din_softmax bayes] BO+ASHA trial, kept for the ensemble library |
| 91 | ok | 0.6005 | 0.6660 | 0.5350 | -0.0011 | 0.3747 | 139 | [fm_aux init] BO+ASHA trial, kept for the ensemble library |
| 92 | ok | 0.6010 | 0.6671 | 0.5349 | -0.0006 | 0.3755 | 185 | [fm_aux init] BO+ASHA trial, kept for the ensemble library |
| 93 | ok | 0.6010 | 0.6671 | 0.5349 | -0.0006 | 0.3755 | 170 | [fm_aux bayes] BO+ASHA trial, kept for the ensemble library |
| 94 | ok | 0.6010 | 0.6672 | 0.5349 | -0.0006 | 0.3754 | 170 | [fm_aux bayes] BO+ASHA trial, kept for the ensemble library |
| 95 | ensemble | – | – | – | – | – | – | Swapping fm_lambda (which earned exactly zero weight in the previous selection) for fm_aux should improve the portfolio: fm_aux is only baseline-level alone (0.6016) but it is the most decorrelated family available -- it is the only one that reads the auxiliary engagement labels and watch time, and it produced the run's best unbiased score (0.3772) while being validation-negative, which is the signature of an error pattern the other four families do not have. Caruana-style greedy selection should give it non-trivial weight and clear the previous ensemble's 0.6054. |
| 96 | final | – | – | – | – | – | – | FINAL SUBMISSION: ensemble |
| 97 | ok | 0.6023 | 0.6686 | 0.5360 | +0.0007 | 0.3755 | 127 | CrossNetV2 (DCN-V2, Wang et al. 2021, arXiv:2008.13535) as a zero-initialised explicit-cross head on the FM base trains stably and scores at or above FM under the iteration-22 regularisation recipe. |
| 98 | ok | 0.6020 | 0.6683 | 0.5358 | +0.0004 | 0.3764 | 469 | CIN (xDeepFM, Lian et al. KDD'18) as a zero-initialised vector-wise crossing head on the FM base trains stably and scores at or above FM under the iteration-22 recipe. |
| 99 | ok | 0.6022 | 0.6682 | 0.5363 | +0.0006 | 0.3736 | 220 | [tune 1/16, init] Under a wide joint search over regularisation (lr, p, l2, cross dropout) and CrossNetV2 structure (depth, low-rank r), DCN-V2 can move its peak off epoch 1 and match or beat the fm_softmax family's best single model (~0.6040). |
| 100 | ok | 0.6039 | 0.6709 | 0.5370 | +0.0023 | 0.3711 | 103 | [tune 2/16, init] Under a wide joint search over regularisation (lr, p, l2, cross dropout) and CrossNetV2 structure (depth, low-rank r), DCN-V2 can move its peak off epoch 1 and match or beat the fm_softmax family's best single model (~0.6040). |
| 101 | ok | 0.6032 | 0.6697 | 0.5367 | +0.0016 | 0.3712 | 82 | [tune 3/16, init] Under a wide joint search over regularisation (lr, p, l2, cross dropout) and CrossNetV2 structure (depth, low-rank r), DCN-V2 can move its peak off epoch 1 and match or beat the fm_softmax family's best single model (~0.6040). |
| 102 | ok | 0.6041 | 0.6710 | 0.5371 | +0.0025 | 0.3721 | 584 | [tune 4/16, init] Under a wide joint search over regularisation (lr, p, l2, cross dropout) and CrossNetV2 structure (depth, low-rank r), DCN-V2 can move its peak off epoch 1 and match or beat the fm_softmax family's best single model (~0.6040). |
| 103 | ok | 0.6034 | 0.6700 | 0.5367 | +0.0018 | 0.3711 | 259 | [tune 5/16, init] Under a wide joint search over regularisation (lr, p, l2, cross dropout) and CrossNetV2 structure (depth, low-rank r), DCN-V2 can move its peak off epoch 1 and match or beat the fm_softmax family's best single model (~0.6040). |
| 104 | ok | 0.6041 | 0.6713 | 0.5369 | +0.0025 | 0.3729 | 362 | [tune 6/16, init] Under a wide joint search over regularisation (lr, p, l2, cross dropout) and CrossNetV2 structure (depth, low-rank r), DCN-V2 can move its peak off epoch 1 and match or beat the fm_softmax family's best single model (~0.6040). |
| 105 | ok | 0.6043 | 0.6712 | 0.5374 | +0.0027 | 0.3725 | 510 | [tune 7/16, bayes] Under a wide joint search over regularisation (lr, p, l2, cross dropout) and CrossNetV2 structure (depth, low-rank r), DCN-V2 can move its peak off epoch 1 and match or beat the fm_softmax family's best single model (~0.6040). |
| 106 | timeout | 0.6029 | 0.6694 | 0.5364 | +0.0013 | 0.3740 | 378 | [tune 8/16, bayes] Under a wide joint search over regularisation (lr, p, l2, cross dropout) and CrossNetV2 structure (depth, low-rank r), DCN-V2 can move its peak off epoch 1 and match or beat the fm_softmax family's best single model (~0.6040). |
| 107 | timeout | 0.6032 | 0.6697 | 0.5367 | +0.0016 | 0.3736 | 386 | [tune 9/16, bayes] Under a wide joint search over regularisation (lr, p, l2, cross dropout) and CrossNetV2 structure (depth, low-rank r), DCN-V2 can move its peak off epoch 1 and match or beat the fm_softmax family's best single model (~0.6040). |
| 108 | timeout | 0.6024 | 0.6686 | 0.5362 | +0.0008 | 0.3750 | 166 | [tune 10/16, bayes] Under a wide joint search over regularisation (lr, p, l2, cross dropout) and CrossNetV2 structure (depth, low-rank r), DCN-V2 can move its peak off epoch 1 and match or beat the fm_softmax family's best single model (~0.6040). |
| 109 | ok | 0.6037 | 0.6698 | 0.5375 | +0.0021 | 0.3687 | 264 | [tune 11/16, bayes] Under a wide joint search over regularisation (lr, p, l2, cross dropout) and CrossNetV2 structure (depth, low-rank r), DCN-V2 can move its peak off epoch 1 and match or beat the fm_softmax family's best single model (~0.6040). |
| 110 | ok | 0.6032 | 0.6698 | 0.5365 | +0.0016 | 0.3763 | 485 | [tune 12/16, bayes] Under a wide joint search over regularisation (lr, p, l2, cross dropout) and CrossNetV2 structure (depth, low-rank r), DCN-V2 can move its peak off epoch 1 and match or beat the fm_softmax family's best single model (~0.6040). |
| 111 | ok | 0.6036 | 0.6703 | 0.5369 | +0.0020 | 0.3698 | 791 | [tune 13/16, bayes] Under a wide joint search over regularisation (lr, p, l2, cross dropout) and CrossNetV2 structure (depth, low-rank r), DCN-V2 can move its peak off epoch 1 and match or beat the fm_softmax family's best single model (~0.6040). |
| 112 | timeout | 0.6024 | 0.6683 | 0.5364 | +0.0008 | 0.3659 | 1385 | [tune 14/16, bayes] Under a wide joint search over regularisation (lr, p, l2, cross dropout) and CrossNetV2 structure (depth, low-rank r), DCN-V2 can move its peak off epoch 1 and match or beat the fm_softmax family's best single model (~0.6040). |
| 113 | ok | 0.6042 | 0.6711 | 0.5372 | +0.0026 | 0.3757 | 427 | [tune 15/16, bayes] Under a wide joint search over regularisation (lr, p, l2, cross dropout) and CrossNetV2 structure (depth, low-rank r), DCN-V2 can move its peak off epoch 1 and match or beat the fm_softmax family's best single model (~0.6040). |
| 114 | ok | 0.6042 | 0.6711 | 0.5373 | +0.0026 | 0.3762 | 467 | [tune 16/16, bayes] Under a wide joint search over regularisation (lr, p, l2, cross dropout) and CrossNetV2 structure (depth, low-rank r), DCN-V2 can move its peak off epoch 1 and match or beat the fm_softmax family's best single model (~0.6040). |
| 115 | ok | 0.6032 | 0.6697 | 0.5367 | +0.0016 | 0.3746 | 641 | [tune 1/12, init] CIN's vector-wise crossing, searched over the same regularisation axes plus its feature-map width/depth and starting from the low-lr high-dropout regime DCNv2's sweep discovered, matches or beats the fm_softmax family's best single model (~0.6040). |
| 116 | ok | 0.6036 | 0.6703 | 0.5369 | +0.0020 | 0.3739 | 688 | [tune 2/12, init] CIN's vector-wise crossing, searched over the same regularisation axes plus its feature-map width/depth and starting from the low-lr high-dropout regime DCNv2's sweep discovered, matches or beats the fm_softmax family's best single model (~0.6040). |
| 117 | ok | 0.6033 | 0.6699 | 0.5368 | +0.0017 | 0.3764 | 415 | [tune 3/12, init] CIN's vector-wise crossing, searched over the same regularisation axes plus its feature-map width/depth and starting from the low-lr high-dropout regime DCNv2's sweep discovered, matches or beats the fm_softmax family's best single model (~0.6040). |
| 118 | ok | 0.6038 | 0.6704 | 0.5373 | +0.0022 | 0.3753 | 764 | [tune 4/12, init] CIN's vector-wise crossing, searched over the same regularisation axes plus its feature-map width/depth and starting from the low-lr high-dropout regime DCNv2's sweep discovered, matches or beats the fm_softmax family's best single model (~0.6040). |
| 119 | timeout | 0.6040 | 0.6708 | 0.5372 | +0.0024 | 0.3745 | 1631 | [tune 5/12, bayes] CIN's vector-wise crossing, searched over the same regularisation axes plus its feature-map width/depth and starting from the low-lr high-dropout regime DCNv2's sweep discovered, matches or beats the fm_softmax family's best single model (~0.6040). |
| 120 | timeout | 0.6027 | 0.6691 | 0.5363 | +0.0011 | 0.3764 | 261 | [tune 6/12, bayes] CIN's vector-wise crossing, searched over the same regularisation axes plus its feature-map width/depth and starting from the low-lr high-dropout regime DCNv2's sweep discovered, matches or beats the fm_softmax family's best single model (~0.6040). |
| 121 | ok | 0.6033 | 0.6700 | 0.5365 | +0.0017 | 0.3707 | 321 | [tune 7/12, bayes] CIN's vector-wise crossing, searched over the same regularisation axes plus its feature-map width/depth and starting from the low-lr high-dropout regime DCNv2's sweep discovered, matches or beats the fm_softmax family's best single model (~0.6040). |
| 122 | timeout | 0.6035 | 0.6703 | 0.5368 | +0.0019 | 0.3760 | 1934 | [tune 8/12, bayes] CIN's vector-wise crossing, searched over the same regularisation axes plus its feature-map width/depth and starting from the low-lr high-dropout regime DCNv2's sweep discovered, matches or beats the fm_softmax family's best single model (~0.6040). |
| 123 | timeout | 0.6035 | 0.6702 | 0.5368 | +0.0019 | 0.3773 | 1929 | [tune 9/12, bayes] CIN's vector-wise crossing, searched over the same regularisation axes plus its feature-map width/depth and starting from the low-lr high-dropout regime DCNv2's sweep discovered, matches or beats the fm_softmax family's best single model (~0.6040). |
| 124 | timeout | 0.6035 | 0.6702 | 0.5367 | +0.0019 | 0.3750 | 1898 | [tune 10/12, bayes] CIN's vector-wise crossing, searched over the same regularisation axes plus its feature-map width/depth and starting from the low-lr high-dropout regime DCNv2's sweep discovered, matches or beats the fm_softmax family's best single model (~0.6040). |
| 125 | ok | 0.6038 | 0.6703 | 0.5373 | +0.0022 | 0.3752 | 219 | [tune 1/14, init] A finer duration discretisation (dur_buckets 20/50/200) beats the default 10 ONCE the regularisation is re-tuned for the extra parameters it adds; the earlier neutral/negative A/B was confounded because it held p and l2 at values tuned for the 10-bucket encoding. |
| 126 | ok | 0.6032 | 0.6697 | 0.5366 | +0.0016 | 0.3687 | 385 | [tune 2/14, init] A finer duration discretisation (dur_buckets 20/50/200) beats the default 10 ONCE the regularisation is re-tuned for the extra parameters it adds; the earlier neutral/negative A/B was confounded because it held p and l2 at values tuned for the 10-bucket encoding. |
| 127 | ok | 0.6024 | 0.6687 | 0.5360 | +0.0008 | 0.3769 | 169 | [tune 3/14, init] A finer duration discretisation (dur_buckets 20/50/200) beats the default 10 ONCE the regularisation is re-tuned for the extra parameters it adds; the earlier neutral/negative A/B was confounded because it held p and l2 at values tuned for the 10-bucket encoding. |
| 128 | ok | 0.6040 | 0.6710 | 0.5370 | +0.0024 | 0.3734 | 332 | [tune 4/14, init] A finer duration discretisation (dur_buckets 20/50/200) beats the default 10 ONCE the regularisation is re-tuned for the extra parameters it adds; the earlier neutral/negative A/B was confounded because it held p and l2 at values tuned for the 10-bucket encoding. |
| 129 | ok | 0.6036 | 0.6700 | 0.5373 | +0.0020 | 0.3737 | 207 | [tune 5/14, init] A finer duration discretisation (dur_buckets 20/50/200) beats the default 10 ONCE the regularisation is re-tuned for the extra parameters it adds; the earlier neutral/negative A/B was confounded because it held p and l2 at values tuned for the 10-bucket encoding. |
| 130 | timeout | 0.6003 | 0.6654 | 0.5351 | -0.0013 | 0.3665 | 598 | [tune 6/14, bayes] A finer duration discretisation (dur_buckets 20/50/200) beats the default 10 ONCE the regularisation is re-tuned for the extra parameters it adds; the earlier neutral/negative A/B was confounded because it held p and l2 at values tuned for the 10-bucket encoding. |
| 131 | ok | 0.6038 | 0.6705 | 0.5371 | +0.0022 | 0.3758 | 239 | [tune 7/14, bayes] A finer duration discretisation (dur_buckets 20/50/200) beats the default 10 ONCE the regularisation is re-tuned for the extra parameters it adds; the earlier neutral/negative A/B was confounded because it held p and l2 at values tuned for the 10-bucket encoding. |
| 132 | ok | 0.6036 | 0.6703 | 0.5370 | +0.0020 | 0.3784 | 240 | [tune 8/14, bayes] A finer duration discretisation (dur_buckets 20/50/200) beats the default 10 ONCE the regularisation is re-tuned for the extra parameters it adds; the earlier neutral/negative A/B was confounded because it held p and l2 at values tuned for the 10-bucket encoding. |
| 133 | ok | 0.6038 | 0.6707 | 0.5368 | +0.0022 | 0.3731 | 319 | [tune 9/14, bayes] A finer duration discretisation (dur_buckets 20/50/200) beats the default 10 ONCE the regularisation is re-tuned for the extra parameters it adds; the earlier neutral/negative A/B was confounded because it held p and l2 at values tuned for the 10-bucket encoding. |
| 134 | ok | 0.6041 | 0.6711 | 0.5371 | +0.0025 | 0.3750 | 333 | [tune 10/14, bayes] A finer duration discretisation (dur_buckets 20/50/200) beats the default 10 ONCE the regularisation is re-tuned for the extra parameters it adds; the earlier neutral/negative A/B was confounded because it held p and l2 at values tuned for the 10-bucket encoding. |
| 135 | ok | 0.6043 | 0.6713 | 0.5373 | +0.0027 | 0.3751 | 361 | [tune 11/14, bayes] A finer duration discretisation (dur_buckets 20/50/200) beats the default 10 ONCE the regularisation is re-tuned for the extra parameters it adds; the earlier neutral/negative A/B was confounded because it held p and l2 at values tuned for the 10-bucket encoding. |
| 136 | ok | 0.6044 | 0.6715 | 0.5373 | +0.0028 | 0.3763 | 352 | [tune 12/14, bayes] A finer duration discretisation (dur_buckets 20/50/200) beats the default 10 ONCE the regularisation is re-tuned for the extra parameters it adds; the earlier neutral/negative A/B was confounded because it held p and l2 at values tuned for the 10-bucket encoding. |
| 137 | ok | 0.6044 | 0.6715 | 0.5373 | +0.0028 | 0.3763 | 357 | [tune 13/14, bayes] A finer duration discretisation (dur_buckets 20/50/200) beats the default 10 ONCE the regularisation is re-tuned for the extra parameters it adds; the earlier neutral/negative A/B was confounded because it held p and l2 at values tuned for the 10-bucket encoding. |
| 138 | ok | 0.6044 | 0.6715 | 0.5373 | +0.0028 | 0.3763 | 347 | [tune 14/14, bayes] A finer duration discretisation (dur_buckets 20/50/200) beats the default 10 ONCE the regularisation is re-tuned for the extra parameters it adds; the earlier neutral/negative A/B was confounded because it held p and l2 at values tuned for the 10-bucket encoding. |
| 139 | ok | 0.6045 | 0.6716 | 0.5374 | +0.0029 | 0.3758 | 511 | The tuned 200-bucket duration encoding, at the high dropout the optimiser selected for it, beats the 10-bucket default on the seed ensemble rather than merely tying on the noisier per-seed mean. |
| 140 | ok | 0.6040 | 0.6711 | 0.5370 | +0.0024 | 0.3748 | 520 | Control for iteration 139: with lr, p and l2 held at the tuned values, reverting only dur_buckets 200 -> 10 should lose the duration resolution and score lower, isolating the encoding from the regularisation that was tuned alongside it. |

### Iteration 5 — timeout

**Error:** `None`

**Recovery:** hit the 420s budget and returned the best checkpoint so far; the number is valid but may be under-trained

### Iteration 8 — failed

**Error:** `UnicodeEncodeError: 'charmap' codec can't encode character '\u2212' in position 160: character maps to <undefined>`

**Recovery:** outer loop caught it and continued to the next turn

### Iteration 9 — failed

**Error:** `KeyError: "unknown model type 'din'; known: ['deepfm', 'fm', 'fm_dropout']"`

**Recovery:** choose a model from ['deepfm', 'fm', 'fm_dropout'], or add one to agent_kit/models.py

### Iteration 10 — timeout

**Error:** `None`

**Recovery:** hit the 600s budget and returned the best checkpoint so far; the number is valid but may be under-trained

### Iteration 12 — timeout

**Error:** `None`

**Recovery:** hit the 600s budget and returned the best checkpoint so far; the number is valid but may be under-trained

### Iteration 13 — timeout

**Error:** `None`

**Recovery:** hit the 600s budget and returned the best checkpoint so far; the number is valid but may be under-trained

### Iteration 14 — timeout

**Error:** `None`

**Recovery:** hit the 600s budget and returned the best checkpoint so far; the number is valid but may be under-trained

### Iteration 31 — timeout

**Error:** `None`

**Recovery:** (none recorded)

### Iteration 32 — timeout

**Error:** `None`

**Recovery:** (none recorded)

### Iteration 33 — timeout

**Error:** `None`

**Recovery:** (none recorded)

### Iteration 34 — timeout

**Error:** `None`

**Recovery:** (none recorded)

### Iteration 35 — timeout

**Error:** `None`

**Recovery:** (none recorded)

### Iteration 58 — failed

**Error:** `IndexError: index 33749 is out of bounds for dimension 0 with size 33749`

**Recovery:** inspect the traceback; fix the config or the module it points at

### Iteration 106 — timeout

**Error:** `None`

**Recovery:** at least one seed hit the 900s PER-SEED budget and returned its best checkpoint so far, so this number is an under-trained LOWER BOUND, not a verdict on the config. epochs_run=[5, 15] secs_per_epoch=[18.5, 19.0]. If the epoch curve was still improving, re-run with a larger time_budget_s before concluding anything.

### Iteration 107 — timeout

**Error:** `None`

**Recovery:** at least one seed hit the 900s PER-SEED budget and returned its best checkpoint so far, so this number is an under-trained LOWER BOUND, not a verdict on the config. epochs_run=[5, 15] secs_per_epoch=[19.0, 19.3]. If the epoch curve was still improving, re-run with a larger time_budget_s before concluding anything.

### Iteration 108 — timeout

**Error:** `None`

**Recovery:** at least one seed hit the 900s PER-SEED budget and returned its best checkpoint so far, so this number is an under-trained LOWER BOUND, not a verdict on the config. epochs_run=[5, 13] secs_per_epoch=[9.0, 9.2]. If the epoch curve was still improving, re-run with a larger time_budget_s before concluding anything.

### Iteration 112 — timeout

**Error:** `None`

**Recovery:** at least one seed hit the 900s PER-SEED budget and returned its best checkpoint so far, so this number is an under-trained LOWER BOUND, not a verdict on the config. epochs_run=[7, 4] secs_per_epoch=[48.8, 260.5]. If the epoch curve was still improving, re-run with a larger time_budget_s before concluding anything.

### Iteration 119 — timeout

**Error:** `None`

**Recovery:** at least one seed hit the 900s PER-SEED budget and returned its best checkpoint so far, so this number is an under-trained LOWER BOUND, not a verdict on the config. epochs_run=[14, 9] secs_per_epoch=[16.2, 155.9]. If the epoch curve was still improving, re-run with a larger time_budget_s before concluding anything.

### Iteration 120 — timeout

**Error:** `None`

**Recovery:** at least one seed hit the 900s PER-SEED budget and returned its best checkpoint so far, so this number is an under-trained LOWER BOUND, not a verdict on the config. epochs_run=[5, 14] secs_per_epoch=[13.0, 13.9]. If the epoch curve was still improving, re-run with a larger time_budget_s before concluding anything.

### Iteration 122 — timeout

**Error:** `None`

**Recovery:** at least one seed hit the 900s PER-SEED budget and returned its best checkpoint so far, so this number is an under-trained LOWER BOUND, not a verdict on the config. epochs_run=[10, 10] secs_per_epoch=[96.6, 96.4]. If the epoch curve was still improving, re-run with a larger time_budget_s before concluding anything.

### Iteration 123 — timeout

**Error:** `None`

**Recovery:** at least one seed hit the 900s PER-SEED budget and returned its best checkpoint so far, so this number is an under-trained LOWER BOUND, not a verdict on the config. epochs_run=[8, 6] secs_per_epoch=[126.4, 151.9]. If the epoch curve was still improving, re-run with a larger time_budget_s before concluding anything.

### Iteration 124 — timeout

**Error:** `None`

**Recovery:** at least one seed hit the 900s PER-SEED budget and returned its best checkpoint so far, so this number is an under-trained LOWER BOUND, not a verdict on the config. epochs_run=[6, 8] secs_per_epoch=[166.0, 112.1]. If the epoch curve was still improving, re-run with a larger time_budget_s before concluding anything.

### Iteration 130 — timeout

**Error:** `None`

**Recovery:** at least one seed hit the 900s PER-SEED budget and returned its best checkpoint so far, so this number is an under-trained LOWER BOUND, not a verdict on the config. epochs_run=[5, 30] secs_per_epoch=[17.3, 17.0]. If the epoch curve was still improving, re-run with a larger time_budget_s before concluding anything.
