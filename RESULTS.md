# Final Submission & Results Summary

**Benchmark:** KuaiRand-Pure (required). Bonus benchmarks KuaiRand-1k / 27k were **not attempted**.
**Label:** `long_view` · **Task:** within-user ranking over logged impressions · **Primary metric:** mean(GAUC, nDCG@5)

---

## 1. Final submission

| | |
|---|---|
| File | [`Tiktok/submission.csv`](Tiktok/submission.csv) |
| Schema | `row_id,user_id,video_id,score` — the Starter Kit schema |
| Rows | 170,588 (the full `test` split, `20220429–20220508`) |
| Format check | `python submit.py --check --split test submission.csv` → **PASSED** |
| Designated by | the agent itself, via the `finalize` tool, at journal iteration 96 |
| Record of the decision | [`Tiktok/runs/FINAL.json`](Tiktok/runs/FINAL.json) (includes the agent's reasoning, verbatim) |

> **A later run disagreed, and was overruled.** On 2026-09-01 a further exploration run reached the
> finalize step and proposed replacing the ensemble with the best single model (DCN-V2, validation
> 0.6050), reasoning: *"The ensemble edges validation (0.6056 vs 0.6050) but its unbiased-log score
> moved the opposite way (0.3713 vs 0.3720), so that last +0.0006 isn't corroborated."* That run was
> stopped by the operator before it could overwrite the submission, and the original ensemble was
> kept. This is recorded here because it is the one point in the project where a human overrode the
> agent's judgement about the final artifact — the agent's argument is a reasonable one, and the
> single-model alternative is reported in the table below for comparison.

**What the model is.** A greedy-selected 10-member ensemble spanning three structurally different
training objectives over the same within-user impression lists:

| Family | Members | Total weight | What it contributes |
|---|---|---|---|
| `fm_bpr` | 3 | 0.34 | pairwise BPR — within-user pairwise ranking |
| `fm_bce` | 4 | 0.28 | pointwise BCE — the baseline objective, retained for decorrelation |
| `din_softmax` | 3 | 0.38 | DIN-style target attention over each user's watch history, listwise softmax |

Member weights and per-member validation scores are in
[`Tiktok/runs/ensemble_search.json`](Tiktok/runs/ensemble_search.json) and in the `log_tail`
field of `FINAL.json`.

---

## 2. Results table

All figures are on the **validation** split (`20220422–20220428`). The hidden test split was never
read during development — see §4.

| Model | GAUC | nDCG@5 | **Primary** | **Δ vs FM baseline** |
|---|---|---|---|---|
| random (lower-bound self-check) | 0.4993 | 0.4675 | 0.4834 | −0.1182 |
| item popularity (trivial) | 0.6387 | 0.5227 | 0.5807 | −0.0209 |
| **FM — official baseline** | **0.6674** | **0.5357** | **0.6016** | — |
| Our best single model (iter 148, DCN-V2 + softmax) | 0.6721 | 0.5379 | 0.6050 | **+0.0034** |
| **Our final submission (10-member ensemble)** | **0.6729** | **0.5383** | **0.6056** | **+0.0040** |
| oracle ceiling (perfect ranking) | 1.0000 | 0.6968 | 0.8484 | +0.2468 |

**Per-metric delta of the final submission vs FM:** GAUC **+0.0055**, nDCG@5 **+0.0026**.

### Reading the delta honestly

- The FM baseline's standard deviation across 5 seeds is **0.0008**, so **+0.0040 is ≈5σ** — it is a
  real improvement, not a seed draw.
- Two independent `ensemble_search` runs reproduced it (0.6054, then 0.6056).
- It is nonetheless a **modest** improvement: it captures **1.6%** of the 0.2468 gap between the FM
  baseline and the oracle ceiling. We report it against the oracle rather than against 1.0 because
  27.1% of users are all-negative (nDCG = 0 for any model) and 9.2% are all-positive — the metric's
  true ceiling is 0.8484 on validation, not 1.0.
- **Unbiased cross-check.** On the randomised-exposure log (`log_random_4_22_to_5_08_pure.csv`,
  1.18M rows, never trained on), the FM control scores 0.3642 and the final ensemble scores
  **0.3713 (+0.0071)** — the improvement holds off-policy. Against the *best single models*
  (~0.372) the ensemble is flat, so the ensemble's edge over its own members is a
  variance-reduction effect rather than a demonstrated off-policy gain. The agent recorded this
  itself in `FINAL.json` rather than claiming more.

---

## 3. Resource usage

Machine-readable copy: [`Tiktok/runs/resources.json`](Tiktok/runs/resources.json).

| | |
|---|---|
| **LLM tokens (agent), input** | **10,887,397** |
| **LLM tokens (agent), output** | **272,944** |
| **LLM tokens (agent), total** | **11,160,341** |
| Model | `claude-opus-5` via the Claude Agent SDK |
| LLM cost | **$0 metered** — Claude Pro subscription transport, no API credits consumed |
| **Agent wall-clock** | **13.5 h** (union of 10 Agent SDK session spans) |
| Experiment compute | 15.2 h, single CPU core |
| **GPU-hours** | **0.0** — no GPU was used at any point |
| **Agent iterations** | **45** (36 of them produced the designated final submission) |
| Experiment runs recorded | 153 |
| **Manual interventions (in-loop)** | **0** |

Of the input tokens, **9,491,503 (87%) were prompt-cache reads** — the direct effect of holding the
scaffold and the journal digest byte-stable across turns, and of feeding the agent a compact digest
instead of re-dumping the full history each iteration.

### On the iteration count and the 50-iteration cap

We report two numbers because they measure different things, and only one of them is the
"iteration" the cap refers to:

- **45 agent iterations** — distinct hypothesis → experiment → evaluate → reflect cycles. This is
  the loop the cap governs. **36** of them had been spent when the agent designated its final
  submission; the remaining 9 were a post-final exploration phase (DCN-V2, CIN, duration encoding,
  explicit crosses) that did not displace the ensemble.
- **153 experiment runs** — rows in `runs/journal.jsonl`. One agent iteration often spends many
  training runs, because the `tune` tool dispatches a Bayesian-optimisation + ASHA batch of up to
  16 trials inside a *single* hypothesis, and every trial is journalled individually for evidence.

The run was spread over 8 operator-launched sessions, each configured under its own cap
(`--max-iterations` 6/50/90). No single session exceeded its cap.

### Note on token accounting

`run_agent.py` accumulates `ResultMessage.usage` per turn into `runs/resources.json`, but the
Claude Pro subscription transport does not populate that field, so the live counter recorded 0.
The figures above were **recovered after the fact** by summing the `usage` blocks of the Agent SDK
session transcripts in `~/.claude/projects/`, partitioned into the 10 agent-loop sessions (counted
above) and the 4 interactive development sessions in which the harness itself was written
(**170,450,304** further tokens, reported separately since they are not the agent's own
research loop).

---

## 4. Test-set integrity

The hidden test labels are physically present in the Starter Kit's CSVs, so "we didn't look" is a
claim that needs enforcing rather than asserting. Two mechanisms enforce it, and both are tested:

1. **An eval-access lock** (`agent_kit/dataset.py`). During model construction and training the
   validation/test frames are locked; any model that reaches for them raises `EvalAccessError` and
   the run fails loudly instead of posting a great score.
2. **A `PreToolUse` hook** (`run_agent.py`) that refuses the agent's writes to the files defining
   the metric and the split — `evaluate.py`, `data.py`, `baseline.py`, `submit.py`.

`python tests/test_leak_guard.py` builds a deliberately cheating model that reads
`frames['valid']['y']` and asserts that training it fails. **All checks pass** (0 blocked writes
were needed during the run — the agent never attempted one).

Every number in §2 is a validation number. The test split was scored exactly once, by
`submit_final.py`, to *write* the submission's scores — never to read its labels.
