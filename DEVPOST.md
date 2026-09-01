# An Autonomous ML Research Agent for KuaiRand-Pure

*TikTok TechJam 2026 — Challenge 2. Written project description (Devpost).*

---

## The problem, and how this addresses it

Challenge 2 asks for an agent that autonomously iterates the ML engineering loop on a recommender
benchmark — read the problem, explore the data, engineer features, train, tune, evaluate, reflect,
repeat — and pushes the score above the official Factorization Machine baseline on **KuaiRand-Pure**.
It is scored on two things at once: a real ML result, and the *agent* that produced it.

**The result.** The agent beat the baseline and designated its own final submission.

| | GAUC | nDCG@5 | Primary | Δ vs FM |
|---|---|---|---|---|
| FM — official baseline | 0.6674 | 0.5357 | 0.6016 | — |
| **Final submission (10-member ensemble)** | **0.6729** | **0.5383** | **0.6056** | **+0.0040** |

Validation split; the hidden test split was never read during development. +0.0040 is ≈5σ against
the baseline's 0.0008 seed noise, and two independent selection runs reproduced it (0.6054, 0.6056).

**The agent.** 45 iterations, 153 journalled experiments, **0 manual interventions**, 13.5 h
wall-clock, 11.2M LLM tokens, **0 GPU-hours** — the entire run was a single CPU core.

Four design decisions do most of the work:

**1. The agent's edit surface is a config, not a codebase.** Every experiment is
`train_once(config, prepared, seed)` over four pluggable slots — `features`, `model`, `loss`,
`train`. The agent proposes a config; the harness trains it and returns structured metrics. A bad
idea fails in seconds instead of corrupting a pipeline, and every iteration's diff is a small
readable object rather than a patch that has to be reviewed. When the agent genuinely needs a new
function class it writes a registered `nn.Module` into `agent_kit/models.py` — it added DIN
target-attention, DCN-V2 and CIN heads this way, after fetching the papers' actual formulations
with web search rather than reciting them from memory.

**2. The research journal is memory, reflection, and evidence in one file.** Every experiment —
successful, failed, or abandoned — appends `{hypothesis, rationale, config diff, metrics, error,
recovery}` to an append-only `journal.jsonl`. The agent reads back a compact *digest* rather than
its full history, which is why 87% of its input tokens are prompt-cache reads and the whole run
cost 11M tokens instead of ~100M. The same file renders directly to the required per-iteration
run log.

**3. Test-label integrity is enforced, not promised.** The hidden test labels ship inside the
Starter Kit's own CSVs. An agent that writes its own model code could read them and post a
wonderful, worthless score, and nothing downstream would notice. Two mechanisms prevent it: an
**eval-access lock** that blocks the validation/test frames during model construction and training,
and a **`PreToolUse` hook** that refuses the agent's writes to the files defining the metric and the
split. A unit test builds a deliberately cheating model and asserts that training it raises rather
than scores.

**4. Robustness is designed for, because a 6-hour run will hit errors.** Every experiment runs under
a timeout and a try/except; a failure returns the traceback *plus a suggested recovery* to the agent
as a normal tool result, so the loop routes around it instead of crashing. Across the run, 20
timeouts and 3 hard errors (a `UnicodeEncodeError`, an unknown model type, an index-out-of-bounds)
were all recovered by the agent with no human involvement. Timeouts are explicitly labelled
"under-trained lower bound, not a verdict on this config" so the agent does not draw a false
negative from a starved run — it re-ran several with a bigger budget, and one of those became the
regularisation finding that produced most of the run's gains.

**What it found.** The Starter Kit's top-ranked suggestion was to swap the pointwise loss for a
ranking loss. The agent tried that first — BCE 0.6017, BPR 0.6021, softmax 0.6023, LambdaRank
0.6017 — and concluded the axis was inside the noise band and closed it. The real lever turned out
to be **regularisation**: embedding dropout plus 100× l2 let the model train past epoch 1 instead of
peaking there, worth more than every loss change combined. It then diagnosed its own DeepFM
negative as one-epoch memorisation rather than "deep models don't work", which is why re-opening
the family later under proper regularisation produced the best single model of the run (DCN-V2,
0.6050). The final submission is a portfolio: four objectives plus a history-based DIN head make
*decorrelated* per-user rank errors, and averaging them beat every individual member.

It also declined a tempting result: `fm_aux` (per-video engagement priors) produced the run's best
*unbiased* score but was validation-negative, so the agent recorded it as a finding and excluded it
from the submission rather than optimise a metric the task is not scored on.

**Full detail:** [`README.md`](README.md) · [`RESULTS.md`](RESULTS.md) ·
[`Tiktok/runs/RUN_LOG.md`](Tiktok/runs/RUN_LOG.md) · [`Tiktok/runs/FINAL.json`](Tiktok/runs/FINAL.json)

---

## Development tools used

| Tool | Use |
|---|---|
| **Claude Code** (VS Code extension + CLI) | Primary development environment for building the harness and the agent loop |
| **Visual Studio Code** | Editor |
| **Windows PowerShell** | Deployment (`run.ps1`), detached process management, live monitoring |
| **Python 3.12.5** | Everything |
| **Git / GitHub** | Version control |
| Custom terminal dashboard (`dashboard.py`) | Live view of a running agent — reads `runs/status.json` as a separate process, so closing it can never kill a 6-hour run |

Platform: Windows 11, single CPU core, no GPU.

## APIs used

| API | Use |
|---|---|
| **Claude API — `claude-opus-5`**, via the **Claude Agent SDK** (`claude-agent-sdk` 0.2.148) | The agent's reasoning: hypothesis formation, code authoring, result interpretation, and the decision of what to try next. Run on the Claude Code CLI transport against a Claude Pro subscription, so no metered API spend. |
| **Claude Agent SDK — in-process MCP server** (`create_sdk_mcp_server`) | The agent's tools are exposed as an MCP server defined in the same process: `run_experiment`, `tune`, `ensemble_search`, `read_journal`, `get_search_space`, `record_finding`, `finalize`. |
| **Claude Agent SDK — `PreToolUse` hooks** (`HookMatcher`) | The leak guard: refuses agent writes to `evaluate.py`, `data.py`, `baseline.py`, `submit.py`. |
| **WebSearch / WebFetch** (Claude Code built-in tools) | The agent fetched the actual formulations of DCN-V2 (Wang et al., 2021) and xDeepFM/CIN (Lian et al., KDD'18) before implementing them, rather than reconstructing them from memory. |

No other external APIs. No external services, no hosted inference, no third-party data.

## Libraries and frameworks used

| Library | Version | Use |
|---|---|---|
| **PyTorch** | 2.13.0 (CPU) | All models: FM, FM+dropout, DeepFM, DIN target attention, DCN-V2 (CrossNetV2), CIN, `fm_aux` — and all four losses (BCE, BPR, listwise softmax, LambdaRank) |
| **NumPy** | 2.3.2 | Feature encoding, the fast GAUC/nDCG@5 implementation, rank averaging |
| **claude-agent-sdk** | 0.2.148 | The autonomous loop, its MCP tool server, and the hook system |
| **pandas** | ≥2.0 | Used by the Starter Kit's own data loader |

Deliberately **no** RecBole, LightGBM, Optuna, Ray Tune, or scikit-learn. Bayesian optimisation
with ASHA early stopping, greedy ensemble selection with replacement, and the metric implementation
are all written directly against NumPy — roughly 700 lines in `agent_kit/` — because the search
space is small, the dependency surface stays trivially reproducible, and the tuner needed to share
the same timeout-and-recover semantics as the rest of the harness.

One notable piece of engineering: `agent_kit/metrics.py` reimplements GAUC and nDCG@5 to run **4×
faster** than the official `evaluate.py`, and `tests/test_metrics.py` asserts agreement to **1e-14**
across adversarial cases (heavy ties, all-identical scores, all-positive and all-negative users,
singleton users, oracle and adversarial orderings). The speedup is what made 153 experiments fit in
the budget; the equivalence test is what makes it safe to rely on. The official `evaluate.py`
remains the scoring authority and is unmodified.

## Datasets and assets used

| Asset | Source | Use |
|---|---|---|
| **KuaiRand-Pure** | [Zenodo record 10439422](https://zenodo.org/records/10439422) (official, no registration) | The required benchmark. Official date-based split: train `20220408–20220421` (1,141,112 rows) / valid `20220422–20220428` (124,909) / test `20220429–20220508` (170,588). Label `long_view`. |
| `log_random_4_22_to_5_08_pure.csv` | Within KuaiRand-Pure | **Unbiased validation.** 1.18M randomised-exposure rows, never trained on, used as a secondary off-policy check that a gain was not just fitting the logging policy. This is what caught `fm_aux` as interesting-but-not-submittable. |
| Auxiliary log columns | Within KuaiRand-Pure | `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, `play_time_ms` — used to build smoothed per-video engagement priors for the `fm_aux` model family. |
| User watch history | Derived from the train split | Reconstructed per-user sequences of previously long-viewed videos, for the DIN target-attention head. Built from train rows only. |
| Starter Kit | Organisers | `evaluate.py`, `data.py`, `baseline.py`, `submit.py`, `ablation_features.py`, `baseline_scores.json` — used unmodified; `evaluate.py` is the scoring authority. |

**No external training data, no pretrained weights, no data outside the benchmark.** KuaiRand-1k
and KuaiRand-27k (the bonus benchmarks) were not attempted.
