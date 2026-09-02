# An Autonomous ML Research Agent for KuaiRand-Pure

**TikTok TechJam 2026 — Challenge 2: Autonomous ML Research Agent**

An agent that runs the whole ML engineering loop on the KuaiRand-Pure recommender benchmark by
itself — form a hypothesis, write the model code, train it, evaluate it, decide what the number
means, and choose what to try next — and drives the score past the official Factorization Machine
baseline.

| | Validation primary | Δ vs FM baseline |
|---|---|---|
| FM (official baseline) | 0.6016 | — |
| **Final submission (10-member ensemble)** | **0.6056** | **+0.0040** (≈5σ) |

**0 manual interventions.** 45 agent iterations, 153 journalled experiments, 13.5 h wall-clock,
11.2M LLM tokens, **0 GPU-hours**. Full accounting in **[RESULTS.md](RESULTS.md)**.

---

## Overview

The problem the challenge poses is not really "beat FM on KuaiRand" — it is "build something that
can beat FM on KuaiRand *without you*." So the deliverable here is a loop, and the score is its
output.

Three design decisions shaped it.

**1. The agent edits a config, not a codebase.** A pluggable harness splits every experiment into
four slots — `features`, `model`, `loss`, `train` — behind one function,
`train_once(config, prepared, seed)`. The agent proposes a config; the harness trains it and hands
back structured metrics. This makes the search space explicit and legible, makes every iteration's
"diff" a small readable object, and means a malformed idea fails in seconds instead of corrupting
the pipeline. When the agent genuinely needs a new function class (DIN attention, DCN-V2, CIN) it
writes a new registered `nn.Module` into `agent_kit/models.py` — the one place it writes real code.

**2. The journal is the memory, the reflect step, and the deliverable — all at once.** Every
experiment appends `{hypothesis, rationale, config diff, metrics, error, recovery}` to
`runs/journal.jsonl`. The agent reads back a compact *digest* rather than its full history, which
is why 87% of its input tokens are prompt-cache reads and the run costs 11M tokens rather than
100M. The same file renders to [`runs/RUN_LOG.md`](Tiktok/runs/RUN_LOG.md), the per-iteration
evidence log.

**3. Not looking at the test labels is enforced, not promised.** The hidden test labels ship inside
the Starter Kit's CSVs, so an agent writing its own model code could trivially read them and post a
wonderful, worthless score. An **eval-access lock** blocks the validation/test frames during model
construction and training, and a `PreToolUse` **hook** refuses the agent's writes to the files that
define the metric and the split. `tests/test_leak_guard.py` builds a deliberately cheating model
and asserts that it fails loudly.

### What the agent actually found

The Starter Kit ships two known dead ends (more features, more capacity) and a ranked list of
guesses. The agent confirmed the dead ends without re-testing them, then produced its own findings:

- **The loss function was not the bottleneck.** The kit's top-ranked suggestion — swap pointwise
  logloss for a ranking loss — was tried first (iterations 2–4). BCE 0.6017 → BPR 0.6021 → softmax
  0.6023 → LambdaRank 0.6017. A 0.0006 spread, inside the noise band. The agent recorded the axis
  as closed rather than tuning it further.
- **Regularisation was.** Iteration 12 found that embedding dropout plus 100× l2 let FM train past
  epoch 1 instead of peaking there — worth +0.0019, more than every loss change combined. Nearly
  every later gain came from this axis.
- **Deep crossing overfits in one epoch, and the first negative was wrong.** DeepFM lost by 0.0040,
  but the agent diagnosed it (`best_epoch=1` on all seeds) rather than concluding "MLPs don't work"
  — and correctly refused to re-tune it. When later re-opened under proper regularisation, DCN-V2
  became the best *single* model of the run (0.6050).
- **The winning move was decorrelation, not a better model.** Four objectives and a history-based
  DIN head make different per-user rank errors; the greedy-selected portfolio beats its best member.
- **One honest negative it declined to chase.** `fm_aux` (per-video engagement priors) produced the
  run's best *unbiased* score (0.3772) but was validation-negative. The agent recorded it as a
  finding and excluded it — chasing it would have meant optimising a metric the task is not scored
  on.

---

## Repository layout

```
├── README.md               ← you are here          ├── RESULTS.md   ← results table + resources
├── DEVPOST.md              ← project description   └── requirements.txt
└── Tiktok/
    ├── run_agent.py            the autonomous loop: MCP tool server, prompts, leak-guard hook
    ├── run.ps1 / DEPLOY.md     one-command deploy + live dashboard
    ├── dashboard.py            terminal dashboard (reads runs/status.json; never shares a process)
    ├── agent_kit/
    │   ├── harness.py          train_once(config, prepared, seed) — the whole experiment surface
    │   ├── dataset.py          encoding, official split, and the eval-access lock
    │   ├── models.py           fm · fm_dropout · deepfm · din · din_dropout · fm_aux · dcnv2 · cin
    │   ├── losses.py           bce · bpr · softmax · lambdarank
    │   ├── metrics.py          GAUC / nDCG@5, matched to evaluate.py to 1e-14, 4× faster
    │   ├── journal.py          the research journal + RUN_LOG.md renderer
    │   ├── tuner.py/search.py  Bayesian optimisation + ASHA early stopping
    │   └── ensemble_search.py  greedy portfolio selection with replacement
    ├── runs/
    │   ├── journal.jsonl       153 experiments, append-only
    │   ├── RUN_LOG.md          ← DELIVERABLE 3: per-iteration hypothesis / diff / metrics / recovery
    │   ├── FINAL.json          the submission the agent designated, and why (its own words)
    │   └── resources.json      tokens, wall-clock, iterations, interventions
    ├── submission.csv          ← DELIVERABLE 4: the final KuaiRand-Pure output
    ├── tests/                  metric-equivalence and leak-guard tests
    └── evaluate.py · data.py · baseline.py · submit.py · README.md   (Starter Kit, unmodified)
```

`Tiktok/README.md` is the **original Starter Kit README** (Chinese), left untouched as the
authoritative statement of the task conventions. `Tiktok/PROJECT_BRIEF.md` is the handoff brief the
project was built from.

---

## Setup

Requires **Python 3.9+**. No GPU. The full run was done on a single CPU core.

```bash
git clone https://github.com/DubuMizu/ML-Agent-For-Recommender-Systems.git
cd ML-Agent-For-Recommender-Systems
pip install -r requirements.txt
```

Get the dataset (the repo currently vendors it; if you cloned without it):

```bash
cd Tiktok
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz          # -> ./KuaiRand-Pure/
```

`claude-agent-sdk` is only needed to **run the agent**. Reproducing the submission needs only
`torch` and `numpy`.

---

## Reproducing the results

All commands run from `Tiktok/`. The first run spends ~2 min encoding the dataset into `.cache/`
(deterministic; reused afterwards).

**1. Verify the harness against the official baseline.**

```bash
python baseline.py --model fm         # official FM        -> primary ~ 0.5946 (test)
python baseline.py --model random     # harness self-check -> primary ~ 0.4753
```

**2. Verify the guarantees the results depend on.**

```bash
python tests/test_metrics.py          # our fast GAUC/nDCG == evaluate.py to 1e-14
python tests/test_leak_guard.py       # a cheating model is refused, not rewarded
```

Both print `ALL PASS`.

**3. Regenerate the final submission** from the ensemble the agent designated (~20 min, CPU):

```bash
python submit_final.py --ensemble-json runs/ensemble_search.json --split test --out submission.csv
python submit.py --check --split test submission.csv        # format + alignment
```

This retrains all 10 members from scratch and rank-averages them. Expected:
`validation GAUC 0.6729 | nDCG@5 0.5383 | primary 0.6056`.

**4. Re-run a single journalled iteration** — every row in `RUN_LOG.md` is replayable:

```bash
python submit_final.py --list                 # show journal candidates
python submit_final.py --iteration 148 --score-valid
```

**5. Run the agent itself** (Windows PowerShell; needs a Claude subscription or API key):

```powershell
cd Tiktok
.\run.ps1                              # 50 iterations, 6h cap, 3 seeds, + live dashboard
.\run.ps1 -Fresh                       # archive the journal and start the count at zero
.\run.ps1 -Status                      # one-shot snapshot
```

On Linux/macOS, invoke the loop directly (`run.ps1` is a convenience wrapper):

```bash
python run_agent.py --max-iterations 50 --hours 6 --seeds 3 --model claude-opus-5
```

See [`Tiktok/DEPLOY.md`](Tiktok/DEPLOY.md). The agent is deliberately restart-safe: it reads
`runs/journal.jsonl` on boot and resumes with its full history, so stopping it is never destructive.

---

## Limitations, and what I would do with more time

**The honest size of the result.** +0.0040 is ≈5σ against the FM seed noise and it replicates, but
it captures only **1.6%** of the FM→oracle gap. The agent moved the number reliably; it did not
crack the benchmark. Roughly a third of that gain is ensembling — a technique, not an insight.

**The ensemble's edge does not clearly replicate off-policy.** Against the FM control the ensemble
is up on the unbiased randomised-exposure log (+0.0071), but against its own best *members* it is
flat. The portfolio gain is variance reduction, which should transfer to a later time window, but
we cannot demonstrate that it does. A held-out *later* validation window would have tested the
thing that actually matters — the test split is a later period, and we never simulated that shift.

**The behaviour-sequence direction was under-explored.** DIN target attention earned the largest
single share of the final ensemble weight (0.38), which suggests user history is the most
undervalued signal here — and it was tried with only an 8–32 event window, no position or time
encoding, and no SIM-style long-sequence retrieval. This is where I would spend the next 50
iterations.

**Wall-clock, not tokens, was the binding constraint.** Nearly all of the 13.5 h was CPU training;
the LLM cost 11M tokens. Twenty of 153 experiments hit their time budget and returned under-trained
lower bounds. The agent handled these correctly (it labels them "not a verdict on the config"), but
a faster inner loop — mixed precision, a smaller candidate sample for intermediate evaluation —
would have bought substantially more search, which is the cheapest available improvement.

**Multi-task heads were never properly tried.** `is_click` / `is_like` / `play_time_ms` entered only
as static per-video priors (`fm_aux`), not as auxiliary training objectives. The kit ranks this #4
and the agent's own reflection at iteration 7 identified it as the best next move — then the
regularisation thread out-competed it for attention. It remains open.

**Bonus benchmarks were not attempted.** KuaiRand-1k and KuaiRand-27k were out of scope for the
time available; only the required KuaiRand-Pure benchmark was run.

**Autonomy has an asterisk.** In-loop interventions were 0 — no experiment was ever hand-fixed. But
I launched 8 sessions and supplied **2 opening directives** that set a research priority for a run's
first few turns (`directive_crossnets.md`, `directive_encoding.md`). They supply no code and no
configs and expire after N turns, but a fully hands-off run would not have had them. Both are in
the repo; judge them for yourself. I also **overrode the agent once on the final choice**: a
later run proposed swapping the ensemble for the best single model on the grounds that the
ensemble's validation edge was not corroborated off-policy, and I stopped it and kept the
ensemble. See [RESULTS.md](RESULTS.md) §1.

---

## Team

Participants — **DubuMizu**, **dkjw75**, **maxwellguico**, **smellywesley**, **shaunpann**. The ML results
were produced by the agent described above. The Starter Kit files (`evaluate.py`, `data.py`,
`baseline.py`, `submit.py`, `ablation_features.py`) are the organisers' and are unmodified.
