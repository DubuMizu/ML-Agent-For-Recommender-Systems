# TikTok TechJam 2026 — Challenge 2: Autonomous ML Research Agent

> **Purpose of this file:** self-contained handoff brief. A fresh Claude Code instance
> (or teammate) should be able to read only this file + the starter kit and continue.
> If you want it auto-loaded by Claude Code, rename this file to `CLAUDE.md`.

---

## 0. Goal in one sentence

Build an **autonomous ML research agent** that, on its own, iterates the ML engineering
loop (read problem → EDA → engineer features → train+tune → evaluate → reflect → repeat)
on the **KuaiRand-Pure** recommender benchmark, and drives the score **above the official
Factorization Machine (FM) baseline**.

You are scored on **two things at once**: (a) a real ML result (beat the FM baseline), and
(b) the *agent* that produced it (autonomy, innovation, robustness, efficiency).

---

## 1. The task (from the official brief)

For the required benchmark **KuaiRand-Pure**, the agent must autonomously:

1. **Reproduce the official baseline** (stand up a working end-to-end pipeline).
2. **Iterate on the pipeline** using established methods from industry/academia, applied in
   code. Develop on **train + validation only** — never touch the hidden test set.
3. **Improve over the baseline.** Improvement need not be monotonic; show a clear, sustained
   ability to keep improving. Final ranking = one-shot score on the hidden test using the
   submission the agent designates as final.

**Key requirements:**
- Runs end-to-end on KuaiRand-Pure and converges. Bonus benchmarks (KuaiRand-1k / 27k) optional.
- Iterates **autonomously across the full stack** (not just model architecture). Autonomy is
  measured by **number of manual interventions** — fewer is better, 0 is ideal. A
  well-instrumented semi-automated pipeline with a handful of interventions is acceptable.
- **Robust:** when a step fails (code error, timeout, bad input) the agent recovers/retries/
  routes around it; long runs must not crash, stall, or diverge.

---

## 2. Constraints & scope

- **In scope:** any open-source lib (PyTorch, RecBole, LightGBM, TorchRec…), any papers/public
  solutions/pretrained weights, changes to **any** pipeline stage.
- **Out of scope:** no external training data; no pretrained weights trained on these
  benchmarks' test labels; no hidden-test access during dev.
- **Compute budget:** **50 iterations/run** hard cap; convergence rule (ε=0.002, N=3) usually
  triggers first; 6h wall-clock backstop. Compute is deliberately NOT the binding constraint —
  ~100 baseline iterations take ~28 min on a single CPU core, no GPU needed.
- GPU-hours and LLM tokens are **reported** (for Feasibility scoring), not capped.

---

## 3. Ground truth from the starter kit (VERIFIED by reading the code)

**Metric (from `evaluate.py`, `baseline_scores.json` — authoritative):**
- Label = **`long_view`** (0/1 native column).
- Task = **within-user ranking** over each user's logged impressions (NOT full-catalog retrieval).
- Metrics = **GAUC** and **nDCG@5**. **Primary = mean(GAUC, nDCG@5).**
- ⚠️ The brief's "Limits" row saying *NDCG@10 / Recall@50, click=positive* is **STALE/WRONG**.
  Trust `evaluate.py`. It takes only `(user_ids, labels, scores)` and is model-agnostic.

**Baseline ladder (hidden test / "test" split):**

| Model | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (self-check lower bound) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline — BEAT THIS)** | **0.6610** | **0.5282** | **0.5946** |
| oracle ceiling (perfect ranking) | 1.0000 | 0.7289 | **0.8645** |

- FM config: `k=16, lr=0.001, batch=8192, max_epochs=40, patience=4`, 5 fields:
  `[user_id, video_id, author_id, tab, dur_bucket]`. FM std over 5 seeds = 0.0008.
- **Headroom is ~0.27, not ~0.41.** nDCG ceiling is 0.7289 (not 1.0) because 27.1% of test
  users are all-negative (nDCG=0 for any model) and 9.2% are all-positive (nDCG=1). Only 63.7%
  of users are discriminative / count toward GAUC. **Judge progress against 0.8645, not 1.0.**
- Convergence: ε=0.002 (≈2.5σ), N=3 → converged when validation primary hasn't improved by
  more than ε over the last 3 consecutive iterations.

**Data splits (date-based, fixed):**
- train `20220408–20220421` (1,141,112 rows)
- valid `20220422–20220428` (124,909 rows)
- test  `20220429–20220508` (170,588 rows) ← hidden test
- Dev on train+valid only.

**Submission format (`submit.py`):** CSV header `row_id,user_id,video_id,score`, one row per
eval-split row. `row_id` is 0-based strictly-increasing index into `data.load()[split]` order.
`(user_id, video_id)` is NOT unique (3.06% repeats, up to 12×) — that's why `row_id` is required.
`score` = any real number (only relative order matters); NaN/Inf rejected.
- `python3 submit.py --make` generates an example (FM baseline).
- `python3 submit.py --check` validates header/alignment/row_id gaps/numeric.
- `python3 submit.py --score --split valid` scores locally on validation.

**Dataset download (from README):**
```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz   # → ./KuaiRand-Pure/
```

---

## 4. Where the headroom actually is (⚠️ read this before choosing what to try)

The organizers ALREADY TESTED the two obvious ideas and they **do NOT help** — don't waste
iterations on them (see `ablation_features.py` + README):

- ❌ **Adding static features** (all 13 CWM feature fields): primary 0.5940 vs 0.5950 — noise,
  slightly worse. `user_id × video_id` cross already saturates the learnable signal.
- ❌ **More model capacity** (k = 8/16/32): no change. 1.14M rows can't support more capacity.
- ❗ **Structural fact:** scoring is *within-user*, so any term constant within a user cancels
  out of the ordering. **Pure user-side first-order features contribute EXACTLY 0.** User-side
  features only help via **crosses with item-side features.**

**Unexplored directions the organizers believe hold the headroom (ranked, their guess):**

| Priority | Direction | Effort | Why |
|---|---|---|---|
| **1** | **Within-user pairwise (BPR) / listwise softmax loss** | Low | FM uses pointwise logloss, but GAUC/nDCG are *ranking* metrics. Align the objective. **Organizers' top pick + our money move.** |
| 2 | **LambdaRank / LambdaLoss** (nDCG-weighted pairs) | Low–Med | Directly optimizes nDCG@5. |
| 3 | **User behavior sequences** (DIN / SIM target attention) | Med–High | Behavior sequences are completely unused; each user has 100s–1000s of train interactions. |
| 4 | **Multi-task** (aux heads: click/like/follow/comment/forward/play_time_ms) | Med | Regularizes the sparse `long_view` target. |
| 5 | **Watch-time censored regression** (CWM-style) | High | Research depth = innovation, but risky (CWM needs torch 1.6). Stretch only. |
| 6 | **Time features + drift** (`hourmin`, `date`, train/test drift) | Low–Med | — |
| 7 | **Unbiased validation** via `log_random_4_22_to_5_08_pure.csv` (1.18M randomized-exposure rows) | Med | Extra unbiased check that a change isn't just overfitting biased traffic. |

**Bottom line:** Priority #1 (loss-function swap) should clear the FM baseline on its own,
fast, on CPU. Do it first.

---

## 5. Recommended architecture for the agent

**Design an AIDE-style loop** (AIDE [2] is cited in the brief: ML engineering as code
optimization via tree search), specialized for RecSys.

**Two decisions that matter most:**

1. **Port the FM to a small PyTorch harness early.** Keep the numpy FM as the *verified
   reference* (must still reproduce 0.5946). But swapping losses (BPR/softmax/LambdaRank),
   adding sequence attention, and multi-task heads are trivial in PyTorch autograd and painful
   in hand-written numpy gradients. Build a **pluggable harness**:
   `(feature_module, model_module, loss_module, train_config)` wrapping
   `data.load()` → model → `evaluate.py`. The agent edits those four slots = the solution space.

2. **Seed the agent's knowledge with the kit's own ablation results.** Feed it
   `baseline_scores.json` + the README "already tested / dead ends" as prior knowledge so it:
   - doesn't re-test features/capacity (saves Feasibility budget),
   - reasons explicitly about *why* loss-alignment helps (Innovation points),
   - front-loads the big-ticket loss change first (avoids premature ε-convergence).

**Loop components:**
- **Fixed scaffold the agent only edits** (diff-based), not writes from scratch → far more reliable.
- **Tools:** `run_experiment` (train+eval on validation, returns structured JSON metrics),
  `read_logs`, `edit_code`, `record_finding`.
- **Reflect step with memory:** a running "research journal" of
  `{hypothesis, code diff, val metrics, delta vs baseline, error/recovery}`. This is
  simultaneously the reflect mechanism, the Autonomy/Robustness evidence, and a **required
  deliverable** (per-iteration run log). Make it first-class and automatic from iteration 1.
- **Robustness wrapper:** every experiment under timeout + try/catch; on failure the agent gets
  the stack trace and must fix/retry/route-around. Long runs must never crash or stall.
- **Anti-overfit self-check:** use `log_random` as a secondary *unbiased* validation signal.
  If a change helps biased validation but not the unbiased log, it's likely overfitting the
  logging policy and won't transfer to hidden test.

---

## 6. Model / loss roadmap (what the agent should try, in order)

1. Reproduce FM (`python baseline.py --model fm`) → confirm ≈ 0.5946. Also confirm
   `--model random` ≈ 0.475 (harness self-check).
2. **Within-user pairwise (BPR) or listwise softmax loss** → first baseline-beating result.
3. **LambdaRank / LambdaLoss** targeting nDCG@5.
4. **DIN-style user behavior sequence** modeling (needs torch).
5. **Multi-task** auxiliary heads.
6. Stretch: watch-time censored regression / ensembling.

---

## 7. Pitfalls specific to this spec

- **Premature convergence:** convergence triggers at ε=0.002 over 3 flat iterations. Make BIG
  early moves (loss swap) so a strong validation-best checkpoint is banked before small tweaks
  stall the run. **Scored on the converged result, not the peak.**
- **Validation overfitting:** final ranking is on hidden test, scored once. Prefer changes that
  generalize; use the `log_random` unbiased check.
- **Manage the ceiling:** movable range is ~0.27; modest deltas still score (continuous scoring).
- **Feasibility gate:** resource scoring (tokens + wall-clock, coarse low/med/high tiers) only
  counts **if you beat the baseline**. So beat it first, then keep LLM usage lean (summarize the
  run-log into context instead of re-dumping full history; cache the stable scaffold).
- **Autonomy counted literally:** log the number of manual interventions; target 0.

---

## 8. Judging criteria (weights)

| Criterion | Weight | What it rewards here |
|---|---|---|
| Technical Execution (primary metric + robustness) | 35% | Absolute delta over FM baseline on hidden test; graceful failure recovery. |
| Innovation & Problem Insight | 20% | *What* the agent chose to try and *why*; originality drawing on published methods. |
| Impact & Relevance (Autonomy) | 20% | How much of the loop the agent drives alone; fewer manual interventions. |
| Feasibility & Practicality (resources) | 15% | Token + wall-clock cost, gated on beating the baseline; coarse tiers. |
| Presentation & Communication | 10% (final event only) | Report / results table / run-log narrative / optional 3-min video. |

---

## 9. Deliverables

1. **Written project description** (Devpost): how it addresses the problem, dev tools, APIs,
   libraries/frameworks, datasets.
2. **Public GitHub repo:** well-structured commented code, README (overview, setup, reproduce
   steps, limitations/future work, team contributions).
3. **Run & iteration logs:** per-iteration hypothesis + code diff + metrics + error/recovery,
   plus a summary of the number of manual interventions.
4. **Final submission & results summary:** final KuaiRand-Pure output in the kit schema; results
   table (validation-best GAUC/nDCG@5 + absolute delta over baseline); reported resource usage
   (total LLM tokens in+out, agent wall-clock, iterations used out of 50; GPU-hours if any).

Video optional (~3 min recommended); without a video, a detailed report is highly encouraged.

---

## 10. Auth / setup notes

- **Using Claude Pro plan.** The Claude Agent SDK (Claude Code as a library) runs on the Claude
  Code CLI auth — logged in with the Pro subscription, usage counts against the subscription,
  **no separate API credits needed**. An unset `ANTHROPIC_API_KEY` is fine when logged in.
- Alternatively, a plain Python tool-call loop against the API works too, but that would need
  API credits. The Agent SDK saves writing the loop + file tools + retries.
- Challenge also allows any LLM coding agent (Trae from ByteDance offers a 7-day free trial).

---

## 11. Starter kit file reference

| File | What |
|---|---|
| `evaluate.py` | Metric implementation + all pinned conventions. **DO NOT MODIFY.** |
| `data.py` | Data loading, official split, feature encoding. Add features here. |
| `baseline.py` | Three baselines (`pop`/`fm`/`random`). FM is the one to beat. |
| `baseline_scores.json` | Official scores + seed variance + convergence params. |
| `submit.py` | Generate / validate / score submission files. |
| `ablation_features.py` | Reproduces the "adding features gives no gain" result. |
| `README.md` | Original starter-kit README (Chinese). |

---

## 12. Suggested first actions for the next instance

1. Download KuaiRand-Pure (§3) if not already present.
2. `python baseline.py --model fm` → confirm ≈ 0.5946; `--model random` → ≈ 0.475.
3. Read `evaluate.py` end-to-end to lock in the exact metric.
4. Build the pluggable PyTorch harness reproducing FM (verify against 0.5946).
5. Implement the within-user pairwise/listwise loss as experiment #1 (first baseline-beater).
6. Wrap in the autonomous loop + auto run-log, then let it explore directions #2–#4.
