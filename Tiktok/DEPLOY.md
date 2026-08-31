# Deploying the agent

One command. The agent does the rest — hypothesis, literature, code, training,
tuning, ensembling, and the final submission.

```powershell
.\run.ps1
```

That starts `run_agent.py` as a detached process and attaches the live dashboard
to this window. Press `q` to detach; the agent keeps running.

| Command | What it does |
|---|---|
| `.\run.ps1` | start the agent (50 iterations, 6h cap, 3 seeds) + dashboard |
| `.\run.ps1 -Hours 8 -MaxIterations 60` | same, different budget |
| `.\run.ps1 -Fresh` | archive the previous journal first, then start clean |
| `.\run.ps1 -DashboardOnly` | attach the dashboard to an agent already running |
| `.\run.ps1 -Status` | one-shot snapshot, no live view |
| `.\run.ps1 -Stop` | stop the agent (the journal survives; a restart resumes from it) |
| `.\run.ps1 -NoDashboard` | agent in the foreground, raw scrolling log |

Restarting is always safe: the agent reads `runs/journal.jsonl` on boot and
picks up at the next iteration with its full history in the digest.

## The iteration cap counts the whole journal

`--max-iterations` is a cap on the journal, not on this session. A journal with
34 experiments in it leaves only 16 of the default 50 for the new run, and the
agent will spend them accordingly. Two ways to give it a full budget:

```powershell
.\run.ps1 -MaxIterations 90     # keep the history, raise the ceiling
.\run.ps1 -Fresh                # archive the history, start the count at zero
```

Prefer `-MaxIterations`: the journal is what stops the agent re-testing things
already known not to work, and `-Fresh` throws that away along with the count.

## If the run stops unexpectedly

The dashboard tells the two cases apart. `STALLED` means the process is alive
but has not written status for 90 seconds — usually a very long epoch, check
`runs/agent_stdout.log`. `KILLED` means the process is gone: something outside
the run stopped it. Look at `runs/agent_stderr.log` — empty means it did not
crash, it was terminated.

Relaunching picks up from the journal. Note that a killed `ensemble_search`
loses its model library, which lives only in that process's memory; its
completed trials are still in the journal, but the portfolio has to be rebuilt.

## What the agent decides for itself

Everything inside the loop. It is handed a compact digest of the journal each
turn and chooses one move from its own toolbox:

| Tool | Used for |
|---|---|
| `run_experiment` | one config, one structural question |
| `tune` | Bayesian search (GP + EI, ASHA pruning) over a knob surface |
| `ensemble_search` | BO inside several structural families, then greedy ensemble selection over every model trained |
| `WebSearch` / `WebFetch` | read the literature before implementing a method |
| `Read` / `Edit` / `Write` | extend `agent_kit/models.py` and `agent_kit/losses.py` with new architectures and objectives, hot-reloaded on the next run |
| `record_finding` | park an insight or a dead end without spending a training run |
| `finalize` | designate the final submission and build `submission.csv` |

If the loop ends for any reason — iteration cap, wall clock, or convergence —
a closing turn runs whose only job is `finalize`, so a run never ends without a
designated submission.

## What it cannot touch

A `PreToolUse` hook denies writes to the files that define what "better" means:
the metric, the split, the encoder, the harness, the journal, and the
instrumentation (`agent_kit/status.py`, `agent_kit/progress.py`, `dashboard.py`).
An agent that can edit the metric can manufacture a win, and an agent that can
edit the dashboard can hide what it is doing. Denials are counted in
`runs/resources.json`.

## Reading the dashboard

```
PHASE     what it is doing right now, and whether it is alive
BEST      best validation primary so far, against the FM baseline and the oracle ceiling
BUDGET    iterations and wall clock against their caps
COST      tokens and dollars (a scored deliverable), plus ok/timeout/failed counts
CURRENT   the running experiment: hypothesis, config, seed, epoch curve
PORTFOLIO the last ensemble_search result
SCORE HISTORY  every scored experiment, with the FM baseline drawn through it
AGENT     the agent's own reasoning for this turn
EVENTS    new bests, timeouts, failures and recoveries as they happen
```

`STALLED` in red means no status write for 90 seconds — check
`runs/agent_stdout.log`.

## Files the run produces

| Path | What |
|---|---|
| `runs/status.json` | live state (the dashboard's only input besides the journal) |
| `runs/journal.jsonl` | append-only: every hypothesis, config, metric, error and recovery |
| `runs/RUN_LOG.md` | the same, rendered as the run-log deliverable |
| `runs/ensemble_search.json` | the selected portfolio, with member configs and weights |
| `runs/FINAL.json` | which submission was designated, and why |
| `runs/resources.json` | tokens, cost, wall clock, manual interventions, blocked writes |
| `submission.csv` | the submission itself |

## Requirements

```powershell
pip install claude-agent-sdk torch numpy pandas
```

The SDK drives the Claude Code CLI and inherits its login, so a Pro subscription
works with no API key — an unset `ANTHROPIC_API_KEY` is expected. If the CLI was
installed through the VS Code extension rather than npm, `run_agent.py` finds it
in the extension directory on its own; `--cli-path` overrides that.
