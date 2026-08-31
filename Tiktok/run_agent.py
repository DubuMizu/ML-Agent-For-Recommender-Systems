"""Autonomous ML research agent for KuaiRand-Pure (TikTok TechJam Challenge 2).

Architecture: an AIDE-style outer loop. Each iteration the agent is handed a
compact digest of the research journal -- not the full history -- and asked for
exactly one next experiment. It calls tools to run that experiment against a
fixed harness, reads back structured metrics, and reflects into the journal.

Three deliberate choices:

  * The agent edits config *and* code, but only inside two plug-in files
    (models.py, losses.py). Widening the space to new architectures and
    objectives is what reaches the directions that config alone cannot -- user
    behaviour sequences, multi-task heads. Everything that defines what "better"
    means (metric, split, encoder, harness) is refused by a PreToolUse hook, so a
    bad idea costs one iteration instead of invalidating the run.
  * Context per turn is the digest plus the search space, so token use stays flat
    as iterations accumulate rather than growing with history.
  * Every failure is structured, not fatal: a bad config, a syntax error in an
    agent edit, a divergent run and a slow run each come back as a result the
    agent can read and act on, and the outer loop survives an SDK error too.

Auth: runs on the Claude Code CLI login (Pro subscription). An unset
ANTHROPIC_API_KEY is expected and fine.

Usage:
    python run_agent.py --max-iterations 50 --seeds 3
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

from claude_agent_sdk import (
    AssistantMessage, ClaudeAgentOptions, HookMatcher, ResultMessage, TextBlock,
    create_sdk_mcp_server, query, tool,
)

from agent_kit import ensemble_search as ES
from agent_kit import status as STATUS
from agent_kit.experiment import safe_run, summarise
from agent_kit.journal import Journal, FM_VALID_PRIMARY, ORACLE_VALID_PRIMARY
from agent_kit.progress import Reporter
from agent_kit.tuner import tune, summarise_tuning
from agent_kit.models import REGISTRY as MODEL_REGISTRY
from agent_kit.losses import REGISTRY as LOSS_REGISTRY

# The agent writes mathematical text (minus signs, arrows, Greek) that the
# Windows console's default cp1252 codec cannot encode. Without this, a print
# raises UnicodeEncodeError mid-stream and the turn dies before its usage totals
# are read -- losing the token accounting that Feasibility is scored on.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

# Honour AGENT_RUNS_DIR the way journal.py and status.py already do, so a
# smoke test can point the whole run at a scratch directory instead of
# overwriting the real journal and the real ensemble result.
RUNS_DIR = os.environ.get('AGENT_RUNS_DIR', './runs')

JOURNAL = Journal()
REPORTER = Reporter()
STATE = {'seeds': (0, 1, 2), 'time_budget_s': 900, 'iterations': 0,
         'tokens_in': 0, 'tokens_out': 0, 'cost_usd': 0.0, 'interventions': 0}


# ------------------------------------------------------------------ tools ---
@tool('run_experiment',
      'Train and evaluate one configuration on the validation split, then record '
      'it in the research journal. This is the only way to get a number. Always '
      'supply the hypothesis you are testing and why you believe it.',
      {'type': 'object',
       'properties': {
           'hypothesis': {'type': 'string',
                          'description': 'What you are testing, in one sentence.'},
           'rationale': {'type': 'string',
                         'description': 'Why you expect this to help, grounded in the '
                                        'metric or in published method.'},
           'config': {'type': 'object',
                      'description': 'Experiment config: {"model": {...}, "loss": {...}, '
                                     '"train": {...}}. See get_search_space.'},
           'parent': {'type': 'integer',
                      'description': 'Iteration number this is derived from, if any.'},
       },
       'required': ['hypothesis', 'rationale', 'config']})
async def run_experiment_tool(args):
    cfg = args['config']
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    res = safe_run(cfg, seeds=STATE['seeds'], time_budget_s=STATE['time_budget_s'],
                   reporter=REPORTER, hypothesis=args['hypothesis'],
                   iteration=JOURNAL.next_iteration)
    entry = JOURNAL.record(
        hypothesis=args['hypothesis'], rationale=args.get('rationale'),
        config=cfg, result=res, status=res['status'], error=res.get('error'),
        recovery=res.get('hint'), parent=args.get('parent'))
    STATE['iterations'] += 1
    JOURNAL.render_markdown()
    _best = JOURNAL.best()
    REPORTER.finish(res, iteration=entry['iteration'],
                    is_best=bool(_best and _best['iteration'] == entry['iteration']))
    REPORTER.scoreboard(JOURNAL)
    # resource totals are a scored deliverable; write them per EXPERIMENT, not
    # per agent turn -- a turn can run many experiments and last hours, and a
    # kill mid-turn used to lose the whole accounting
    _save_resources(STATE.get('t0') or time.time(), STATE.get('start_iter', 0))

    body = ['iteration %d -- %s' % (entry['iteration'], res['status']), summarise(res)]
    if res['status'] == 'ok':
        body.append('delta vs FM baseline: %+.4f  (%.1f%% of the remaining headroom)'
                    % (entry['delta_vs_fm'], entry['pct_of_headroom']))
        body.append('per-seed valid primary: %s' % (res.get('per_seed_valid_primary'),))
        if res.get('valid_primary_seedens') is not None:
            body.append(
                'SEED ENSEMBLE valid primary: %.4f (%+.4f vs the mean above), '
                'GAUC %.4f, nDCG@5 %.4f, unbiased %.4f. This is what submitting '
                'this config would actually score -- averaging the seeds\' '
                'within-user ranks cancels seed noise instead of averaging it in. '
                'Prefer it when comparing two configs; the mean is the noisier '
                'view of the same thing.'
                % (res['valid_primary_seedens'], res.get('seedens_gain', 0.0),
                   res.get('valid_GAUC_seedens', float('nan')),
                   res.get('valid_nDCG@5_seedens', float('nan')),
                   res.get('unbiased_primary_seedens', float('nan'))))
        body.append('valid epoch curves: %s' % (res.get('epoch_curves'),))
        best = JOURNAL.best()
        if best and best['iteration'] == entry['iteration']:
            body.append('*** new best ***')
        elif best:
            body.append('current best is still iteration %d at %.4f'
                        % (best['iteration'], best['valid_primary']))
    else:
        body.append('traceback:\n%s' % res.get('traceback', '')[-1200:])
        body.append('suggested recovery: %s' % res.get('hint'))
    conv, why = JOURNAL.converged()
    body.append('convergence check: %s (%s)' % ('CONVERGED' if conv else 'not converged', why))
    return {'content': [{'type': 'text', 'text': '\n'.join(body)}]}


@tool('read_journal', 'Read the compact digest of every experiment run so far.',
      {'type': 'object', 'properties': {}})
async def read_journal_tool(args):
    return {'content': [{'type': 'text', 'text': JOURNAL.digest()}]}


@tool('get_search_space',
      'List the model types, loss types and training knobs the harness accepts, '
      'plus the prior knowledge you should not re-derive.',
      {'type': 'object', 'properties': {}})
async def get_search_space_tool(args):
    return {'content': [{'type': 'text', 'text': SEARCH_SPACE}]}


@tool('record_finding',
      'Record an insight, a dead end, or a plan in the journal without running an '
      'experiment. Use this to reason about what to try next.',
      {'type': 'object',
       'properties': {'note': {'type': 'string'},
                      'tags': {'type': 'array', 'items': {'type': 'string'}}},
       'required': ['note']})
async def record_finding_tool(args):
    JOURNAL.record(hypothesis=args['note'], config=None, status='note',
                   tags=args.get('tags', []))
    return {'content': [{'type': 'text', 'text': 'recorded'}]}


@tool('tune',
      'Run a Bayesian hyper-parameter search (GP + expected improvement, with '
      'ASHA pruning) over continuous knobs of a fixed structure, locally. Use '
      'this INSTEAD of spending one iteration per hyper-parameter value: it runs '
      'many trials and returns one summary. Best for lr / dropout p / l2 / '
      'group_size / k once you have chosen a model and loss.',
      {'type': 'object',
       'properties': {
           'hypothesis': {'type': 'string'},
           'base_config': {'type': 'object',
                           'description': 'The fixed structure: model + loss + train defaults.'},
           'space': {'type': 'object',
                     'description': 'Dotted-key search space, e.g. '
                                    '{"train.lr": ["float", 0.0001, 0.003, true], '
                                    '"model.p": ["float", 0.1, 0.7, false], '
                                    '"train.group_size": ["int", 2, 16]}. '
                                    'Forms: ["float", lo, hi, log] / ["int", lo, hi] / '
                                    '["cat", [v1, v2]].'},
           'n_init': {'type': 'integer', 'description': 'random warm-up trials (default 4)'},
           'n_iter': {'type': 'integer', 'description': 'BO trials after warm-up (default 8)'},
       },
       'required': ['hypothesis', 'base_config', 'space']})
async def tune_tool(args):
    space = {k: tuple(v) for k, v in args['space'].items()}
    out = tune(args['base_config'], space,
               n_init=args.get('n_init', 4), n_iter=args.get('n_iter', 8),
               seeds=STATE['seeds'][:2], time_budget_s=STATE['time_budget_s'],
               journal=JOURNAL, reporter=REPORTER, hypothesis=args['hypothesis'])
    STATE['iterations'] += out.get('n_trials', 0)
    JOURNAL.render_markdown()
    REPORTER.scoreboard(JOURNAL)
    _save_resources(STATE.get('t0') or time.time(), STATE.get('start_iter', 0))
    return {'content': [{'type': 'text', 'text': summarise_tuning(out)}]}


@tool('ensemble_search',
      'Run the full portfolio pipeline: Bayesian optimisation with ASHA pruning '
      'inside EACH structural family you name, keep every model trained along the '
      'way in a library, then pick a weighted ensemble by greedy forward '
      'selection (Caruana 2004). This is the single highest-leverage tool you '
      'have -- an ensemble of decorrelated families beats the best single model '
      'by roughly the size of the gain you are chasing, and a model that is '
      'mediocre alone can still be the most valuable member. Give it 3-5 families '
      'that differ STRUCTURALLY (different loss, different architecture); knobs '
      'alone do not decorrelate errors. Costs (families x (n_init+n_iter) x seeds) '
      'trainings, so budget it: it is the right way to spend a big block of time, '
      'not something to run every iteration.',
      {'type': 'object',
       'properties': {
           'hypothesis': {'type': 'string'},
           'families': {'type': 'object',
                        'description': 'Dict of family name -> {"base": <full config>, '
                                       '"space": {dotted knob: ["float", lo, hi, log] | '
                                       '["int", lo, hi] | ["cat", [values]]}}. Example: '
                                       '{"fm_softmax": {"base": {"model": {"type": '
                                       '"fm_dropout", "k": 16, "p": 0.4}, "loss": {"type": '
                                       '"softmax"}, "train": {"lr": 0.001, "batch": 8192, '
                                       '"epochs": 40, "patience": 4, "l2": 1e-4, '
                                       '"group_size": 8}}, "space": {"train.lr": ["float", '
                                       '3e-4, 2.5e-3, true]}}}'},
           'n_init': {'type': 'integer', 'description': 'random warm-up trials per family (default 2)'},
           'n_iter': {'type': 'integer', 'description': 'BO trials per family after warm-up (default 2)'},
           'seeds': {'type': 'integer', 'description': 'seeds per trial (default 2); each seed is a separate library member'},
           'max_ensemble': {'type': 'integer', 'description': 'max members with replacement (default 12)'},
           'features': {'type': 'object',
                        'description': 'encoding shared by every family, e.g. '
                                       '{"dur_buckets": 50, "crosses": [["video_id", "tab"]]}. '
                                       'Shared, not per-family: the library averages member '
                                       'ranks row by row and the portfolio has to be rebuildable '
                                       'from one encoding at submission time.'},
       },
       'required': ['hypothesis', 'families']})
async def ensemble_search_tool(args):
    fams_in = args['families']
    if isinstance(fams_in, str):
        fams_in = json.loads(fams_in)
    families = {}
    for name, spec in fams_in.items():
        if isinstance(spec, str):
            spec = json.loads(spec)
        base = spec.get('base') or spec.get('base_config')
        if not base:
            return {'content': [{'type': 'text', 'text':
                    'family %r has no "base" config; each family needs '
                    '{"base": {...}, "space": {...}}' % name}]}
        families[name] = (base, {k: tuple(v) for k, v in (spec.get('space') or {}).items()})

    n_seeds = int(args.get('seeds', 2))
    STATUS.update(phase='ensembling',
                  phase_detail='%d families: %s' % (len(families), ', '.join(families)))
    STATUS.event('tool', 'ensemble_search over %s' % ', '.join(families))
    try:
        out = ES.run(families, n_init=int(args.get('n_init', 2)),
                     n_iter=int(args.get('n_iter', 2)),
                     seeds=tuple(range(n_seeds)),
                     time_budget_s=STATE['time_budget_s'], journal=JOURNAL,
                     reporter=REPORTER, max_ensemble=int(args.get('max_ensemble', 12)),
                     features=args.get('features'))
    except Exception as exc:                                      # noqa: BLE001
        import traceback
        JOURNAL.record(hypothesis=args['hypothesis'], config=None, status='failed',
                       error='%s: %s' % (type(exc).__name__, exc),
                       recovery='ensemble_search raised; the agent can fix the '
                                'family specs and retry')
        STATUS.event('fail', 'ensemble_search failed: %s' % exc)
        return {'content': [{'type': 'text', 'text':
                'ensemble_search FAILED: %s: %s\n%s\nCheck that every family base '
                'config is a complete {model, loss, train} dict and every space entry '
                'is ["float", lo, hi, log] / ["int", lo, hi] / ["cat", [...]].'
                % (type(exc).__name__, exc, traceback.format_exc(limit=6))}]}

    scored = [t for t in out['trials'] if t.get('score') is not None]
    STATE['iterations'] += len(scored)
    text = ES.summarise(out)

    library, weights = out.pop('library'), out.pop('weights')
    members = [{'key': e['key'], 'weight': float(w), 'seed': e['seed'],
                'config': e['config']}
               for w, e in zip(weights, library.entries) if w > 0]
    payload = dict(out, member_configs=members)
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(os.path.join(RUNS_DIR, 'ensemble_search.json'), 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, default=str)

    # Recorded under a status the journal's best() ignores on purpose: an
    # ensemble is not a config submit_final.py could train from --best, and
    # letting it win best() would silently break the single-model path.
    JOURNAL.record(
        hypothesis=args['hypothesis'], config=None, status='ensemble',
        rationale='portfolio over %d structural families' % len(families),
        notes={'ensemble_valid': out['ensemble_valid'],
               'ensemble_unbiased': out['ensemble_unbiased'],
               'best_single': out['best_single'], 'members': out['members']},
        tags=['ensemble'])
    JOURNAL.render_markdown()
    STATUS.set_section('ensemble', {
        'valid': out['ensemble_valid'], 'unbiased': out['ensemble_unbiased'],
        'best_single': out['best_single'],
        'delta': out['delta_vs_fm'], 'n_models': out['n_models'],
        'n_members': len(members), 'updated': time.time()})
    STATUS.event('best' if out['delta_vs_fm'] > 0 else 'ok',
                 'ensemble %.4f (%+.4f vs FM, %+.4f vs best single)'
                 % (out['ensemble_valid'], out['delta_vs_fm'],
                    out['gain_over_best_single']))
    _save_resources(STATE.get('t0') or time.time(), STATE.get('start_iter', 0))
    REPORTER.scoreboard(JOURNAL)
    return {'content': [{'type': 'text', 'text': text + (
        '\n\nSaved to runs/ensemble_search.json. To submit this ensemble rather '
        'than a single model, call finalize with choice="ensemble".')}]}


@tool('finalize',
      'Designate the final submission and build submission.csv from it. This is '
      'the ONLY step that touches the hidden test split, and it touches it only '
      'to score rows -- never to select a model. Call it once, at the end of the '
      'run, when you are done improving. Choose "ensemble" if the ensemble beats '
      'the best single model on validation AND moves the unbiased log the same '
      'way; choose "best"/"iteration" otherwise.',
      {'type': 'object',
       'properties': {
           'choice': {'type': 'string', 'enum': ['best', 'iteration', 'ensemble'],
                      'description': '"best" = best journal entry; "iteration" = a '
                                     'specific one; "ensemble" = runs/ensemble_search.json'},
           'iteration': {'type': 'integer'},
           'reason': {'type': 'string',
                      'description': 'Why this is the right final submission, including '
                                     'what the unbiased log says about it.'},
           'seeds': {'type': 'integer', 'description': 'seeds to train for the single-model path (default 5)'},
       },
       'required': ['choice', 'reason']})
async def finalize_tool(args):
    choice = args['choice']
    cmd = [sys.executable, 'submit_final.py', '--score-valid']
    if choice == 'ensemble':
        cmd += ['--ensemble-json', os.path.join(RUNS_DIR, 'ensemble_search.json')]
    elif choice == 'iteration':
        if args.get('iteration') is None:
            return {'content': [{'type': 'text', 'text':
                    'choice="iteration" needs an iteration number.'}]}
        cmd += ['--iteration', str(int(args['iteration'])), '--seeds', str(int(args.get('seeds', 5)))]
    else:
        cmd += ['--best', '--seeds', str(int(args.get('seeds', 5)))]

    STATUS.update(phase='submitting', phase_detail=choice)
    STATUS.event('tool', 'finalize: %s -- %s' % (choice, str(args['reason'])[:160]))
    print('[final] %s' % ' '.join(cmd), flush=True)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              encoding='utf-8', errors='replace')
        output = (proc.stdout or '') + (proc.stderr or '')
        ok = proc.returncode == 0
    except Exception as exc:                                      # noqa: BLE001
        output, ok = '%s: %s' % (type(exc).__name__, exc), False
    print(output[-4000:], flush=True)

    record = {'choice': choice, 'iteration': args.get('iteration'),
              'reason': args['reason'], 'ok': ok,
              'submission': 'submission.csv' if ok else None,
              'log_tail': output[-3000:],
              'designated_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(os.path.join(RUNS_DIR, 'FINAL.json'), 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=2)
    JOURNAL.record(hypothesis='FINAL SUBMISSION: %s' % choice, config=None,
                   status='final', rationale=args['reason'],
                   notes={'ok': ok}, tags=['final'])
    STATE['finalized'] = ok
    STATUS.set_section('final', {k: v for k, v in record.items() if k != 'log_tail'})
    STATUS.event('best' if ok else 'fail',
                 'FINAL submission (%s): %s' % (choice, 'written' if ok else 'FAILED'))
    return {'content': [{'type': 'text', 'text':
            ('submission written to submission.csv\n\n' if ok
             else 'FINALIZE FAILED -- read the output, fix it, and call finalize again\n\n')
            + output[-3000:]}]}


# Files the agent must never modify. The metric, the split and the encoder define
# what "better" means; an agent that can edit them can trivially manufacture a
# win, and any number it produced afterwards would be meaningless. Extending
# models.py / losses.py is the intended way to widen the search space.
PROTECTED = ('evaluate.py', 'data.py', 'baseline.py', 'submit.py',
             'baseline_scores.json', 'agent_kit/dataset.py', 'agent_kit/metrics.py',
             'agent_kit/harness.py', 'agent_kit/journal.py', 'agent_kit/experiment.py',
             'run_agent.py', 'submit_final.py', 'tests/test_metrics.py',
             # instrumentation: the run is supervised through these, and an
             # agent that edits them can blind the operator mid-run
             'agent_kit/status.py', 'agent_kit/progress.py', 'dashboard.py')


async def guard_pre_tool_use(input_data, tool_use_id, context):
    """PreToolUse hook: refuse writes to the files that define the metric.

    This must be a hook, not a can_use_tool callback. Under
    permission_mode='bypassPermissions' the SDK auto-approves every tool call
    before can_use_tool is ever consulted, so that callback is silently inert --
    the SDK warns about exactly this. A PreToolUse hook runs regardless of
    permission mode, which is what makes the guarantee real.
    """
    tool_name = input_data.get('tool_name', '')
    tool_input = input_data.get('tool_input', {}) or {}
    if tool_name in ('Edit', 'Write', 'NotebookEdit', 'MultiEdit'):
        path = str(tool_input.get('file_path', '')).replace('\\', '/')
        for prot in PROTECTED:
            if path.endswith(prot):
                STATE['denied_writes'] = STATE.get('denied_writes', 0) + 1
                print('[guard] denied write to %s' % prot, flush=True)
                return {'hookSpecificOutput': {
                    'hookEventName': 'PreToolUse',
                    'permissionDecision': 'deny',
                    'permissionDecisionReason': (
                        '%s defines the evaluation contract (metric, split, encoder) '
                        'and is read-only for the whole run -- results would not be '
                        'comparable if it changed. Extend agent_kit/models.py or '
                        'agent_kit/losses.py instead: add a new @register(...) entry '
                        'and select it by name from the experiment config.' % prot)}}
    return {}


SEARCH_SPACE = """\
CONFIG SHAPE
  {"features": {...}, "model": {"type": <model>, ...}, "loss": {"type": <loss>, ...},
   "train": {"lr":, "batch":, "epochs":, "patience":, "l2":, "group_size":}}

FEATURES -- THE ENCODING ITSELF IS SEARCHABLE (this is new)
  "features": {"dur_buckets": 10, "crosses": [["video_id", "tab"]]}

  dur_buckets  how finely duration is discretised. Default 10, which is the
               official baseline value. long_view is DEFINED against duration,
               and measured on validation the duration signal alone ranks at
               GAUC 0.5319 with 10 buckets versus 0.5582 with 200 -- the default
               is throwing information away. Try 20 / 50 / 200.
  crosses      explicit conjunction fields, appended to the 5 base fields.
               Crossable: user_id, video_id, author_id, tab, dur_bucket.
               A video x tab rate ranks at GAUC 0.6479 alone against 0.6387 for
               a plain video rate. FM does cross fields at order 2 through their
               embeddings, but one id is far more sample-efficient than a k-dim
               embedding cross for a video seen a handful of times.

  BOTH ADD PARAMETERS, so they move the regularisation optimum. This regime is
  overfitting-limited: a naive 2-epoch A/B of both came out NEUTRAL to slightly
  negative against the baseline encoding. Do not test them at fixed p/l2 and
  conclude anything -- tune lr, p and l2 jointly with the encoding knob, or you
  will be measuring under-regularisation rather than the feature.

MODELS   %s
  fm          k (default 16)                 -- the official baseline architecture
  fm_dropout  k, p (embedding dropout)       -- FM overfits after ~epoch 7
  deepfm      k, hidden (list), p            -- FM + MLP over field embeddings

LOSSES   %s
  bce         (pointwise log-loss; the baseline objective, the control)
  bpr         within-user pairwise
  softmax     within-user listwise, uses train.group_size negatives+1
  lambdarank  k (=5), sigma; nDCG@5-weighted pairs, uses train.group_size

TRAIN KNOBS
  lr (0.001), batch (8192), epochs (40), patience (4), l2 (1e-6),
  group_size (8, listwise only), steps_per_epoch (auto)

EXTENDING THE SPACE WITH CODE
  You are not limited to the configs above. You may Read and Edit exactly two
  files to add new architectures and objectives:

    agent_kit/models.py  -- add @register('name') on an nn.Module whose
                            forward(X) maps X (B, F) int64 field ids -> logit (B,)
    agent_kit/losses.py  -- add @register('name', kind) where kind is
                            'point' | 'pair' | 'list'; the sampler builds
                            {'X','y'} / {'Xp','Xn'} / {'Xg'} respectively

  Your edit is hot-reloaded on the next run_experiment, so add the class and
  then select it by name from the config. Everything else -- the metric, the
  split, the encoder, the harness -- is read-only and will refuse writes. That
  is deliberate: those files define what "better" means, and a run in which they
  changed would not be comparable to the baseline.

YOUR TOOLS, AND WHEN EACH IS THE RIGHT ONE
  run_experiment   one config, one question. Use it to test a STRUCTURAL idea.
  tune             many trials over continuous knobs of a fixed structure, in one
                   call. Never spend iterations hand-stepping lr or dropout.
  ensemble_search  BO inside several structural families + greedy ensemble
                   selection over every model trained. The biggest single lever
                   available; plan a block of time for it once you know which
                   families are worth including.
  WebSearch        read the literature. You are expected to use it: when you are
  WebFetch         about to invent a method, check what has been published on
                   this benchmark, on within-user ranking losses, or on the
                   specific failure you are looking at, and implement the real
                   version rather than your guess. Cite what you used in the
                   hypothesis so the journal records where the idea came from.
  record_finding   park an insight or a dead end without spending a training run.
  finalize         designate the final submission and build submission.csv. The
                   last thing you do.

DIRECTIONS THE ORGANIZERS BELIEVE HOLD THE REMAINING HEADROOM
  Ranked by their estimate, with the loss swap (1) already done:
    1. within-user pairwise / listwise loss   [DONE -- see the journal]
    2. LambdaRank / LambdaLoss on nDCG@5      [available as 'lambdarank']
    3. user behaviour sequences (DIN / SIM target attention) -- completely
       unused so far; each user has hundreds to thousands of train interactions,
       and the frames carry time_ms so a per-user history is reconstructable
    4. multi-task auxiliary heads (is_click / is_like / is_follow / is_comment /
       is_forward / play_time_ms) to regularise the sparse long_view target
    5. watch-time censored regression (CWM-style) -- research depth, higher risk
    6. time features and train/test drift (hourmin, date)
  Items 3 and 4 need code, not just config. That is what the Edit tool is for.

A DIRECTION THE JOURNAL CLOSED TOO EARLY: EXPLICIT FEATURE CROSSING
  deepfm scored 0.5948 and 0.5983 at iterations 5 and 6 -- below the FM
  baseline -- and the journal reads as a closed negative. Read those two entries
  before you believe it. Both peaked at best_epoch=1 on EVERY seed, which is the
  exact pathology diagnosed later from iteration 11's epoch curves (validation
  peaks at epoch 1, then falls off a cliff) and fixed at iteration 12, where
  p=0.3 with l2 raised 100x to 1e-4 moved the peak to epochs 2-4 and broke the
  plateau for the first time. Iteration 22 is the refinement of that same recipe
  and is the current best. This regime is overfitting-limited, not
  capacity-limited. deepfm was measured with
  p=0.2 and l2=1e-06, i.e. entirely inside the broken regime and never under the
  regularisation recipe that produced the current best. Its result is therefore
  CONFOUNDED, not settled -- it says the MLP overfits fastest, not that explicit
  crossing has no value here.

  Two function classes in that family have never been tried at all:
    * DCNv2 (Wang et al., 2021) -- CrossNetV2 stacks explicit bounded-degree
      crosses with far fewer parameters per unit of interaction order than an
      MLP, which is the right shape for a regime where everything dies of
      overfitting by epoch 2.
    * xDeepFM (Lian et al., 2018) -- CIN learns vector-wise crosses explicitly,
      a different inductive bias from DeepFM's bit-wise MLP.
  Neither is in the registry; both need code in agent_kit/models.py. Look the
  architectures up with WebSearch rather than reconstructing them from memory --
  CrossNetV2's low-rank projection and CIN's Hadamard-then-sum-pool step are
  both easy to get subtly wrong, and a wrong implementation reads as a negative
  result you will then wrongly trust.

  Temper the expectation with the two facts below: only 5 sparse ID fields
  exist, and anything constant within a user cancels out of the metric. Whatever
  a cross network learns has to vary per candidate video to move the score at
  all. Test one of them under the iteration-22 regularisation recipe, not under
  deepfm's original settings.

WHAT IS ALREADY KNOWN -- DO NOT SPEND ITERATIONS RE-TESTING
  * Adding static features (all 13 CWM feature fields) does NOT help: the
    organizers measured 0.5940 vs 0.5950. The user_id x video_id cross already
    saturates the learnable signal.
  * More EMBEDDING WIDTH does not help: k = 8 / 16 / 32 all score the same.
    1.14M rows cannot support more parameters of that kind. Note precisely what
    this measured -- the width of a rank-k bilinear form, not its ORDER. It is
    evidence against a bigger FM; it is not evidence against a different
    interaction function, which is a separate question the journal has not
    cleanly answered (see the crossing direction above).
  * STRUCTURAL FACT: scoring is within-user, so any term constant within a user
    cancels out of that user's ordering. Pure user-side first-order features
    therefore contribute EXACTLY ZERO. User-side features can only help through
    crosses with item-side features.

MEASUREMENT DISCIPLINE
  * FM seed sigma is 0.0008 on a single seed. You are running %d seeds, so the
    standard error is about %.4f. A delta smaller than that is NOT a result --
    treat it as noise and say so.
  * 'unbiased' is the randomised-exposure log (log_random). Its scale differs
    from validation (random 0.3120, oracle 0.6854) so compare it only against
    other experiments, never against the validation number. A change that helps
    validation but not the unbiased log is probably fitting the logging policy,
    which will not transfer to the hidden test.
  * The hidden test is a LATER time window than validation, so robustness to
    distribution shift matters more than squeezing validation.
""" % (sorted(MODEL_REGISTRY), sorted(LOSS_REGISTRY), 3, 0.0008 / (3 ** 0.5))


SYSTEM_PROMPT = """\
You are an autonomous ML research agent working on the KuaiRand-Pure recommender
benchmark. You run the full ML engineering loop yourself: form a hypothesis,
run the experiment, read the result, reflect, and choose the next move.

THE TASK
  Within-user ranking over each user's logged impressions. Label is `long_view`.
  Primary metric = mean(GAUC, nDCG@5) on validation.
  Official FM baseline: %.4f. Oracle ceiling: %.4f. You must beat the baseline
  and keep improving toward the ceiling. Judge progress against the ceiling,
  not against 1.0.

  You develop on train + validation only. The hidden test is never available to
  you and you must never ask for it.

YOU OWN THE WHOLE RUN
  Nobody is going to launch a script for you, pick your search space, run the
  ensemble, or decide what gets submitted. Every one of those is yours, through
  the tools listed by get_search_space. A human watching this run should never
  need to intervene; if you find yourself wishing someone would run something,
  that is a tool call you have not made yet.

HOW YOU WORK
  * Each turn, propose and run exactly ONE experiment with run_experiment,
    unless a failure needs an immediate fix -- then fix it and re-run.
  * Research is part of the loop, not a detour. Before committing code for a new
    architecture or objective, search for how it is actually specified in the
    literature -- WebSearch and WebFetch are available and their cost is small
    next to a training run spent on a misremembered formula.
  * Spend the whole budget. Convergence on single models is not the end of the
    run: an ensemble over decorrelated families reliably adds more than the last
    few single-model tweaks, so if you are near-converged and time remains, run
    ensemble_search rather than stopping.
  * Before the run ends you must call finalize exactly once. A run that improves
    on the baseline but designates nothing has produced nothing.
  * You may extend the search space by editing agent_kit/models.py and
    agent_kit/losses.py to add new architectures or objectives; call
    get_search_space for the contract. The metric, data pipeline and harness are
    read-only by design and writes to them are refused.
  * Every experiment needs a real hypothesis and a rationale grounded in the
    metric or in published method. "Try a bigger k" with no reason is a wasted
    iteration and k is already known not to matter.
  * Read the result honestly. If a delta is inside the noise band, say so and
    do not build on it. Negative results are worth recording -- they narrow the
    space -- but do not repeat them.
  * If an experiment fails, read the traceback and hint, fix the config, retry.
    Do not abandon a direction because of one crash.
  * Make BIG moves early. Convergence is declared when the best result stops
    improving by more than 0.002 over 3 consecutive iterations, so opening with
    small tweaks will stall the run before the real gains are banked.

BE HONEST
  Your job is a real improvement that transfers to a held-out later time window,
  not a number that looks good on validation. If you think you are overfitting
  validation, say so and check the unbiased signal.

RESOURCES ARE SCORED
  Token spend and wall-clock are part of your evaluation, but only if you beat
  the baseline. Beat it first, then stay lean: read the digest rather than
  re-deriving history, and do not re-run an experiment the journal already
  answers.

Reply with a short statement of what you are trying and why, then call the tool.
Keep it to two or three sentences -- it is read live on a dashboard.
""" % (FM_VALID_PRIMARY, ORACLE_VALID_PRIMARY)


TURN_PROMPT = """\
Iteration {n} of {maxn}. Elapsed {elapsed:.0f} min of {budget:.0f} min.

{digest}

{convergence}

Choose the single most promising next move and make it. State your hypothesis
and reasoning first, in two or three sentences, then call the tool that fits:
run_experiment for one structural question, tune for a knob surface,
ensemble_search for a portfolio, WebSearch/WebFetch first if the method needs
grounding.

If single-model search has genuinely converged but budget remains, run
ensemble_search instead of stopping. Say exactly CONVERGED only when no
direction is left worth testing AND the portfolio has already been built -- then
call finalize to designate the submission.
"""


CLOSING_PROMPT = """\
The run is ending ({why}). Nothing further will be trained.

{digest}

{ensemble}

Designate the final submission now by calling finalize exactly once. Choose the
option with the best validation primary whose unbiased-log score moved the same
way -- a validation gain the unbiased log does not corroborate is a fit to the
logging policy and will not survive the hidden test's later time window. State
the reason in one or two sentences.
"""



OPENING_PROMPT = """\
OPERATOR DIRECTIVE  (shown on turn {n} of the first {of} turns of this run)

The operator has set a priority task for the opening of this run. Work on it
before pursuing your own agenda. When it is answered -- including when the
answer is negative -- record the verdict with record_finding, say plainly that
the directive is complete, and go back to choosing your own moves.

{task}

Nothing else about how you work changes. The measurement discipline, the noise
floor and the honesty requirements all still apply: if the directive's idea
turns out not to work, the useful output is a clean negative, not a forced win.
"""


def load_opening_move(value):
    """--opening-move is either the directive text or a path to a file of it.

    A useful directive runs to several paragraphs -- which architecture, under
    which recipe, measured against which control -- and that does not survive
    being typed as a shell argument. Treating an existing path as a file is the
    difference between the flag being usable and being theoretical.
    """
    if not value:
        return None
    try:
        if os.path.exists(value):
            with open(value, encoding='utf-8') as fh:
                return fh.read().strip()
    except OSError:
        pass
    return value.strip()


def find_cli():
    """Locate the Claude Code CLI the SDK drives.

    The SDK shells out to the CLI and inherits its login, so a Pro subscription
    works with no API key. On a machine where Claude Code was installed through
    the VS Code extension rather than npm, the binary is not on PATH -- look in
    the extension directory before giving up.
    """
    import glob
    import shutil
    found = shutil.which('claude')
    if found:
        return found
    patterns = [
        os.path.expanduser('~/.vscode/extensions/anthropic.claude-code-*/'
                           'resources/native-binary/claude*'),
        os.path.expanduser('~/.vscode-server/extensions/anthropic.claude-code-*/'
                           'resources/native-binary/claude*'),
        os.path.expanduser('~/.local/bin/claude'),
        os.path.expanduser('~/AppData/Roaming/npm/claude.cmd'),
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]                       # newest version wins
    return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-iterations', type=int, default=50)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--hours', type=float, default=6.0)
    ap.add_argument('--time-budget-s', type=int, default=900)
    ap.add_argument('--model', default='claude-opus-5')
    ap.add_argument('--cli-path', default=None, help='override CLI autodiscovery')
    ap.add_argument('--no-finalize', action='store_true',
                    help='skip the closing turn that designates a submission')
    ap.add_argument('--opening-move', default=None,
                    help='operator directive for the opening turns: the text itself, or a path to a file containing it')
    ap.add_argument('--opening-turns', type=int, default=8,
                    help='how many turns the directive stays in the prompt')
    a = ap.parse_args()

    cli = a.cli_path or find_cli()
    if not cli:
        raise SystemExit(
            'Could not find the Claude Code CLI. The Agent SDK drives it and '
            'inherits its login. Install it (npm i -g @anthropic-ai/claude-code) '
            'or pass --cli-path.')
    print('[loop] using CLI at %s' % cli, flush=True)

    STATE['seeds'] = tuple(range(a.seeds))
    STATE['time_budget_s'] = a.time_budget_s
    opening = load_opening_move(a.opening_move)
    if opening:
        print('[loop] operator directive active for the first %d turns (%d chars)'
              % (a.opening_turns, len(opening)), flush=True)

    server = create_sdk_mcp_server('research', tools=[
        run_experiment_tool, read_journal_tool, get_search_space_tool,
        record_finding_tool, tune_tool, ensemble_search_tool, finalize_tool])
    options = ClaudeAgentOptions(
        model=a.model,
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={'research': server},
        # Read/Edit/Write let the agent extend models.py and losses.py, so the
        # search space includes new architectures and objectives -- not just
        # hyper-parameters. guard_tool_use below keeps the metric and the data
        # pipeline out of reach.
        allowed_tools=['mcp__research__run_experiment', 'mcp__research__read_journal',
                       'mcp__research__get_search_space', 'mcp__research__record_finding',
                       'mcp__research__tune', 'mcp__research__ensemble_search',
                       'mcp__research__finalize',
                       # literature access: the agent is asked to implement
                       # published methods, and guessing at a formula it could
                       # have looked up costs a whole training run
                       'WebSearch', 'WebFetch',
                       'Read', 'Edit', 'Write'],
        hooks={'PreToolUse': [HookMatcher(matcher='Edit|Write|NotebookEdit|MultiEdit',
                                          hooks=[guard_pre_tool_use])]},
        permission_mode='bypassPermissions',
        # a turn that researches before it experiments needs room: search,
        # fetch, read the plug-in file, edit it, then run. 12 truncated those
        # turns mid-edit and the iteration was wasted.
        max_turns=30,
        cli_path=cli,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    t0 = time.time()
    budget_s = a.hours * 3600
    start_iter = STATE['iterations'] = len(
        [e for e in JOURNAL.entries if e.get('status') != 'note'])
    STATE['t0'], STATE['start_iter'] = t0, start_iter
    STATUS.update(phase='starting', started=t0, budget_s=budget_s,
                  max_iterations=a.max_iterations, iteration=STATE['iterations'],
                  model=a.model, seeds=list(STATE['seeds']),
                  time_budget_s=STATE['time_budget_s'], alive=True)
    STATUS.event('info', 'run started: %d/%d iterations used, %.1fh budget'
                 % (start_iter, a.max_iterations, a.hours))
    # PID file so run.ps1 -Stop / -DashboardOnly work even when the agent was
    # launched directly rather than through the script.
    try:
        with open(os.path.join(RUNS_DIR, 'agent.pid'), 'w') as fh:
            fh.write(str(os.getpid()))
    except OSError:
        pass
    REPORTER.scoreboard(JOURNAL)          # publishes best+history to the dashboard

    while STATE['iterations'] < a.max_iterations:
        elapsed = time.time() - t0
        if elapsed > budget_s:
            print('[loop] wall-clock budget reached; stopping', flush=True)
            break
        conv, why = JOURNAL.converged()
        conv_txt = ('CONVERGENCE CHECK: %s -- %s' %
                    ('converged' if conv else 'not converged', why))
        prompt = TURN_PROMPT.format(
            n=STATE['iterations'] + 1, maxn=a.max_iterations,
            elapsed=elapsed / 60, budget=budget_s / 60,
            digest=JOURNAL.digest(), convergence=conv_txt)

        # The directive rides in front of the turn prompt for a bounded number
        # of turns, then drops out. Bounded because a multi-step task (research
        # it, write it, verify it trains, tune it) does not fit in one turn, and
        # dropped because a directive that never expires would quietly become a
        # permanent change to the agent's objective.
        STATE['turns'] = STATE.get('turns', 0) + 1
        if opening and STATE['turns'] <= a.opening_turns:
            prompt = OPENING_PROMPT.format(task=opening, n=STATE['turns'],
                                           of=a.opening_turns) + '\n\n' + prompt
            STATUS.update(directive_turns_left=a.opening_turns - STATE['turns'] + 1)
        elif STATE.get('directive_cleared') is None and opening:
            STATE['directive_cleared'] = True
            STATUS.event('info', 'operator directive expired; agent is on its own agenda')
            STATUS.update(directive_turns_left=0)

        STATUS.update(iteration=STATE['iterations'], elapsed_s=elapsed,
                      phase='thinking', phase_detail='choosing the next move',
                      converged=conv, converged_why=why)
        before = STATE['iterations']
        said_converged = False
        try:
            async for msg in query(prompt=prompt, options=options):
                if isinstance(msg, AssistantMessage):
                    for b in msg.content:
                        if isinstance(b, TextBlock) and b.text.strip():
                            print('[agent] %s' % b.text.strip()[:2000], flush=True)
                            STATUS.update(agent_text=b.text.strip()[:1200],
                                          agent_text_at=time.time())
                            STATUS.event('agent', b.text.strip()[:300])
                            if 'CONVERGED' in b.text:
                                said_converged = True
                elif isinstance(msg, ResultMessage):
                    _accumulate_usage(msg)
                    STATUS.set_section('tokens', {
                        'in': STATE['tokens_in'], 'out': STATE['tokens_out'],
                        'cost_usd': round(STATE['cost_usd'], 4)})
        except Exception as exc:                                   # noqa: BLE001
            # A transport/API failure must not end the run: log it and continue.
            print('[loop] agent turn failed (%s: %s); retrying next turn'
                  % (type(exc).__name__, exc), flush=True)
            STATUS.event('fail', 'agent turn failed (%s); retrying'
                         % type(exc).__name__)
            JOURNAL.record(hypothesis='(agent turn failed)', config=None,
                           status='failed', error='%s: %s' % (type(exc).__name__, exc),
                           recovery='outer loop caught it and continued to the next turn')
            await asyncio.sleep(5)
            continue

        if said_converged and conv:
            print('[loop] agent declared convergence and the rule agrees; stopping',
                  flush=True)
            break
        if STATE['iterations'] == before:
            print('[loop] no experiment ran this turn; nudging', flush=True)

        _save_resources(t0, start_iter)

    if not a.no_finalize and not STATE.get('finalized'):
        await closing_turn(options, t0, budget_s, why=_stop_reason(
            STATE['iterations'], a.max_iterations, time.time() - t0, budget_s))

    _save_resources(t0, start_iter)
    JOURNAL.render_markdown()
    STATUS.update(iteration=STATE['iterations'], elapsed_s=time.time() - t0)
    STATUS.finish(ok=True, note='finalized' if STATE.get('finalized')
                  else 'ended without a designated submission')
    b = JOURNAL.best()
    print('\n=== run complete ===')
    print('iterations: %d' % STATE['iterations'])
    if b:
        print('best: iteration %d  valid primary %.4f (%+.4f vs FM)'
              % (b['iteration'], b['valid_primary'], b['delta_vs_fm']))
        print('config: %s' % json.dumps(b['config']))
    print('tokens in/out: %d / %d   cost $%.2f'
          % (STATE['tokens_in'], STATE['tokens_out'], STATE['cost_usd']))
    print('wall clock: %.1f min' % ((time.time() - t0) / 60))
    try:
        os.remove(os.path.join(RUNS_DIR, 'agent.pid'))
    except OSError:
        pass


def _stop_reason(iters, max_iters, elapsed, budget_s):
    if iters >= max_iters:
        return 'the %d-iteration cap is reached' % max_iters
    if elapsed >= budget_s:
        return 'the wall-clock budget is spent'
    return 'search has converged'


async def closing_turn(options, t0, budget_s, why):
    """One last turn whose only job is to designate the final submission.

    Separate from the main loop because the loop can end for three different
    reasons -- cap, clock, convergence -- and in two of them the agent never gets
    another turn. Without this, a run that hits its iteration cap mid-thought
    leaves submission.csv unwritten and the whole run undeliverable.
    """
    ens = {}
    path = os.path.join(RUNS_DIR, 'ensemble_search.json')
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as fh:
                d = json.load(fh)
            ens = {k: d.get(k) for k in ('ensemble_valid', 'ensemble_unbiased',
                                         'best_single', 'delta_vs_fm')}
        except Exception:                                         # noqa: BLE001
            ens = {}
    ens_txt = ('PORTFOLIO RESULT (runs/ensemble_search.json): %s' % json.dumps(ens)
               if ens else 'No ensemble was built during this run.')
    print('[loop] closing turn: designating the final submission', flush=True)
    STATUS.update(phase='finalizing', phase_detail=why)
    prompt = CLOSING_PROMPT.format(why=why, digest=JOURNAL.digest(), ensemble=ens_txt)
    for attempt in (1, 2):
        try:
            async for msg in query(prompt=prompt, options=options):
                if isinstance(msg, AssistantMessage):
                    for blk in msg.content:
                        if isinstance(blk, TextBlock) and blk.text.strip():
                            print('[agent] %s' % blk.text.strip()[:1200], flush=True)
                            STATUS.event('agent', blk.text.strip()[:300])
                elif isinstance(msg, ResultMessage):
                    _accumulate_usage(msg)
            if STATE.get('finalized'):
                return True
        except Exception as exc:                                  # noqa: BLE001
            print('[loop] closing turn failed (%s: %s)' % (type(exc).__name__, exc),
                  flush=True)
        if attempt == 1:
            prompt = ('You did not call finalize. Call it now -- one call, with '
                      'choice and reason. Nothing else is needed.\n\n' + prompt)
    STATUS.event('fail', 'no final submission was designated')
    print('[loop] WARNING: the agent did not designate a final submission; '
          'run submit_final.py --best by hand', flush=True)
    return False


def _accumulate_usage(msg):
    """Fold one turn's token/cost totals into STATE.

    Kept defensive on purpose: resource reporting is a scored deliverable, but a
    surprise in the usage payload must never be what ends a 50-iteration run.
    """
    try:
        u = msg.usage or {}
        STATE['tokens_in'] += (u.get('input_tokens', 0)
                               + u.get('cache_read_input_tokens', 0)
                               + u.get('cache_creation_input_tokens', 0))
        STATE['tokens_out'] += u.get('output_tokens', 0)
        STATE['cost_usd'] += msg.total_cost_usd or 0.0
        for model, mu in (msg.model_usage or {}).items():
            per = STATE.setdefault('per_model', {}).setdefault(
                model, {'in': 0, 'out': 0, 'cost_usd': 0.0})
            per['in'] += mu.get('inputTokens', 0) + mu.get('cacheReadInputTokens', 0)                 + mu.get('cacheCreationInputTokens', 0)
            per['out'] += mu.get('outputTokens', 0)
            per['cost_usd'] += mu.get('costUSD', 0.0)
    except Exception as exc:                                      # noqa: BLE001
        print('[loop] could not read usage from this turn (%s)' % exc, flush=True)


def _save_resources(t0, start_iter):
    """Resource report is a required deliverable; write it continuously."""
    # Also the moment to republish the counters the dashboard shows: a tool
    # can hold the turn for an hour (ensemble_search does), and publishing
    # these only at the top of each loop turn froze the iteration count and
    # the clock for the whole of it.
    STATUS.update(iteration=STATE['iterations'], elapsed_s=time.time() - t0)
    path = os.path.join(RUNS_DIR, 'resources.json')
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({
            'iterations_used': STATE['iterations'],
            'iterations_this_session': STATE['iterations'] - start_iter,
            'llm_tokens_in': STATE['tokens_in'],
            'llm_tokens_out': STATE['tokens_out'],
            'llm_cost_usd': round(STATE['cost_usd'], 4),
            'agent_wall_clock_min': round((time.time() - t0) / 60, 1),
            'llm_cost_usd_by_model': STATE.get('per_model', {}),
            'gpu_hours': 0.0,
            'manual_interventions': STATE['interventions'],
            'blocked_writes_to_protected_files': STATE.get('denied_writes', 0),
            'device': 'CPU only (no GPU used)',
        }, fh, indent=2)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Record the stop, so the dashboard reports 'stopped' instead of leaving
        # its last frame claiming the run is still alive.
        STATUS.finish(ok=False, note='interrupted (Ctrl-C)')
        print('\ninterrupted; journal is on disk at runs/journal.jsonl')
        sys.exit(130)
