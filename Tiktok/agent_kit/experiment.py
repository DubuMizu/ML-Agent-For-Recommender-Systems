"""Robustness wrapper around run_experiment -- the agent's `run_experiment` tool.

The loop must survive a bad config without losing the run. Every failure mode
seen so far is turned into a structured result the agent can read and act on,
rather than an exception that ends the process:

  * an unknown model/loss name, or a bad hyper-parameter type -> 'failed' with
    the exception text and the list of valid names;
  * a divergent config (non-finite loss) -> 'failed' with the epoch and lr;
  * a config that is merely slow -> 'timeout', but with the best checkpoint so
    far preserved, so the iteration still yields a usable number.

Nothing here reads the hidden test split.
"""
import importlib
import os
import time
import traceback

from .harness import run_experiment
from . import models as _models
from . import losses as _losses

MODEL_REGISTRY = _models.REGISTRY
LOSS_REGISTRY = _losses.REGISTRY

# Snapshot mtimes at import, i.e. at the moment these modules were actually
# loaded. Populating this lazily on the first reload_plugins() call instead would
# record the post-edit mtime and skip the reload, so a class the agent wrote
# before the first experiment would be invisible to the registry -- which is
# exactly what happened on the first live run.
_MTIMES = {m.__file__: os.path.getmtime(m.__file__)
           for m in (_models, _losses)
           if getattr(m, '__file__', None) and os.path.exists(m.__file__)}


def reload_plugins():
    """Pick up agent edits to models.py / losses.py without restarting.

    Only these two modules are reloadable; the harness, the encoder and the
    metric are fixed for the whole run so that every iteration stays comparable.
    Returns the list of modules that actually changed.
    """
    global MODEL_REGISTRY, LOSS_REGISTRY
    changed = []
    for mod in (_models, _losses):
        path = getattr(mod, '__file__', None)
        if not path or not os.path.exists(path):
            continue
        mtime = os.path.getmtime(path)
        if _MTIMES.get(path) != mtime:
            importlib.reload(mod)
            changed.append(mod.__name__)
            _MTIMES[path] = mtime
    MODEL_REGISTRY = _models.REGISTRY
    LOSS_REGISTRY = _losses.REGISTRY
    return changed

# Per seed, not per experiment. The regularised configs that are working need
# ~10 epochs at ~50s/epoch, so a 600s-per-experiment budget starved them.
DEFAULT_TIME_BUDGET_S = 900


def _hint(exc):
    """Turn a raw exception into an actionable next step for the agent."""
    msg = str(exc)
    if isinstance(exc, KeyError) and 'unknown model' in msg:
        return 'choose a model from %s, or add one to agent_kit/models.py' % sorted(MODEL_REGISTRY)
    if isinstance(exc, KeyError) and 'unknown loss' in msg:
        return 'choose a loss from %s, or add one to agent_kit/losses.py' % sorted(LOSS_REGISTRY)
    if 'non-finite loss' in msg or 'diverged' in msg:
        return 'lower train.lr (try 1/3 of it) or raise train.l2; the run diverged'
    if isinstance(exc, TypeError) and 'unexpected keyword' in msg:
        return 'a config key is not accepted by that model/loss; check its signature'
    if isinstance(exc, MemoryError) or 'out of memory' in msg.lower():
        return 'reduce train.batch or train.group_size'
    return 'inspect the traceback; fix the config or the module it points at'


def safe_run(cfg, seeds=(0, 1, 2), time_budget_s=DEFAULT_TIME_BUDGET_S, log=None,
             reporter=None, hypothesis=None, iteration=None, prune_fn=None,
             keep_models=False):
    """Run an experiment, converting any failure into a structured result.

    Returns {'status': 'ok'|'failed'|'timeout', ...metrics..., 'error', 'hint'}.
    """
    t0 = time.time()
    try:
        reloaded = reload_plugins()
    except Exception as exc:                                      # noqa: BLE001
        return {'status': 'failed', 'config': cfg,
                'error': 'reloading your edited module failed -- %s: %s'
                         % (type(exc).__name__, exc),
                'traceback': traceback.format_exc(limit=8),
                'hint': 'there is a syntax or import error in the file you just '
                        'edited; read the traceback and fix it',
                'wall_clock_s': round(time.time() - t0, 1)}
    if reporter is not None:
        reporter.start(iteration if iteration is not None else '?', cfg,
                       hypothesis=hypothesis, seeds=seeds, budget=time_budget_s)
        if log is None:
            log = reporter.epoch
    try:
        res = run_experiment(cfg, seeds=seeds, time_budget_s=time_budget_s, log=log,
                             reporter=reporter, prune_fn=prune_fn,
                             keep_models=keep_models)
    except Exception as exc:                                  # noqa: BLE001
        failed = {
            'status': 'failed',
            'config': cfg,
            'error': '%s: %s' % (type(exc).__name__, exc),
            'traceback': traceback.format_exc(limit=8),
            'hint': _hint(exc),
            'wall_clock_s': round(time.time() - t0, 1),
        }
        if reporter is not None:
            reporter.finish(failed)
        return failed
    res['reloaded_modules'] = reloaded
    stopped = any(r.get('stopped_early') for r in res['runs'])
    res['status'] = 'timeout' if stopped else 'ok'
    if stopped:
        res['hint'] = (
            'at least one seed hit the %ds PER-SEED budget and returned its best '
            'checkpoint so far, so this number is an under-trained LOWER BOUND, '
            'not a verdict on the config. epochs_run=%s secs_per_epoch=%s. If the '
            'epoch curve was still improving, re-run with a larger time_budget_s '
            'before concluding anything.'
            % (time_budget_s, res.get('epochs_run'), res.get('secs_per_epoch')))
    # keep the payload small: per-epoch history stays on disk, not in the agent's context
    res['epoch_curves'] = [[e['valid']['primary'] for e in r['history']] for r in res['runs']]
    res.pop('runs', None)
    return res


def summarise(res):
    """One-line human/agent readable summary of a result."""
    if res.get('status') != 'ok' and res.get('error'):
        return 'FAILED %s | hint: %s' % (res['error'][:160], res.get('hint'))
    return ('valid primary %.4f +-%.4f (GAUC %.4f, nDCG@5 %.4f) | unbiased %.4f | '
            'best epochs %s | %.0fs%s' % (
                res['valid_primary'], res.get('valid_primary_std', 0.0),
                res['valid_GAUC'], res['valid_nDCG@5'], res['unbiased_primary'],
                res.get('best_epochs'), res['wall_clock_s'],
                ' [TIMEOUT]' if res.get('status') == 'timeout' else ''))
