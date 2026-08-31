"""The `tune` tool: many evaluations per agent turn, at zero extra LLM cost.

Hyper-parameter search is the part of the loop the LLM adds least to -- it is
numeric optimisation over a fixed structure, not reasoning. Spending one agent
turn (and its tokens) per dropout value is the single most wasteful thing the
loop can do. This module runs a whole search locally and returns one summary,
so the agent spends its turns on ideas and its tools on arithmetic.

Combines the two mechanisms in search.py:
  * Bayesian optimisation picks WHICH config to try, using a GP whose
    observation-noise level is fixed to the measured seed noise;
  * ASHA stops trials that are clearly out of contention, with a late first rung
    so slow-starting configs are not killed before they overtake.
"""
import copy
import time

import numpy as np

from .experiment import safe_run
from .search import AshaPruner, Space, suggest, OBSERVED_NOISE_SD


def _apply(base, params):
    """Overlay flat tuned params onto a nested experiment config."""
    cfg = copy.deepcopy(base)
    for key, val in params.items():
        section, _, name = key.partition('.')
        cfg.setdefault(section, {})[name] = val
    return cfg


def tune(base_cfg, space_spec, n_init=4, n_iter=10, seeds=(0, 1),
         time_budget_s=900, journal=None, reporter=None, rng_seed=0,
         hypothesis=None, prune=True):
    """Bayesian search over `space_spec`, layered on `base_cfg`.

    Returns {'best_params', 'best_config', 'best_score', 'trials', ...}.

    Trials use fewer seeds than a confirmation run (default 2): the search only
    needs to rank candidates, and the winner is re-run properly afterwards. That
    is also why the reported best_score is treated as provisional.
    """
    space = Space(space_spec)
    rng = np.random.default_rng(rng_seed)
    pruner = AshaPruner() if prune else None

    X, y, trials = [], [], []
    t0 = time.time()

    for i in range(n_init + n_iter):
        if i < n_init:
            u = space.sample(rng, 1)[0]             # random warm-up
            how = 'init'
        else:
            u, _gp = suggest(space, np.array(X), np.array(y), rng)
            how = 'bayes'
        params = space.from_unit(u)
        # Re-encode: int and categorical dimensions quantise, so the point we
        # actually evaluate is not the point EI proposed. Feeding the GP the
        # proposal rather than the evaluated config would make it model a
        # configuration that was never run.
        u = space.to_unit(params)
        cfg = _apply(base_cfg, params)

        res = safe_run(cfg, seeds=seeds, time_budget_s=time_budget_s,
                       reporter=reporter, iteration='tune %d/%d' % (i + 1, n_init + n_iter),
                       hypothesis=hypothesis,
                       prune_fn=pruner.make_callback() if pruner else None)
        score = res.get('valid_primary')
        if score is None:                           # failed trial
            trials.append({'params': params, 'status': res['status'],
                           'error': res.get('error'), 'how': how})
            continue

        X.append(u)
        y.append(score)
        trials.append({'params': params, 'score': round(score, 6), 'how': how,
                       'status': res['status'],
                       'epochs_run': res.get('epochs_run'),
                       'unbiased': res.get('unbiased_primary'),
                       'secs': res.get('wall_clock_s')})
        if journal is not None:
            journal.record(
                hypothesis='[tune %d/%d, %s] %s' % (i + 1, n_init + n_iter, how,
                                                    hypothesis or 'hyper-parameter search'),
                rationale='Bayesian optimisation over %s; GP noise fixed at the '
                          'measured seed SD (%.4f).' % (list(space_spec), OBSERVED_NOISE_SD),
                config=cfg, result=res, status=res['status'],
                error=res.get('error'), recovery=res.get('hint'), tags=['tune', how])

    if not y:
        return {'error': 'every trial failed', 'trials': trials}

    b = int(np.argmax(y))
    return {
        'best_params': space.from_unit(X[b]),
        'best_config': _apply(base_cfg, space.from_unit(X[b])),
        'best_score': round(float(y[b]), 6),
        'n_trials': len(trials),
        'n_scored': len(y),
        'pruned': pruner.pruned if pruner else 0,
        'wall_clock_s': round(time.time() - t0, 1),
        'trials': trials,
        'note': ('best_score comes from a %d-seed search trial and is provisional; '
                 're-run the winner at full seeds before trusting it. The search '
                 'noise floor is ~%.4f, so any spread below that is not real.'
                 % (len(seeds), OBSERVED_NOISE_SD)),
    }


def summarise_tuning(out):
    if 'error' in out:
        return 'tuning failed: %s' % out['error']
    lines = ['tuned %d trials (%d scored, %d pruned) in %.0fs'
             % (out['n_trials'], out['n_scored'], out['pruned'], out['wall_clock_s']),
             'best %.4f with %s' % (out['best_score'],
                                    {k: (round(v, 5) if isinstance(v, float) else v)
                                     for k, v in out['best_params'].items()})]
    ranked = sorted([t for t in out['trials'] if 'score' in t],
                    key=lambda t: -t['score'])[:8]
    lines.append('top trials:')
    for t in ranked:
        lines.append('  %.4f  %s  (%s)' % (
            t['score'],
            {k: (round(v, 5) if isinstance(v, float) else v) for k, v in t['params'].items()},
            t['how']))
    lines.append(out['note'])
    return '\n'.join(lines)
