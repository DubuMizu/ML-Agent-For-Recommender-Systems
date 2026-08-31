"""ASHA + Bayesian optimisation + ensemble selection, in one pass. Zero LLM tokens."""
import sys
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
import json
from agent_kit import ensemble_search as ES
from agent_kit.journal import Journal
from agent_kit.progress import Reporter

TRAIN = {'lr': 0.001, 'batch': 8192, 'epochs': 40, 'patience': 4,
         'l2': 1e-4, 'group_size': 8}
KNOBS = {'train.lr': ('float', 3e-4, 2.5e-3, True),
         'model.p':  ('float', 0.20, 0.70, False),
         'train.l2': ('float', 1e-5, 2e-3, True)}

# Four STRUCTURAL families. Diversity has to be designed in: BO tunes continuous
# knobs well but cannot cheaply discover that a different loss decorrelates errors.
FAMILIES = {
 'fm_softmax':  ({'model': {'type': 'fm_dropout', 'k': 16, 'p': 0.4},
                  'loss': {'type': 'softmax'}, 'train': dict(TRAIN)},
                 {**KNOBS, 'train.group_size': ('int', 4, 12)}),
 'fm_bpr':      ({'model': {'type': 'fm_dropout', 'k': 16, 'p': 0.4},
                  'loss': {'type': 'bpr'}, 'train': dict(TRAIN)}, dict(KNOBS)),
 'fm_lambda':   ({'model': {'type': 'fm_dropout', 'k': 16, 'p': 0.4},
                  'loss': {'type': 'lambdarank', 'k': 5}, 'train': dict(TRAIN)},
                 {**KNOBS, 'train.group_size': ('int', 4, 12)}),
 'din_softmax': ({'model': {'type': 'din_dropout', 'k': 16, 'L': 8, 'p': 0.3},
                  'loss': {'type': 'softmax'}, 'train': dict(TRAIN)},
                 {**KNOBS, 'model.L': ('int', 4, 24)}),
}

out = ES.run(FAMILIES, n_init=2, n_iter=2, seeds=(0, 1), time_budget_s=400,
             journal=Journal(), reporter=Reporter(), max_ensemble=12)
print('\n' + '=' * 72); print(ES.summarise(out))
lib = out.pop('library'); w = out.pop('weights')
json.dump({**out, 'member_configs': [
    {'key': e['key'], 'weight': float(wi), 'config': e['config']}
    for wi, e in zip(w, lib.entries) if wi > 0]},
    open('runs/ensemble_search.json', 'w'), indent=2)
print('\nwrote runs/ensemble_search.json')
