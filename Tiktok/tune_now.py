"""Direct Bayesian search over the regularisation family -- zero LLM tokens."""
import sys
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
from agent_kit.tuner import tune, summarise_tuning
from agent_kit.journal import Journal
from agent_kit.progress import Reporter

BASE = {'model': {'type': 'fm_dropout', 'k': 16, 'p': 0.4},
        'loss': {'type': 'softmax'},
        'train': {'lr': 0.001, 'batch': 8192, 'epochs': 40, 'patience': 4,
                  'l2': 1e-4, 'group_size': 8}}
SPACE = {'train.lr':        ('float', 2e-4, 3e-3, True),
         'model.p':         ('float', 0.20, 0.75, False),
         'train.l2':        ('float', 1e-5, 3e-3, True),
         'train.group_size': ('int', 4, 16),
         'model.k':         ('int', 8, 32)}

out = tune(BASE, SPACE, n_init=5, n_iter=9, seeds=(0, 1), time_budget_s=600,
           journal=Journal(), reporter=Reporter(),
           hypothesis='BO over the regularisation family: regularisation is the only '
                      'lever that has produced a real gain, so map that surface properly.')
print('\n' + '='*72); print(summarise_tuning(out))
