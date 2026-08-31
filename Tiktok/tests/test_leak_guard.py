"""The eval-access lock must make label leakage fail loudly, not score well.

Models are the one component the agent writes freely. A model that peeked at
frames['valid']['y'] would post an excellent validation number and be worthless;
nothing else in the pipeline would notice. These tests pin the guarantee.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn

from agent_kit import dataset as D
from agent_kit import models as M
from agent_kit.harness import prepare, train_once

ok = True

def check(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    print('  %-58s %s %s' % (name, 'PASS' if cond else 'FAIL', detail))

print('1. accessor behaviour under the lock')
D.unlock_eval_access()
check('load_frames() works when unlocked', set(D.load_frames().keys()) == {'train','valid','test'})
D.lock_eval_access('unit test')
try:
    D.load_frames(); leaked = True; err = ''
except D.EvalAccessError as e:
    leaked = False; err = str(e)[:60]
check('load_frames() BLOCKED when locked', not leaked, err)
tf = D.load_train_frame()
check('load_train_frame() still works when locked', 'y' in tf and len(tf['y']) == 1141112)
check('load_train_frame() cannot expose eval rows', len(tf['y']) == 1141112)
D.unlock_eval_access()

print('\n2. a deliberately cheating model is caught')
@M.register('__cheater__')
class Cheater(M.FM):
    """Reads the validation labels. This is the exact failure being guarded."""
    def __init__(self, total_dim, n_fields, k=16, seed=0, **kw):
        super().__init__(total_dim, n_fields, k=k, seed=seed)
        frames = D.load_frames()               # <-- the leak
        self.stolen = frames['valid']['y']

prep = prepare()
try:
    train_once({'model': {'type': '__cheater__'}, 'loss': {'type': 'bce'},
                'train': {'epochs': 1}}, prep, 0)
    caught, msg = False, 'model trained successfully -- LEAK NOT CAUGHT'
except D.EvalAccessError as e:
    caught, msg = True, 'raised EvalAccessError'
except Exception as e:
    caught, msg = False, '%s: %s' % (type(e).__name__, e)
check('cheating model is refused during training', caught, msg)
check('lock released after the failure', not D.eval_access_locked())
M.REGISTRY.pop('__cheater__', None)

print('\n3. honest models are unaffected')
r = train_once({'model': {'type': 'fm', 'k': 8}, 'loss': {'type': 'bce'},
                'train': {'epochs': 1}}, prep, 0)
check('plain FM still trains under the lock', r['valid']['primary'] > 0.55,
      'valid %.4f' % r['valid']['primary'])
if 'din' in M.REGISTRY:
    r2 = train_once({'model': {'type': 'din', 'k': 8, 'L': 8}, 'loss': {'type': 'bpr'},
                     'train': {'epochs': 1, 'batch': 8192}}, prep, 0)
    check('DIN (uses train history) still trains', r2['valid']['primary'] > 0.55,
          'valid %.4f' % r2['valid']['primary'])
check('lock released after normal training', not D.eval_access_locked())

print('\nALL PASS' if ok else '\nFAILURES PRESENT')
sys.exit(0 if ok else 1)
