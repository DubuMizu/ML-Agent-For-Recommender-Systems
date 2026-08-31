"""Does ensembling push the score past the 0.6037 plateau?

Three configs now sit within 0.0002 of each other -- statistically identical.
That looks like a dead end, but it is actually the ideal setup for an ensemble:
fm_dropout(p=0.3), fm_dropout(p=0.5) and din_dropout are structurally different
models that happen to score the same, so their errors should be partly
decorrelated. Averaging decorrelated rankings is the cheapest reliable gain in
ranking systems, and we already pay to train multiple seeds anyway.

This reframes the DIN negative result: DIN is not a better model, but it may
still be a useful ensemble member precisely because it is a different one.

Combination is by RANK, not raw score: the metric only reads ordering, and
different models put their logits on different scales, so averaging raw scores
would let whichever model has the widest spread dominate. Ranks are computed
WITHIN each user, because that is the only comparison the metric ever makes.
"""
import sys
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import itertools
import time

import numpy as np

from agent_kit.harness import prepare, train_once, predict
from agent_kit.metrics import fast_evaluate, group_offsets
from agent_kit.journal import FM_VALID_PRIMARY

CONFIGS = {
    'fmdrop_p3': {'model': {'type': 'fm_dropout', 'k': 16, 'p': 0.3},
                  'loss': {'type': 'softmax'},
                  'train': {'lr': 0.001, 'batch': 4096, 'epochs': 40, 'patience': 4,
                            'l2': 1e-4, 'group_size': 8}},
    'fmdrop_p5': {'model': {'type': 'fm_dropout', 'k': 16, 'p': 0.5},
                  'loss': {'type': 'softmax'},
                  'train': {'lr': 0.001, 'batch': 8192, 'epochs': 40, 'patience': 4,
                            'l2': 1e-4, 'group_size': 8}},
    'din_p3':    {'model': {'type': 'din_dropout', 'k': 16, 'L': 8, 'p': 0.3},
                  'loss': {'type': 'softmax'},
                  'train': {'lr': 0.001, 'batch': 4096, 'epochs': 40, 'patience': 4,
                            'l2': 1e-4, 'group_size': 8}},
}
SEEDS = (0, 1, 2)


def within_user_rank(scores, users):
    """Rank each row among that user's rows, scaled to [0, 1].

    Per-user rather than global: the metric never compares across users, and a
    global rank would let one user's score range distort another's ordering.
    """
    order = np.lexsort((scores, users))
    starts = group_offsets(users[order])
    sizes = np.diff(starts)
    pos = np.arange(len(scores)) - np.repeat(starts[:-1], sizes)
    denom = np.repeat(np.maximum(sizes - 1, 1), sizes)
    out = np.empty(len(scores), dtype=np.float64)
    out[order] = pos / denom
    return out


def main():
    prep = prepare()
    u_va, y_va = prep.users['valid'], prep.y['valid']
    u_ub, y_ub = prep.users['unbiased'], prep.y['unbiased']
    # users must be sorted-groupable for the rank helper
    order_va = np.argsort(u_va, kind='stable')
    inv_va = np.empty_like(order_va)
    inv_va[order_va] = np.arange(len(order_va))

    preds, singles = {}, {}
    t0 = time.time()
    for name, cfg in CONFIGS.items():
        for s in SEEDS:
            r = train_once(cfg, prep, s)
            key = '%s:s%d' % (name, s)
            preds[key] = {'valid': predict(r['model'], prep.T['valid']),
                          'unbiased': predict(r['model'], prep.T['unbiased'])}
            singles[key] = r['valid']['primary']
            print('  %-16s valid %.4f  (best ep %d, %d epochs)'
                  % (key, r['valid']['primary'], r['best_epoch'], len(r['history'])),
                  flush=True)
    print('training took %.0f min\n' % ((time.time() - t0) / 60))

    def combo(keys, split='valid'):
        u = u_va if split == 'valid' else u_ub
        acc = np.zeros(len(u))
        for k in keys:
            acc += within_user_rank(preds[k][split].astype(np.float64), u)
        return acc / len(keys)

    def score(keys, split='valid'):
        u, y = (u_va, y_va) if split == 'valid' else (u_ub, y_ub)
        return fast_evaluate(u, y, combo(keys, split))['primary']

    print('%-42s %8s %8s %9s' % ('combination', 'valid', 'unbiased', 'vs FM'))
    print('-' * 72)
    base = np.mean(list(singles.values()))
    print('%-42s %8.4f %8s %+9.4f' % ('single model (mean of 9)', base, '-', base - FM_VALID_PRIMARY))

    rows = []
    for name in CONFIGS:
        keys = ['%s:s%d' % (name, s) for s in SEEDS]
        v = score(keys)
        rows.append(('%s x3 seeds' % name, v, score(keys, 'unbiased')))
    # cross-config: one seed each, then everything
    for combi in itertools.combinations(CONFIGS, 2):
        keys = ['%s:s%d' % (n, s) for n in combi for s in SEEDS]
        rows.append(('+'.join(combi) + ' x3', score(keys), score(keys, 'unbiased')))
    allk = list(preds)
    rows.append(('ALL 3 configs x 3 seeds (9 models)', score(allk), score(allk, 'unbiased')))

    for label, v, ub in rows:
        star = '  <-- best' if v == max(r[1] for r in rows) else ''
        print('%-42s %8.4f %8.4f %+9.4f%s' % (label, v, ub, v - FM_VALID_PRIMARY, star))

    best_label, best_v, _ = max(rows, key=lambda r: r[1])
    print('\nbest single model : %.4f' % max(singles.values()))
    print('best ensemble     : %.4f  (%s)' % (best_v, best_label))
    print('ensemble gain     : %+.4f over the best single model'
          % (best_v - max(singles.values())))
    print('total vs FM       : %+.4f' % (best_v - FM_VALID_PRIMARY))


if __name__ == '__main__':
    main()
