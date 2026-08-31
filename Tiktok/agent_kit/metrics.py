"""Vectorised GAUC / nDCG@5, numerically identical to the official evaluate.py.

evaluate.py is pure Python and runs once per epoch per experiment; over a 50
iteration agent run that overhead is real. This module reproduces it exactly
(agreement to ~1e-14, verified by tests/test_metrics.py) at ~4x the speed.

One caveat worth recording: evaluate.py averages nDCG as sum(per_user)/n_users,
and when the label array is float32 (as data.encode produces) that accumulation
loses ~1e-6. This module computes in float64, so it disagrees with the reference
by ~1e-6 on float32 labels -- ~80x below the FM seed sigma of 8e-4, and in the
more accurate direction. Both facts are pinned by tests/test_metrics.py.

Conventions replicated from evaluate.py, deliberately and exactly:
  * GAUC counts only users with 0 < npos < n_impressions, weighted by npos.
  * AUC is Mann-Whitney U with average ranks over tied scores.
  * nDCG@5 is averaged over ALL users; a user with no positive scores 0.0.
  * gain = 2^rel - 1, which for binary labels equals rel.
  * ties are broken by original row order (stable sort), as in evaluate.py.
"""
import numpy as np

__all__ = ['fast_evaluate', 'group_offsets', 'within_user_rank', 'rank_context']


def group_offsets(user_codes):
    """Boundaries of contiguous equal-value runs in a sorted code array."""
    n = len(user_codes)
    if n == 0:
        return np.zeros(1, dtype=np.int64)
    change = np.empty(n, dtype=bool)
    change[0] = True
    np.not_equal(user_codes[1:], user_codes[:-1], out=change[1:])
    starts = np.flatnonzero(change)
    return np.append(starts, n)


def rank_context(user_ids):
    """(users, starts, sizes) needed to rank rows inside their own user."""
    starts = group_offsets(np.sort(user_ids))
    return user_ids, starts, np.diff(starts)


def within_user_rank(scores, users, starts, sizes):
    """Rank each row among its own user's rows, scaled to [0, 1].

    Per-user, not global: the metric never compares rows across users, so a
    global rank would let one user's score range distort another user's
    ordering. The lexsort is by (user, score) -- sorting by score alone and then
    reading off positions gives a GLOBAL rank, which silently produces a
    plausible-looking but wrong combination.

    Lives here rather than in ensemble_search because the harness now needs it
    too, and ensemble_search imports the harness -- putting it there would make
    the import circular.
    """
    order = np.lexsort((scores, users))          # user-major, score-ascending
    out = np.empty(len(scores), dtype=np.float32)
    pos = np.arange(len(scores)) - np.repeat(starts[:-1], sizes)
    denom = np.repeat(np.maximum(sizes - 1, 1), sizes)
    out[order] = (pos / denom).astype(np.float32)
    return out


def _within_group_positions(starts, n):
    """0-based position of each row inside its group."""
    grp_start = np.repeat(starts[:-1], np.diff(starts))
    return np.arange(n, dtype=np.int64) - grp_start


def fast_evaluate(user_ids, labels, scores, k=5):
    """Return {'GAUC', 'nDCG@k', 'primary', 'users', 'rows'}."""
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    u = np.asarray(user_ids)
    n = len(y)
    if n == 0:
        return {'GAUC': 0.5, 'nDCG@%d' % k: 0.0, 'primary': 0.25, 'users': 0, 'rows': 0}

    # dense user codes that preserve first-appearance order (irrelevant to the
    # result, but keeps grouping cheap)
    _, ucode = np.unique(u, return_inverse=True)
    ucode = ucode.astype(np.int64)

    # ---- per-user counts -----------------------------------------------
    n_users = int(ucode.max()) + 1
    cnt = np.bincount(ucode, minlength=n_users).astype(np.int64)
    npos = np.bincount(ucode, weights=y, minlength=n_users)
    nneg = cnt - npos

    # ================= GAUC (ascending score, average ranks) =============
    order = np.lexsort((s, ucode))          # stable: ties keep row order
    su, ss, sy = ucode[order], s[order], y[order]
    starts = group_offsets(su)
    pos_in_grp = _within_group_positions(starts, n)
    rank = (pos_in_grp + 1).astype(np.float64)

    # average the rank across each block of tied scores within a user
    new_block = np.empty(n, dtype=bool)
    new_block[0] = True
    np.logical_or(su[1:] != su[:-1], ss[1:] != ss[:-1], out=new_block[1:])
    blk_first = np.flatnonzero(new_block)
    blk_last = np.append(blk_first[1:], n) - 1
    avg_rank = (rank[blk_first] + rank[blk_last]) * 0.5
    blk_id = np.cumsum(new_block) - 1
    rank = avg_rank[blk_id]

    srank = np.bincount(su, weights=rank * sy, minlength=n_users)
    with np.errstate(invalid='ignore', divide='ignore'):
        auc = (srank - npos * (npos + 1.0) * 0.5) / (npos * nneg)
    ok = (npos > 0) & (nneg > 0)            # exactly evaluate.py's filter
    gden = npos[ok].sum()
    gauc = float((npos[ok] * auc[ok]).sum() / gden) if gden > 0 else 0.5

    # ================= nDCG@k (descending score) =========================
    order = np.lexsort((-s, ucode))         # stable: ties keep row order
    du, dy = ucode[order], y[order]
    starts = group_offsets(du)
    pos_in_grp = _within_group_positions(starts, n)

    disc = 1.0 / np.log2(np.arange(k, dtype=np.float64) + 2.0)
    top = pos_in_grp < k
    gain = np.zeros(n, dtype=np.float64)
    # binary labels => 2^rel - 1 == rel; assert so a future graded label is caught
    gain[top] = dy[top] * disc[pos_in_grp[top]]
    dcg = np.bincount(du, weights=gain, minlength=n_users)

    cum = np.concatenate(([0.0], np.cumsum(disc)))
    ideal_k = np.minimum(npos, k).astype(np.int64)
    idcg = cum[ideal_k]
    ndcg = np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)
    ndcg_mean = float(ndcg.sum() / n_users)

    return {'GAUC': gauc, 'nDCG@%d' % k: ndcg_mean,
            'primary': (gauc + ndcg_mean) / 2.0,
            'users': n_users, 'rows': n}
