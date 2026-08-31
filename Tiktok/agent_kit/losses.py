"""Loss slot of the harness.

The metric is within-user ranking (GAUC + nDCG@5), but the reference FM trains
with pointwise log-loss. A pointwise objective spends capacity on calibrating
absolute probabilities, which the metric never looks at: any term constant
within a user cancels out of that user's ordering. These objectives instead
put the gradient on within-user *comparisons*, which is what is actually scored.

Each objective declares the batch shape it needs, and the sampler in
harness.py builds it:

  'point' -> {'X': (B, F),        'y': (B,)}
  'pair'  -> {'Xp': (B, F), 'Xn': (B, F)}          positive / negative, same user
  'list'  -> {'Xg': (B, G, F)}    column 0 is the positive, 1..G-1 negatives
"""
import math
import torch
import torch.nn.functional as F

REGISTRY = {}


def register(name, batch_kind):
    def deco(fn):
        fn.batch_kind = batch_kind
        REGISTRY[name] = fn
        return fn
    return deco


def build_loss(cfg):
    kind = cfg.get('type', 'bce')
    if kind not in REGISTRY:
        raise KeyError('unknown loss type %r; known: %s' % (kind, sorted(REGISTRY)))
    fn = REGISTRY[kind]
    kw = {k: v for k, v in cfg.items() if k != 'type'}

    def bound(model, batch):
        return fn(model, batch, **kw)
    bound.batch_kind = fn.batch_kind
    bound.name = kind
    bound.kwargs = kw
    return bound


@register('bce', 'point')
def bce(model, batch):
    """Pointwise log-loss -- the reference FM objective, kept as the control."""
    return F.binary_cross_entropy_with_logits(model(batch['X']), batch['y'])


@register('bpr', 'pair')
def bpr(model, batch):
    """Bayesian Personalised Ranking (Rendle et al., 2009).

    -log sigma(s_pos - s_neg) over positive/negative pairs drawn from the *same*
    user, so the gradient only ever moves a within-user comparison.
    """
    return -F.logsigmoid(model(batch['Xp']) - model(batch['Xn'])).mean()


@register('softmax', 'list')
def softmax(model, batch, temperature=1.0):
    """Listwise softmax cross-entropy (sampled InfoNCE over one user's items).

    Generalises BPR from one negative to G-1 negatives; the extra negatives
    sharpen the within-user ordering signal per step.
    """
    Xg = batch['Xg']
    B, G, nF = Xg.shape
    s = model(Xg.reshape(B * G, nF)).reshape(B, G) / temperature
    target = torch.zeros(B, dtype=torch.long)          # column 0 is the positive
    return F.cross_entropy(s, target)


@register('lambdarank', 'list')
def lambdarank(model, batch, k=5, sigma=1.0):
    """LambdaRank with nDCG@k gains (Burges et al.), single-positive groups.

    Each sampled group holds one positive and G-1 negatives, so the group's
    ideal DCG is 1 and its nDCG is 1/log2(1+rank_pos). The pairwise logistic
    loss is weighted by |delta nDCG@k| from swapping the positive with each
    negative -- putting the gradient where it moves the metric being scored,
    and truncating at k exactly as nDCG@5 does.
    """
    Xg = batch['Xg']
    B, G, nF = Xg.shape
    s = model(Xg.reshape(B * G, nF)).reshape(B, G)
    s_pos, s_neg = s[:, :1], s[:, 1:]                  # (B,1), (B,G-1)

    with torch.no_grad():
        # rank of each item within its group under the current scores (1-based)
        rank = torch.argsort(torch.argsort(s, dim=1, descending=True), dim=1) + 1
        r_pos, r_neg = rank[:, :1].double(), rank[:, 1:].double()
        disc = lambda r: torch.where(r <= k, 1.0 / torch.log2(1.0 + r),
                                     torch.zeros_like(r))
        delta = (disc(r_pos) - disc(r_neg)).abs().float()   # IDCG == 1

    return (delta * F.softplus(-sigma * (s_pos - s_neg))).sum(1).mean()
