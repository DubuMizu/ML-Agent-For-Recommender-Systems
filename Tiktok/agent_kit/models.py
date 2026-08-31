"""Model slot of the harness. Every model maps X (B, F) int64 field-ids -> logit (B,).

The scoring task is within-user ranking, so only differences between scores of
the same user matter; any term constant within a user is free to drift.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REGISTRY = {}

_HIST_CACHE = {}


def _build_history(L):
    """Per-user long-view history (encoded video ids), most recent L, train only.

    Built from the TRAIN window only, so scoring valid/test uses strictly past
    behaviour -- the same information a deployed model would have.
    """
    if L in _HIST_CACHE:
        return _HIST_CACHE[L]
    from . import dataset as D
    # load_train_frame() cannot return validation or test rows at all, and it is
    # the only data accessor that still works while a model is being built (see
    # dataset.lock_eval_access). Model code must never call load_frames().
    # getattr keeps this working against a dataset module that was imported
    # before the accessor existed, i.e. a hot reload into a running process.
    _loader = getattr(D, 'load_train_frame', None)
    fr = _loader() if _loader is not None else D.load_frames()['train']
    enc = D.fit_base_encoder(fr)
    Xtr = enc.transform(fr).astype(np.int64)
    pos = np.flatnonzero(fr['y'] > 0)
    u = Xtr[pos, 0]
    v = Xtr[pos, 1]
    t = fr['time_ms'][pos]
    order = np.lexsort((t, u))                     # by user, then chronological
    u, v = u[order], v[order]
    n_u = int(enc.dims[0])
    cnt = np.bincount(u, minlength=n_u).astype(np.int64)
    off = np.concatenate(([0], np.cumsum(cnt)))
    H = np.zeros((n_u, L), dtype=np.int64)
    M = np.zeros((n_u, L), dtype=bool)
    for i in range(n_u):
        c = int(cnt[i])
        if c == 0:
            continue
        s = int(off[i]) + max(0, c - L)
        e = int(off[i + 1])
        m = e - s
        H[i, :m] = v[s:e]
        M[i, :m] = True
    _HIST_CACHE[L] = (H, M)
    return H, M


def register(name):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco


def build_model(cfg, total_dim, n_fields, seed=0):
    kind = cfg.get('type', 'fm')
    if kind not in REGISTRY:
        raise KeyError('unknown model type %r; known: %s' % (kind, sorted(REGISTRY)))
    kw = {k: v for k, v in cfg.items() if k != 'type'}
    return REGISTRY[kind](total_dim=total_dim, n_fields=n_fields, seed=seed, **kw)


@register('fm')
class FM(nn.Module):
    """Factorization Machine, matching baseline.py's parameterisation.

    logit = b + sum_i w_i + 0.5 * (||sum_i v_i||^2 - sum_i ||v_i||^2)

    Initialisation mirrors the reference: V ~ N(0, 0.01), W = 0, b = 0.
    """

    def __init__(self, total_dim, n_fields, k=16, seed=0, init_std=0.01):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        self.k, self.n_fields = k, n_fields
        V = torch.empty(total_dim, k).normal_(0.0, init_std, generator=g)
        self.V = nn.Parameter(V)
        self.W = nn.Parameter(torch.zeros(total_dim))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        E = F.embedding(X, self.V)                    # (B, F, k)
        S = E.sum(dim=1)                              # (B, k)
        inter = 0.5 * (S.pow(2).sum(1) - E.pow(2).sum((1, 2)))
        first = F.embedding(X, self.W.unsqueeze(1)).squeeze(-1).sum(1)
        return self.b + first + inter


@register('fm_dropout')
class FMDropout(FM):
    """FM with embedding dropout -- the reference FM overfits after ~7 epochs."""

    def __init__(self, total_dim, n_fields, k=16, seed=0, init_std=0.01, p=0.1):
        super().__init__(total_dim, n_fields, k=k, seed=seed, init_std=init_std)
        self.p = p

    def forward(self, X):
        E = F.embedding(X, self.V)
        E = F.dropout(E, p=self.p, training=self.training)
        S = E.sum(dim=1)
        inter = 0.5 * (S.pow(2).sum(1) - E.pow(2).sum((1, 2)))
        first = F.embedding(X, self.W.unsqueeze(1)).squeeze(-1).sum(1)
        return self.b + first + inter


@register('din')
class DINFM(FM):  # noqa: E501  (registered target-attention head)
    """FM plus a DIN-style target-attention head over the user's watch history.

    The user's train-window long-view history is attended with the candidate
    video's embedding as the query, giving an interest vector u; the head adds
    w . (u * e_target), i.e. a learned item-item affinity term that does not
    depend on the user embedding at all. The head weight starts at zero, so at
    initialisation the model IS the FM baseline.

    The target video is masked out of its own history, so a train row can never
    match itself -- without that, the head would learn a shortcut that does not
    exist at evaluation time.
    """

    def __init__(self, total_dim, n_fields, k=16, seed=0, init_std=0.01, L=32):
        super().__init__(total_dim, n_fields, k=k, seed=seed, init_std=init_std)
        H, M = _build_history(int(L))
        self.register_buffer('H', torch.from_numpy(H))
        self.register_buffer('M', torch.from_numpy(M))
        self.w = nn.Parameter(torch.zeros(k))
        self.scale = float(k) ** -0.5

    def forward(self, X):
        E = F.embedding(X, self.V)
        S = E.sum(dim=1)
        inter = 0.5 * (S.pow(2).sum(1) - E.pow(2).sum((1, 2)))
        first = F.embedding(X, self.W.unsqueeze(1)).squeeze(-1).sum(1)

        u, v = X[:, 0], X[:, 1]
        h = self.H[u]                                  # (B, L)
        m = self.M[u] & (h != v[:, None])              # drop pad and self-match
        Eh = F.embedding(h, self.V)                    # (B, L, k)
        q = E[:, 1, :]                                 # target video embedding
        s = (Eh * q[:, None, :]).sum(-1) * self.scale
        a = torch.softmax(s.masked_fill(~m, -1e9), dim=1)
        a = a * m.any(dim=1, keepdim=True).float()     # empty history -> no head
        u_int = (a[:, :, None] * Eh).sum(1)            # (B, k)
        head = (u_int * q) @ self.w

        return self.b + first + inter + head


@register('din_dropout')
class DINDropout(DINFM):
    """DIN head + embedding dropout, the regularisation that iteration 12 showed
    is the binding constraint. Dropout is applied to the candidate field
    embeddings and to the history embeddings, but NOT to the attention logits'
    query, so the attention pattern stays well defined while the values it
    averages are regularised."""

    def __init__(self, total_dim, n_fields, k=16, seed=0, init_std=0.01, L=8, p=0.3):
        super().__init__(total_dim, n_fields, k=k, seed=seed, init_std=init_std, L=L)
        self.p = p

    def forward(self, X):
        E = F.embedding(X, self.V)
        Ed = F.dropout(E, p=self.p, training=self.training)
        S = Ed.sum(dim=1)
        inter = 0.5 * (S.pow(2).sum(1) - Ed.pow(2).sum((1, 2)))
        first = F.embedding(X, self.W.unsqueeze(1)).squeeze(-1).sum(1)

        u, v = X[:, 0], X[:, 1]
        h = self.H[u]
        m = self.M[u] & (h != v[:, None])
        Eh = F.embedding(h, self.V)
        q = E[:, 1, :]
        s = (Eh * q[:, None, :]).sum(-1) * self.scale
        a = torch.softmax(s.masked_fill(~m, -1e9), dim=1)
        a = a * m.any(dim=1, keepdim=True).float()
        u_int = (a[:, :, None] * F.dropout(Eh, p=self.p, training=self.training)).sum(1)
        head = (u_int * q) @ self.w

        return self.b + first + inter + head


@register('deepfm')
class DeepFM(FM):
    """FM plus an MLP over the concatenated field embeddings (Guo et al., 2017)."""

    def __init__(self, total_dim, n_fields, k=16, seed=0, init_std=0.01,
                 hidden=(128, 64), p=0.2):
        super().__init__(total_dim, n_fields, k=k, seed=seed, init_std=init_std)
        torch.manual_seed(int(seed))
        layers, d = [], n_fields * k
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, X):
        E = F.embedding(X, self.V)
        S = E.sum(dim=1)
        inter = 0.5 * (S.pow(2).sum(1) - E.pow(2).sum((1, 2)))
        first = F.embedding(X, self.W.unsqueeze(1)).squeeze(-1).sum(1)
        deep = self.mlp(E.reshape(E.shape[0], -1)).squeeze(-1)
        return self.b + first + inter + deep


_AUX_CACHE = {}


def _build_video_aux(alpha=20.0):
    """Per-video behavioural priors from the TRAIN window only.

    Organizers' direction 4/5: the auxiliary engagement labels (is_click,
    is_like, ...) and watch time carry signal that the sparse long_view label
    does not. The batch sampler only carries field ids, so instead of a joint
    multi-task head we inject the same information item-side, as smoothed
    per-video rates. Empirical-Bayes smoothing toward the global rate with a
    pseudo-count keeps rare videos from getting extreme priors.

    NOTE this is NOT the '13 static CWM feature fields' the organizers already
    measured as unhelpful -- those are content attributes. These are realised
    behaviour, computed strictly on train, i.e. past information only.
    """
    if alpha in _AUX_CACHE:
        return _AUX_CACHE[alpha]
    from . import dataset as D
    fr = D.load_train_frame()
    enc = D.fit_base_encoder(fr)
    Xtr = enc.transform(fr).astype(np.int64)
    # field ids are GLOBAL offsets into the concatenated vocabulary, so field 1
    # must be shifted back by the width of field 0 to index a per-video table.
    off = int(enc.dims[0])
    n_v = int(enc.dims[1])
    v = Xtr[:, 1] - off
    cnt = np.bincount(v, minlength=n_v).astype(np.float64)

    dur = np.maximum(fr['duration_ms'].astype(np.float64), 1.0)
    ratio = np.clip(fr['play_time_ms'].astype(np.float64) / dur, 0.0, 3.0)
    cols = [fr[c].astype(np.float64) for c in D.AUX_LABELS]
    cols.append(ratio)                      # mean watch-time ratio
    cols.append((ratio >= 1.0).astype(np.float64))   # finish rate

    feats = []
    for c in cols:
        s = np.bincount(v, weights=c, minlength=n_v)
        prior = c.mean()
        feats.append((s + alpha * prior) / (cnt + alpha))
    feats.append(np.log1p(cnt))             # popularity / exposure
    A = np.stack(feats, axis=1)
    A = (A - A.mean(0)) / (A.std(0) + 1e-6)
    A = A.astype(np.float32)
    _AUX_CACHE[alpha] = (A, off)
    return A, off


@register('fm_aux')
class FMAux(FMDropout):
    """FM + per-video behavioural priors, entered globally and crossed with the
    user embedding.

    Within-user scoring cancels any user-constant term, so the prior vector is
    useful mainly through its cross with the user: g . a_v is a global taste
    prior, and e_u^T P a_v lets each user weight 'liked videos' vs 'finished
    videos' vs 'popular videos' differently.
    """

    def __init__(self, total_dim, n_fields, k=16, seed=0, init_std=0.01,
                 p=0.3, alpha=20.0, aux_scale=0.1):
        super().__init__(total_dim, n_fields, k=k, seed=seed, init_std=init_std, p=p)
        A, off = _build_video_aux(alpha)
        self.register_buffer('A', torch.from_numpy(A))
        self.v_off, self.n_v = off, A.shape[0]
        d = A.shape[1]
        g = torch.Generator().manual_seed(int(seed) + 7717)
        self.P = nn.Parameter(torch.empty(k, d).normal_(0.0, aux_scale, generator=g))
        self.g = nn.Parameter(torch.zeros(d))

    def forward(self, X):
        E = F.embedding(X, self.V)
        Ed = F.dropout(E, p=self.p, training=self.training)
        S = Ed.sum(dim=1)
        inter = 0.5 * (S.pow(2).sum(1) - Ed.pow(2).sum((1, 2)))
        first = F.embedding(X, self.W.unsqueeze(1)).squeeze(-1).sum(1)

        a = self.A[(X[:, 1] - self.v_off).clamp(0, self.n_v - 1)]                           # (B, d) video priors
        eu = Ed[:, 0, :]                              # (B, k) user embedding
        cross = ((eu @ self.P) * a).sum(1)            # personalised weighting
        glob = a @ self.g
        return self.b + first + inter + cross + glob


@register('dcnv2')
class DCNv2(FMDropout):
    """FM + CrossNetV2 head (Wang et al., WWW'21, arXiv:2008.13535).

    Full-rank cross layer:      x_{l+1} = x_0 * (W_l x_l + b_l) + x_l
    Low-rank variant (rank r):  x_{l+1} = x_0 * (U_l (V_l^T x_l) + b_l) + x_l
    with W_l in R^{d x d}, U_l, V_l in R^{d x r}, '*' the Hadamard product and
    x_0 = the concatenated field embeddings (d = n_fields * k).

    Unlike an MLP this is an explicit bounded-degree polynomial: depth L gives
    interactions of order L+1 with O(L d^2) (or O(2 L d r)) parameters, which is
    the right shape for a regime the journal has shown to be overfitting- not
    capacity-limited. The output projection is zero-initialised, so at step 0
    this model is exactly the FM baseline and the cross head has to earn its
    contribution.

    Knobs: k, p (embedding dropout), n_cross (depth), rank (0 = full rank),
    cross_p (dropout on x_0 inside the cross tower).
    """

    def __init__(self, total_dim, n_fields, k=16, seed=0, init_std=0.01,
                 p=0.3, n_cross=2, rank=0, cross_p=0.0):
        super().__init__(total_dim, n_fields, k=k, seed=seed, init_std=init_std, p=p)
        torch.manual_seed(int(seed) + 991)
        d = n_fields * k
        self.d, self.n_cross, self.rank = d, int(n_cross), int(rank)
        self.cross_p = float(cross_p)
        if self.rank > 0:
            self.U = nn.ParameterList()
            self.Vp = nn.ParameterList()
            for _ in range(self.n_cross):
                u = torch.empty(d, self.rank)
                v = torch.empty(d, self.rank)
                nn.init.xavier_uniform_(u)
                nn.init.xavier_uniform_(v)
                self.U.append(nn.Parameter(u))
                self.Vp.append(nn.Parameter(v))
        else:
            self.Wc = nn.ParameterList()
            for _ in range(self.n_cross):
                w = torch.empty(d, d)
                nn.init.xavier_uniform_(w)
                self.Wc.append(nn.Parameter(w))
        self.bc = nn.ParameterList(
            [nn.Parameter(torch.zeros(d)) for _ in range(self.n_cross)])
        self.wout = nn.Parameter(torch.zeros(d))     # zero init => starts as FM

    def _cross(self, x0):
        x = x0
        for l in range(self.n_cross):
            if self.rank > 0:
                t = (x @ self.Vp[l]) @ self.U[l].t()
            else:
                t = x @ self.Wc[l].t()
            x = x0 * (t + self.bc[l]) + x
        return x

    def forward(self, X):
        E = F.embedding(X, self.V)
        Ed = F.dropout(E, p=self.p, training=self.training)
        S = Ed.sum(dim=1)
        inter = 0.5 * (S.pow(2).sum(1) - Ed.pow(2).sum((1, 2)))
        first = F.embedding(X, self.W.unsqueeze(1)).squeeze(-1).sum(1)

        x0 = Ed.reshape(Ed.shape[0], -1)
        if self.cross_p > 0:
            x0 = F.dropout(x0, p=self.cross_p, training=self.training)
        cross = self._cross(x0) @ self.wout
        return self.b + first + inter + cross


@register('cin')
class CIN(FMDropout):
    """FM + Compressed Interaction Network head (xDeepFM, Lian et al., KDD'18).

    Per layer, with X^0 in R^{m x D} the field-embedding matrix and X^{k-1} in
    R^{H_{k-1} x D}:

        Z^k        = X^{k-1} (outer) X^0     -> (H_{k-1}, m, D) along each dim
        X^k_{h,*}  = sum_{i,j} W^{k,h}_{ij} * (X^{k-1}_{i,*} o X^0_{j,*})
        p^k_i      = sum_{j=1..D} X^k_{i,j}      (sum pooling over the emb. dim)
        p+         = [p^1, ..., p^T] -> linear -> logit

    i.e. VECTOR-wise crossing (Hadamard between whole embedding vectors, then a
    1x1 convolution that compresses H_{k-1}*m maps down to H_k), a different
    inductive bias from DeepFM's bit-wise MLP: degree grows exactly one per
    layer and the pooling makes each feature map a scalar. The output linear is
    zero-initialised so the model starts as the FM baseline.

    Knobs: k, p, cin_h (feature maps per layer), cin_depth (T).
    """

    def __init__(self, total_dim, n_fields, k=16, seed=0, init_std=0.01,
                 p=0.3, cin_h=16, cin_depth=2, direct=True):
        super().__init__(total_dim, n_fields, k=k, seed=seed, init_std=init_std, p=p)
        torch.manual_seed(int(seed) + 3313)
        self.m, self.depth, self.H = n_fields, int(cin_depth), int(cin_h)
        self.filters = nn.ParameterList()
        h_prev = n_fields
        total_pool = 0
        for _ in range(self.depth):
            w = torch.empty(self.H, h_prev * self.m)
            nn.init.xavier_uniform_(w)
            self.filters.append(nn.Parameter(w))
            total_pool += self.H
            h_prev = self.H
        self.wout = nn.Parameter(torch.zeros(total_pool))

    def forward(self, X):
        E = F.embedding(X, self.V)
        Ed = F.dropout(E, p=self.p, training=self.training)
        S = Ed.sum(dim=1)
        inter = 0.5 * (S.pow(2).sum(1) - Ed.pow(2).sum((1, 2)))
        first = F.embedding(X, self.W.unsqueeze(1)).squeeze(-1).sum(1)

        X0 = Ed                                        # (B, m, D)
        Xk = X0
        pools = []
        for l in range(self.depth):
            # Hadamard between every pair of maps, along the embedding dim
            Z = Xk.unsqueeze(2) * X0.unsqueeze(1)      # (B, H_{k-1}, m, D)
            B, hp, m, D = Z.shape
            Z = Z.reshape(B, hp * m, D)
            Xk = torch.einsum('hc,bcd->bhd', self.filters[l], Z)   # (B, H, D)
            pools.append(Xk.sum(dim=2))                # sum pooling over D
        cin = torch.cat(pools, dim=1) @ self.wout
        return self.b + first + inter + cin
