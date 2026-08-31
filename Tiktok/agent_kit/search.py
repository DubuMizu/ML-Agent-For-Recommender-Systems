"""Bayesian optimisation + ASHA pruning, sized for this benchmark's noise.

Two facts from the run so far drive every design choice here.

1. The objective is NOISY relative to the effects being chased. The SD of a
   paired per-seed difference is ~0.0012, so a 3-seed comparison has SE ~0.0007
   while real gains are ~0.001-0.002. A textbook GP-EI loop with a
   near-zero noise term would spend its whole budget chasing seed variance. The
   GP here therefore takes a FIXED observation-noise variance measured from the
   journal, which is the one thing we genuinely know about this objective.

2. Early epochs do not rank configs correctly. The winning config (dropout
   p=0.5) is BEHIND p=0.3 for its first four epochs and only overtakes at epoch
   five; a rung at epoch 1-3 prunes the eventual best model. ASHA here refuses
   to prune before `min_rung_epoch`, and prunes on best-so-far rather than the
   current epoch, because these curves are non-monotonic.

Numpy only -- no scipy, no sklearn.
"""
import math

import numpy as np

# Measured from runs/journal.jsonl: SD of the paired per-seed difference between
# configs. Recompute with tools/measure_noise.py if the harness changes.
OBSERVED_NOISE_SD = 0.0012


# --------------------------------------------------------------- the space ---
class Space:
    """Parameter space that maps to and from the unit hypercube.

    spec: {name: ('float', lo, hi, log_bool) | ('int', lo, hi) | ('cat', [vals])}
    """

    def __init__(self, spec):
        self.spec = dict(spec)
        self.names = list(self.spec)

    @property
    def dim(self):
        return len(self.names)

    def to_unit(self, cfg):
        u = []
        for n in self.names:
            s = self.spec[n]
            v = cfg[n]
            if s[0] == 'float':
                lo, hi, log = s[1], s[2], (len(s) > 3 and s[3])
                if log:
                    u.append((math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)))
                else:
                    u.append((v - lo) / (hi - lo))
            elif s[0] == 'int':
                u.append((v - s[1]) / max(1e-9, s[2] - s[1]))
            else:
                u.append(s[1].index(v) / max(1, len(s[1]) - 1))
        return np.clip(np.array(u, dtype=float), 0.0, 1.0)

    def from_unit(self, u):
        cfg = {}
        for i, n in enumerate(self.names):
            s = self.spec[n]
            x = float(np.clip(u[i], 0.0, 1.0))
            if s[0] == 'float':
                lo, hi, log = s[1], s[2], (len(s) > 3 and s[3])
                cfg[n] = (math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo)))
                          if log else lo + x * (hi - lo))
            elif s[0] == 'int':
                cfg[n] = int(round(s[1] + x * (s[2] - s[1])))
            else:
                cfg[n] = s[1][int(round(x * (len(s[1]) - 1)))]
        return cfg

    def sample(self, rng, n):
        return rng.random((n, self.dim))


# ------------------------------------------------------------------- the GP ---
def _norm_cdf(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _norm_pdf(z):
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


class GP:
    """Isotropic RBF Gaussian process with a FIXED observation noise level.

    The noise term is not fitted. Fitting it on ~15 observations of a genuinely
    noisy objective reliably underestimates it, which makes the surrogate
    over-confident exactly where it should hedge.
    """

    def __init__(self, noise_sd=OBSERVED_NOISE_SD):
        self.noise_sd = noise_sd
        self._fitted = False

    def _k(self, A, B, ell):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / (ell * ell))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.X = X
        self.y_mean, self.y_std = y.mean(), max(y.std(), 1e-9)
        yz = (y - self.y_mean) / self.y_std
        noise = max((self.noise_sd / self.y_std) ** 2, 1e-6)

        best = (-np.inf, 0.3, None)
        for ell in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0):
            K = self._k(X, X, ell) + noise * np.eye(len(X))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                continue
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, yz))
            lml = (-0.5 * yz @ alpha - np.log(np.diag(L)).sum()
                   - 0.5 * len(X) * math.log(2 * math.pi))
            if lml > best[0]:
                best = (lml, ell, (L, alpha))
        self.ell, (self.L, self.alpha) = best[1], best[2]
        self.noise = noise
        self._fitted = True
        return self

    def predict(self, Xs):
        Ks = self._k(self.X, np.asarray(Xs, dtype=float), self.ell)
        mu = Ks.T @ self.alpha
        v = np.linalg.solve(self.L, Ks)
        var = np.maximum(1.0 + self.noise - (v * v).sum(0), 1e-12)
        return mu, np.sqrt(var)                     # standardised units


def expected_improvement(mu, sd, best_z, xi=0.01):
    z = (mu - best_z - xi) / sd
    return (mu - best_z - xi) * _norm_cdf(z) + sd * _norm_pdf(z)


def suggest(space, X, y, rng, n_candidates=4096, xi=0.01,
            noise_sd=OBSERVED_NOISE_SD):
    """Next point to evaluate, by maximising expected improvement."""
    gp = GP(noise_sd).fit(X, y)
    cand = space.sample(rng, n_candidates)
    # mix in local perturbations of the incumbent: EI over a purely random
    # candidate set explores well but refines poorly in a flat, noisy landscape
    inc = np.asarray(X)[int(np.argmax(y))]
    local = np.clip(inc + rng.normal(0, 0.08, (n_candidates // 4, space.dim)), 0, 1)
    cand = np.vstack([cand, local])
    mu, sd = gp.predict(cand)
    best_z = (np.max(y) - gp.y_mean) / gp.y_std
    ei = expected_improvement(mu, sd, best_z, xi)
    return cand[int(np.argmax(ei))], gp


# ----------------------------------------------------------------- pruning ---
class AshaPruner:
    """Successive-halving pruning with a late first rung.

    `min_rung_epoch` defaults to 5 because on this benchmark the best config is
    behind for its first four epochs -- pruning earlier removes the winner. See
    the module docstring.
    """

    def __init__(self, rungs=(5, 9, 14), eta=3, min_trials=4, min_rung_epoch=5):
        self.rungs = tuple(r for r in rungs if r >= min_rung_epoch)
        self.eta = eta
        self.min_trials = min_trials
        self.history = {r: [] for r in self.rungs}
        self.pruned = 0

    def make_callback(self):
        """Returns prune_fn(epoch, best_so_far) -> bool for one trial."""
        seen = set()

        def prune_fn(epoch, best_so_far):
            if epoch not in self.rungs or epoch in seen:
                return False
            seen.add(epoch)
            hist = self.history[epoch]
            hist.append(best_so_far)
            if len(hist) <= self.min_trials:
                return False                        # not enough evidence yet
            # prune if outside the top 1/eta of everything seen at this rung
            cutoff = np.quantile(hist, 1.0 - 1.0 / self.eta)
            if best_so_far < cutoff - OBSERVED_NOISE_SD:
                self.pruned += 1
                return True
            return False

        return prune_fn
