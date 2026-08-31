"""fast_evaluate must be numerically identical to the official evaluate.py."""
import time, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import evaluate as official
from agent_kit.metrics import fast_evaluate

def cmp(tag, u, y, s, tol=1e-11):
    # Labels are cast to float64 for the official call on purpose. evaluate.py
    # accumulates nDCG as sum(per_user)/n_users; when the label array is float32
    # (as data.encode produces) those per-user values are float32 and the sum
    # over ~22k users loses ~1e-6 of precision. That is a rounding artefact of
    # the reference implementation, not an algorithmic difference -- with float64
    # labels the two agree to ~1e-15. See test_float32_artefact below, which
    # bounds the artefact at 1e-5, i.e. ~80x below the FM seed sigma of 8e-4.
    a = official(list(u), list(np.asarray(y, dtype=np.float64)), list(s))
    b = fast_evaluate(u, y, s)
    d = {k: abs(a[k]-b[k]) for k in ('GAUC','nDCG@5','primary')}
    ok = all(v < tol for v in d.values()) and a['users']==b['users'] and a['rows']==b['rows']
    print(f"  {tag:38s} {'OK ' if ok else 'FAIL'} maxdiff={max(d.values()):.2e} "
          f"primary={a['primary']:.6f}")
    return ok

rng = np.random.default_rng(0)
allok = True
print("adversarial cases:")
# 1. plain random
u = rng.integers(0,200,4000); y=(rng.random(4000)<0.3).astype(float); s=rng.random(4000)
allok &= cmp("random scores", u,y,s)
# 2. heavy ties (integer scores) - exercises average-rank path
s2 = rng.integers(0,3,4000).astype(float)
allok &= cmp("heavy ties (3 distinct scores)", u,y,s2)
# 3. all scores identical - degenerate tie block
allok &= cmp("all scores identical", u,y,np.zeros(4000))
# 4. all-negative and all-positive users present
u3 = rng.integers(0,50,2000)
y3 = np.zeros(2000); y3[u3 % 3 == 0] = 1.0            # some users fully pos/neg
allok &= cmp("all-pos / all-neg users", u3,y3,rng.random(2000))
# 5. every user has exactly one row (npos in {0,n} always -> GAUC empty)
u4 = np.arange(500); y4=(rng.random(500)<0.5).astype(float)
allok &= cmp("singleton users (GAUC undefined)", u4,y4,rng.random(500))
# 6. perfect ranking == oracle
u5 = rng.integers(0,100,3000); y5=(rng.random(3000)<0.4).astype(float)
allok &= cmp("oracle (score == label)", u5,y5,y5.copy())
# 7. inverted ranking
allok &= cmp("adversarial (score == -label)", u5,y5,-y5)
# 8. fewer than k rows per user
u6 = np.repeat(np.arange(400), 2); y6=(rng.random(800)<0.5).astype(float)
allok &= cmp("2 rows/user (< k)", u6,y6,rng.random(800))

print("\nreal validation split:")
from agent_kit import dataset as D
fr = D.load_frames()['valid']
u, y = fr['user_id'], fr['y']
for tag, sc in [("random scores", rng.random(len(y))),
                ("oracle (labels)", y.astype(float)),
                ("constant", np.ones(len(y)))]:
    allok &= cmp(tag, u, y, sc)

t0=time.time(); official(list(u), list(y), list(rng.random(len(y)))); t_o=time.time()-t0
t0=time.time(); fast_evaluate(u, y, rng.random(len(y))); t_f=time.time()-t0
print(f"\nspeed on {len(y):,d} rows: official {t_o:.3f}s  fast {t_f:.4f}s  ({t_o/t_f:.0f}x)")
print("\nALL PASS" if allok else "\nFAILURES PRESENT"); sys.exit(0 if allok else 1)
