"""Iteration 1-N: establish the loss-function frontier, 3 seeds each."""
import sys
from agent_kit.experiment import safe_run, summarise
from agent_kit.journal import Journal

J = Journal()
BASE = {'model': {'type':'fm','k':16},
        'train': {'lr':0.001,'batch':8192,'epochs':40,'patience':4,'l2':1e-6}}

def run(hyp, loss, over=None, rationale=None, tags=()):
    cfg = {'model': dict(BASE['model']), 'loss': loss,
           'train': {**BASE['train'], **(over or {})}}
    res = safe_run(cfg, seeds=(0,1,2), time_budget_s=900)
    print(f"[iter {J.next_iteration}] {hyp}\n    {summarise(res)}", flush=True)
    J.record(hypothesis=hyp, rationale=rationale, config=cfg, result=res,
             status=res['status'], error=res.get('error'), tags=list(tags))
    return res

run("Reproduce the FM baseline in the torch harness (control).",
    {'type':'bce'},
    rationale="Nothing downstream is comparable until the harness reproduces the "
              "official 0.6016 under its own sampler and optimiser.",
    tags=['control','reproduction'])

run("Swap pointwise log-loss for within-user BPR: the metric ranks within a user, "
    "so put the gradient on within-user comparisons.",
    {'type':'bpr'},
    rationale="GAUC/nDCG only read the ordering inside each user. Pointwise BCE spends "
              "capacity calibrating absolute probabilities, and any term constant within "
              "a user cancels from the metric entirely.",
    tags=['loss','ranking','priority-1'])

run("Listwise softmax with 8 sampled negatives per positive, same user.",
    {'type':'softmax'}, {'batch':4096,'group_size':8},
    rationale="Generalises BPR from 1 negative to G-1; more negatives per step should "
              "sharpen the within-user ordering signal.",
    tags=['loss','ranking'])

run("LambdaRank weighting pairs by |delta nDCG@5|, truncated at k=5 like the metric.",
    {'type':'lambdarank','k':5}, {'batch':4096,'group_size':8},
    rationale="Half the primary score is nDCG@5. Weight each pair by how much swapping "
              "it would move nDCG@5, so gradient goes where the metric moves.",
    tags=['loss','ranking','priority-2'])

J.render_markdown()
print("\n" + J.digest())
