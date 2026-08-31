"""Re-score the regularised branch under a fair PER-SEED time budget.

Iterations 12/13/14 all hit the old shared deadline, so their numbers were
under-trained lower bounds and their ranking was an artefact of which seed ran
out of clock first. Same configs, honest budget.
"""
from agent_kit.experiment import safe_run, summarise
from agent_kit.journal import Journal

J = Journal()
BUDGET = 1200          # per seed

CONFIGS = [
 ("REDO iter12: FM+softmax, dropout p=0.3 l2=1e-4, fair per-seed budget.",
  {'model':{'type':'fm_dropout','k':16,'p':0.3},'loss':{'type':'softmax'},
   'train':{'lr':0.001,'batch':4096,'epochs':40,'patience':4,'l2':1e-4,'group_size':8}}),
 ("REDO iter13: same but p=0.5 -- seed1 reached epoch 10 still improving when the clock died.",
  {'model':{'type':'fm_dropout','k':16,'p':0.5},'loss':{'type':'softmax'},
   'train':{'lr':0.001,'batch':8192,'epochs':40,'patience':4,'l2':1e-4,'group_size':8}}),
 ("REDO iter14: DIN head under iteration 12's regularisation recipe.",
  {'model':{'type':'din_dropout','k':16,'L':8,'p':0.3},'loss':{'type':'softmax'},
   'train':{'lr':0.001,'batch':4096,'epochs':40,'patience':4,'l2':1e-4,'group_size':8}}),
]

for hyp, cfg in CONFIGS:
    res = safe_run(cfg, seeds=(0,1,2), time_budget_s=BUDGET)
    print(f"\n[iter {J.next_iteration}] {hyp}")
    print(f"   {summarise(res)}")
    print(f"   epochs_run={res.get('epochs_run')} secs/epoch={res.get('secs_per_epoch')} "
          f"timed_out={res.get('timed_out_seeds')}")
    for i, c in enumerate(res.get('epoch_curves') or []):
        print(f"   seed{i} curve: {[round(v,4) for v in c]}")
    J.record(hypothesis=hyp, config=cfg, result=res, status=res['status'],
             error=res.get('error'), recovery=res.get('hint'),
             rationale="Old shared deadline cut later seeds to 1 epoch; re-measuring "
                       "with a per-seed budget so the comparison is fair.",
             tags=['rerun','budget-fix'])
J.render_markdown()
print("\n" + J.digest())
