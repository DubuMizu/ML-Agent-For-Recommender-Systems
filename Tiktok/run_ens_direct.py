import json, time
from agent_kit import ensemble_search as ES

F = {
 "fm_softmax": ({"model":{"type":"fm_dropout","k":16,"p":0.34},"loss":{"type":"softmax"},
   "train":{"lr":0.00116,"batch":8192,"epochs":40,"patience":4,"l2":1.2e-5,"group_size":4}},
   {"train.lr":["float",8e-4,1.8e-3,True],"model.p":["float",0.25,0.45,False],"train.group_size":["int",4,8]}),
 "din_softmax": ({"model":{"type":"din_dropout","k":16,"p":0.3},"loss":{"type":"softmax"},
   "train":{"lr":0.0012,"batch":8192,"epochs":40,"patience":4,"l2":3e-5,"group_size":6}},
   {"train.lr":["float",8e-4,2e-3,True],"model.p":["float",0.2,0.5,False],"train.group_size":["int",4,10]}),
 "deepfm_softmax": ({"model":{"type":"deepfm","k":16,"hidden":[128,64],"p":0.3},"loss":{"type":"softmax"},
   "train":{"lr":0.0012,"batch":8192,"epochs":40,"patience":4,"l2":3e-5,"group_size":6}},
   {"train.lr":["float",6e-4,2e-3,True],"model.p":["float",0.2,0.5,False],"train.group_size":["int",4,10]}),
 "fm_bpr": ({"model":{"type":"fm_dropout","k":16,"p":0.26},"loss":{"type":"bpr"},
   "train":{"lr":0.00134,"batch":8192,"epochs":40,"patience":4,"l2":1.1e-4}},
   {"train.lr":["float",9e-4,2e-3,True],"model.p":["float",0.2,0.45,False],"train.l2":["float",3e-5,4e-4,True]}),
}
t0=time.time()
out = ES.run(F, n_init=2, n_iter=1, seeds=(0,1), time_budget_s=420, max_ensemble=12)
print("ELAPSED", round(time.time()-t0,1))
try: print(ES.summarise(out))
except Exception as e: print("summarise err", e)
json.dump(out, open("runs/ensemble_search.json","w"), indent=2, default=float)
print("WROTE runs/ensemble_search.json")
print("ens_valid", out.get("ensemble_valid"), "best_single", out.get("best_single"), "unb", out.get("ensemble_unbiased"))
