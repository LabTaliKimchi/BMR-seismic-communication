import numpy as np, warnings, itertools, json, os, time
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score, accuracy_score
X=np.load("cache/X.npy"); y=np.load("cache/y.npy",allow_pickle=True)
SEX={"R1":"M","R2":"F","R3":"M","R4":"F","R5":"M"}; animals=["R1","R2","R3","R4","R5"]
groups=y
def fit_eval(sex_map):
    s=np.array([sex_map[a] for a in y])
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=0.30,random_state=42).split(X,s,groups))
    if len(tr)>12000:
        tr=np.random.default_rng(0).choice(tr,12000,replace=False)
    clf=RandomForestClassifier(n_estimators=80,max_depth=10,class_weight="balanced",n_jobs=2,random_state=42).fit(X[tr],s[tr])
    p=clf.predict(X[te]); return float(balanced_accuracy_score(s[te],p)), float(accuracy_score(s[te],p)), sorted(set(map(str,y[te])))
ST="cache/sex_results.json"; R=json.load(open(ST)) if os.path.exists(ST) else {}
if "obs" not in R:
    b,a,ta=fit_eval(SEX); R["obs"]={"bal":b,"acc":a,"test":ta}; R["null"]={}; json.dump(R,open(ST,"w"))
    print("observed:",R["obs"],flush=True)
assigns=[c for c in itertools.combinations(animals,2)]
start=time.time()
for females in assigns:
    key=",".join(females); m={a:("F" if a in females else "M") for a in animals}
    if m==SEX or key in R["null"]: continue
    if time.time()-start>33: break
    b,_,_=fit_eval(m); R["null"][key]=b; json.dump(R,open(ST,"w")); print("null",key,round(b,3),flush=True)
nk=[k for k in [",".join(c) for c in assigns] if k!="R2,R4"]
done=[k for k in R["null"]]
print("null done %d/%d"%(len(done),len(nk)))
if len(done)>=len(nk):
    nulls=np.array([R["null"][k] for k in R["null"]]); obs=R["obs"]["bal"]
    p=(np.sum(nulls>=obs)+1)/(len(nulls)+1)
    print("SEX: observed balanced acc=%.3f | null mean=%.3f SD=%.3f max=%.3f | p=%.3f => %s"%(
        obs,nulls.mean(),nulls.std(ddof=1),nulls.max(),p,"NON-significant" if p>0.05 else "significant"))
