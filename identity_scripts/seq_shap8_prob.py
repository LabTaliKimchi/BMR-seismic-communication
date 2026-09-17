import numpy as np, pandas as pd, time, os, json, shap, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
FE4=["mean_gap","sd_gap","n_drums","sd_amplitude"]
LAB={"mean_gap":"Mean inter-drum interval","sd_gap":"SD inter-drum interval","n_drums":"Number of head-drums","sd_amplitude":"SD head-drum amplitude"}
df=pd.read_csv("plots/seq4/sequence_table_file_by_channel.csv").dropna(subset=FE4)
y=df["mole_id"].to_numpy(); X=df[FE4].to_numpy(float)
STORE="cache/seq_shap8_prob.json"
res=json.load(open(STORE)) if os.path.exists(STORE) else {}
start=time.time()
for s in range(42,50):
    if str(s) in res: continue
    if time.time()-start>34: break
    tr,te=train_test_split(np.arange(len(y)),test_size=0.30,random_state=s,stratify=y)
    le=LabelEncoder(); etr=le.fit_transform(y[tr])
    m=RandomForestClassifier(n_estimators=400,max_features="sqrt",class_weight="balanced",n_jobs=3,random_state=s).fit(X[tr],etr)
    bg=X[tr][np.random.RandomState(0).choice(len(tr),min(150,len(tr)),replace=False)]
    ex=shap.TreeExplainer(m,data=bg,feature_perturbation="interventional",model_output="probability")
    sv=ex.shap_values(X[te],check_additivity=False)
    if isinstance(sv,list): gm=np.vstack([np.abs(np.asarray(a)).mean(0) for a in sv]).mean(0)[:4]
    else:
        a=np.asarray(sv); gm=np.vstack([np.abs(a[:,:,k]).mean(0) for k in range(a.shape[2])]).mean(0)[:4]
    res[str(s)]=gm.tolist(); print("seed %d done %.1fs"%(s,time.time()-start),flush=True)
json.dump(res,open(STORE,"w"))
print("have seeds:",sorted(res.keys()))
if len(res)==8:
    V=np.array([res[str(s)] for s in range(42,50)])
    means=V.mean(0); sds=V.std(0,ddof=1)
    ranks=np.zeros_like(V)
    for i in range(len(V)):
        o=np.argsort(-V[i]); rk=np.empty(4); rk[o]=np.arange(1,5); ranks[i]=rk
    mr=ranks.mean(0); rows=[]
    print("\n8-SEED SHAP (4 params, interventional probability):")
    for j,f in enumerate(FE4):
        print("  %-28s: %.4f ± %.4f | mean rank %.2f"%(LAB[f],means[j],sds[j],mr[j]))
        rows.append({"feature":f,"label":LAB[f],"mean_abs_shap":round(float(means[j]),4),"sd":round(float(sds[j]),4),"mean_rank":round(float(mr[j]),2)})
    pd.DataFrame(rows).sort_values("mean_rank").to_csv("plots/seq4/seq4_8seed_shap_prob_summary.csv",index=False)
    print("SAVED summary")
