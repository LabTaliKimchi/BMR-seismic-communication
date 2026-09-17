import sys, numpy as np, pandas as pd, json, warnings, time
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, classification_report, confusion_matrix
CSV="plots/seq4/sequence_table_file_by_channel.csv"
FE4=["mean_gap","sd_gap","n_drums","sd_amplitude"]; FE2=["mean_gap","sd_gap"]
LAB={"mean_gap":"Mean inter-drum interval","sd_gap":"SD inter-drum interval","n_drums":"Number of head-drums","sd_amplitude":"SD head-drum amplitude"}
df=pd.read_csv(CSV).dropna(subset=FE4); y=df["mole_id"].to_numpy()
def fit_rf(Xtr,ytr,seed=42,n=400):
    return RandomForestClassifier(n_estimators=n,max_depth=None,min_samples_split=2,min_samples_leaf=1,
        max_features="sqrt",class_weight="balanced",n_jobs=3,random_state=seed).fit(Xtr,ytr)
def model(feats,name):
    X=df[feats].to_numpy(float)
    tr,te=train_test_split(np.arange(len(y)),test_size=0.30,random_state=42,stratify=y)
    le=LabelEncoder(); etr=le.fit_transform(y[tr]); ete=le.transform(y[te])
    m=fit_rf(X[tr],etr); pred=m.predict(X[te])
    bal=balanced_accuracy_score(ete,pred); acc=accuracy_score(ete,pred); mf1=f1_score(ete,pred,average="macro")
    print(f"=== {name} ({len(feats)} params) ===")
    print(f"Balanced accuracy: {bal:.4f} | Raw acc: {acc:.4f} | Macro-F1: {mf1:.4f} | Train/Test {len(tr)}/{len(te)}")
    if len(feats)==4:
        print(classification_report(ete,pred,target_names=le.classes_,digits=3))
        cm=confusion_matrix(ete,pred); pd.DataFrame(cm,index=list(le.classes_),columns=list(le.classes_)).to_csv("plots/seq4/seq4_confusion_counts.csv")
    return bal
def perm(feats,name,nperm=200,ntree=150):
    X=df[feats].to_numpy(float)
    tr,te=train_test_split(np.arange(len(y)),test_size=0.30,random_state=42,stratify=y)
    le=LabelEncoder(); le.fit(y)
    bal=model(feats,name)  # observed
    rng=np.random.default_rng(42); pv=np.empty(nperm)
    for i in range(nperm):
        yp=rng.permutation(y)
        mp=fit_rf(X[tr],le.transform(yp[tr]),n=ntree); pv[i]=balanced_accuracy_score(le.transform(yp[te]),mp.predict(X[te]))
    print(f"Permutation ({nperm}, {ntree} trees): null mean={pv.mean():.4f} SD={pv.std(ddof=1):.4f}  p={np.mean(pv>=bal):.4f}")
def shap8():
    import shap
    X=df[FE4].to_numpy(float); V=[]
    for s in range(42,50):
        tr,te=train_test_split(np.arange(len(y)),test_size=0.30,random_state=s,stratify=y)
        le=LabelEncoder(); etr=le.fit_transform(y[tr]); m=fit_rf(X[tr],etr,seed=s)
        bg=X[tr][np.random.RandomState(0).choice(len(tr),min(150,len(tr)),replace=False)]
        try:
            ex=shap.TreeExplainer(m,data=bg,feature_perturbation="interventional",model_output="probability"); sv=ex.shap_values(X[te],check_additivity=False)
        except Exception:
            sv=shap.TreeExplainer(m).shap_values(X[te],check_additivity=False)
        if isinstance(sv,list): arrs=[np.abs(np.asarray(a)) for a in sv]
        else:
            a=np.asarray(sv); arrs=[np.abs(a[:,:,k]) for k in range(a.shape[2])] if a.ndim==3 else [np.abs(a)]
        V.append(np.vstack([mm.mean(0) for mm in arrs]).mean(0)[:4])
    V=np.vstack(V); means=V.mean(0); sds=V.std(0,ddof=1)
    ranks=np.zeros_like(V)
    for i in range(len(V)):
        o=np.argsort(-V[i]); rk=np.empty(4); rk[o]=np.arange(1,5); ranks[i]=rk
    mr=ranks.mean(0); rows=[]
    print("8-SEED SHAP (4 params):")
    for j,f in enumerate(FE4):
        print(f"  {LAB[f]:28s}: {means[j]:.4f} ± {sds[j]:.4f} | mean rank {mr[j]:.2f}")
        rows.append({"feature":f,"label":LAB[f],"mean_abs_shap":means[j],"sd":sds[j],"mean_rank":mr[j]})
    pd.DataFrame(rows).sort_values("mean_rank").to_csv("plots/seq4/seq4_8seed_shap_summary.csv",index=False)
{"m4":lambda:perm(FE4,"Sequence 4-param"),"m2":lambda:perm(FE2,"Sequence 2-param"),"shap":shap8}[sys.argv[1]]()
