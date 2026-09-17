import os, sys, json, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
DATA="/sessions/loving-lucid-newton/mnt/Moles/Data"
CL=os.path.join(DATA,"clicks"); ME=os.path.join(DATA,"clicks_metadata")
CACHE="/sessions/loving-lucid-newton/mnt/outputs/rerun/cache"
PLOTS="/sessions/loving-lucid-newton/mnt/outputs/rerun/plots"
os.makedirs(CACHE,exist_ok=True); os.makedirs(PLOTS,exist_ok=True)
SEED=42
NCOLS_META=4

def find_meta(fp):
    b=os.path.basename(fp)[:4]
    for f in os.listdir(ME):
        if f.startswith(b) and f.endswith("-clicks_channels.xlsx"): return os.path.join(ME,f)
    return None

def load():
    files=sorted([f for f in os.listdir(CL) if f.endswith(".xlsx") and not f.startswith("~$") and not f.upper().startswith("R6")])
    feats=[]; labs=[]
    for f in files:
        fp=os.path.join(CL,f)
        cl=pd.read_excel(fp,header=None).values
        mp=find_meta(fp)
        if mp is not None:
            mv=pd.read_excel(mp,header=None).iloc[:,:NCOLS_META].values
            n=min(len(cl),len(mv)); cl=cl[:n]
        merged=cl.astype(float)  # clicks cols only (meta not used for waveform features)
        feats.append(merged); labs+=[f[:2]]*merged.shape[0]
    X=np.vstack(feats); y=np.array(labs)
    return X,y

def norm_rows_max1(m):
    m=m.astype(float,copy=True)
    rmax=np.max(np.nan_to_num(m,nan=-np.inf),axis=1,keepdims=True)
    with np.errstate(divide="ignore",invalid="ignore"):
        np.divide(m,rmax,out=m,where=(rmax>0))
    return m

def features(X):
    clicks=norm_rows_max1(X)          # normalize full waveform row by its max
    return clicks[:,7:30]             # cols 8..30 (1-based) -> 23 features

def stageA():
    t=time.time(); X,y=load()
    Xs=features(X)
    mask=~np.isnan(Xs).any(axis=1); Xs=Xs[mask]; y=y[mask]
    from sklearn.model_selection import train_test_split
    idx=np.arange(len(y))
    tr,te=train_test_split(idx,stratify=y,test_size=0.3,random_state=SEED)
    np.save(f"{CACHE}/X.npy",Xs); np.save(f"{CACHE}/y.npy",y)
    np.save(f"{CACHE}/tr.npy",tr); np.save(f"{CACHE}/te.npy",te)
    print(f"[A] X={Xs.shape} classes={dict(zip(*np.unique(y,return_counts=True)))} tr={len(tr)} te={len(te)} {time.time()-t:.1f}s")

def _oversample(Xtr,ytr):
    from imblearn.over_sampling import BorderlineSMOTE, ADASYN
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder(); e=le.fit_transform(ytr)
    sm=BorderlineSMOTE(random_state=SEED); Xa,ea=sm.fit_resample(Xtr,e)
    ad=ADASYN(random_state=SEED); Xa,ea=ad.fit_resample(Xa,ea)
    return Xa,ea,le

def stageB():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report, confusion_matrix, f1_score
    import joblib
    t=time.time()
    X=np.load(f"{CACHE}/X.npy"); y=np.load(f"{CACHE}/y.npy",allow_pickle=True)
    tr=np.load(f"{CACHE}/tr.npy"); te=np.load(f"{CACHE}/te.npy")
    Xtr,ytr=X[tr],y[tr]; Xte,yte=X[te],y[te]
    Xa,ea,le=_oversample(Xtr,ytr)
    cw=compute_class_weight('balanced',classes=np.unique(ea),y=ea); cwm=dict(zip(np.unique(ea),cw))
    rf=RandomForestClassifier(n_jobs=1,class_weight=cwm,random_state=SEED)
    grid={'n_estimators':[100,150],'max_depth':[10,None],'min_samples_split':[2],'min_samples_leaf':[1],'bootstrap':[True]}
    search=RandomizedSearchCV(rf,grid,n_iter=4,cv=3,n_jobs=-1,random_state=SEED,verbose=0)
    search.fit(Xa,ea); model=search.best_estimator_
    yte_e=le.transform(yte); pred=model.predict(Xte)
    bal=balanced_accuracy_score(yte_e,pred); acc=accuracy_score(yte_e,pred); mf1=f1_score(yte_e,pred,average='macro')
    rep=classification_report(yte_e,pred,target_names=le.classes_,digits=3)
    cm=confusion_matrix(yte_e,pred)
    joblib.dump({'model':model,'le':le},f"{CACHE}/model.joblib")
    pd.DataFrame(cm,index=list(le.classes_),columns=list(le.classes_)).to_csv(f"{PLOTS}/cols8_30_confusion_counts.csv")
    print(f"[B] best={search.best_params_}")
    print(f"[B] Balanced accuracy: {bal:.4f} | Raw acc: {acc:.4f} | Macro-F1: {mf1:.4f}")
    print("[B] Classification report:\n"+rep)
    print("CONFUSION_LABELS="+json.dumps(list(le.classes_)))
    print("CONFUSION_COUNTS="+json.dumps(cm.tolist()))
    print(f"[B] {time.time()-t:.1f}s")

if __name__=="__main__":
    {"A":stageA,"B":stageB}[sys.argv[1]]()
