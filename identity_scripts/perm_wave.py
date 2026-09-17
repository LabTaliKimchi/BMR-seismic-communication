import numpy as np, time, os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
C="/sessions/loving-lucid-newton/mnt/outputs/rerun/cache"
OUT=f"{C}/perm_wave.npy"
TARGET=100; BUDGET=38.0
X=np.load(f"{C}/X.npy"); y=np.load(f"{C}/y.npy",allow_pickle=True)
tr=np.load(f"{C}/tr.npy"); te=np.load(f"{C}/te.npy")
done=list(np.load(OUT)) if os.path.exists(OUT) else []
start=time.time(); i=len(done)
while len(done)<TARGET and (time.time()-start)<BUDGET:
    rng=np.random.default_rng(1000+i)
    yp=rng.permutation(y)
    le=LabelEncoder(); etr=le.fit_transform(yp[tr]); ete=le.transform(yp[te])
    # stratified 12k subsample of train for speed (chance level is model-agnostic)
    if len(tr)>12000:
        sub=rng.choice(len(tr),12000,replace=False)
        Xtr_,etr_=X[tr][sub],etr[sub]
    else:
        Xtr_,etr_=X[tr],etr
    clf=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=3).fit(Xtr_,etr_)
    done.append(balanced_accuracy_score(ete,clf.predict(X[te])))
    i+=1
np.save(OUT,np.array(done))
a=np.array(done)
print(f"perms done={len(done)}/{TARGET}  mean={a.mean():.4f} sd={a.std(ddof=1):.4f} max={a.max():.4f}")
