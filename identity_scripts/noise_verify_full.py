import os, sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
mode=sys.argv[1]
ND="cache/noise_all"; Xs=[];ys=[]
for f in sorted(os.listdir(ND)):
    if mode=="5" and f.upper().startswith("R6"): continue
    W=np.load(f"{ND}/{f}"); Xs.append(W); ys+=[f[:2]]*len(W)
X=np.vstack(Xs); y=np.array(ys); m=~np.isnan(X).any(axis=1); X=X[m]; y=y[m]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.30,stratify=y,random_state=42)  # FULL train, no cap
le=LabelEncoder(); etr=le.fit_transform(ytr); ete=le.transform(yte)
cw=compute_class_weight("balanced",classes=np.unique(etr),y=etr); cwm=dict(zip(np.unique(etr),cw))
t=time.time()
clf=RandomForestClassifier(n_estimators=300,max_depth=None,class_weight=cwm,random_state=42,n_jobs=2).fit(Xtr,etr)
pred=clf.predict(Xte)
print("FULL config mode=%s-class train=%d fit %.1fs | Raw acc=%.4f | Balanced=%.4f | Macro-F1=%.4f"%(
    mode,len(Xtr),time.time()-t,accuracy_score(ete,pred),balanced_accuracy_score(ete,pred),f1_score(ete,pred,average="macro")),flush=True)
