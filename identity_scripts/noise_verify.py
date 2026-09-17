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
print(f"mode={mode}-class X={X.shape} classes={sorted(set(y))}",flush=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.30,stratify=y,random_state=42)
CAP=45000
if len(Xtr)>CAP:
    idx,_=train_test_split(np.arange(len(Xtr)),train_size=CAP,stratify=ytr,random_state=42); Xtr,ytr=Xtr[idx],ytr[idx]
le=LabelEncoder(); etr=le.fit_transform(ytr); ete=le.transform(yte)
cw=compute_class_weight("balanced",classes=np.unique(etr),y=etr); cwm=dict(zip(np.unique(etr),cw))
t=time.time()
clf=RandomForestClassifier(n_estimators=150,max_depth=18,class_weight=cwm,random_state=42,n_jobs=2).fit(Xtr,etr)
pred=clf.predict(Xte)
print("fit %.1fs (train=%d) | Raw acc=%.4f | Balanced acc=%.4f | Macro-F1=%.4f"%(
    time.time()-t,len(Xtr),accuracy_score(ete,pred),balanced_accuracy_score(ete,pred),f1_score(ete,pred,average="macro")),flush=True)
