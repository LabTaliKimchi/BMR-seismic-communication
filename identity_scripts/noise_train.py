import os, sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
ND="cache/noise"
Xs=[];ys=[]
for f in sorted(os.listdir(ND)):
    W=np.load(f"{ND}/{f}"); Xs.append(W); ys+=[f[:2]]*len(W)
X=np.vstack(Xs); y=np.array(ys)
m=~np.isnan(X).any(axis=1); X=X[m]; y=y[m]
print("noise X:",X.shape,"classes",dict(zip(*np.unique(y,return_counts=True))),flush=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.30,stratify=y,random_state=42)
# subsample train for memory/time (stratified), evaluate on FULL test
CAP=200000
if len(Xtr)>CAP:
    idx,_=train_test_split(np.arange(len(Xtr)),train_size=CAP,stratify=ytr,random_state=42)
    Xtr2,ytr2=Xtr[idx],ytr[idx]
else:
    Xtr2,ytr2=Xtr,ytr
le=LabelEncoder(); etr=le.fit_transform(ytr2); ete=le.transform(yte)
cw=compute_class_weight("balanced",classes=np.unique(etr),y=etr); cwm=dict(zip(np.unique(etr),cw))
t=time.time()
clf=RandomForestClassifier(n_estimators=300,max_depth=None,class_weight=cwm,random_state=42,n_jobs=2).fit(Xtr2,etr)
print("fit %.1fs (train subsample %d)"%(time.time()-t,len(Xtr2)),flush=True)
pred=clf.predict(Xte)
acc=accuracy_score(ete,pred); bal=balanced_accuracy_score(ete,pred); mf1=f1_score(ete,pred,average="macro")
print("NOISE-ONLY  Raw accuracy: %.4f | Balanced accuracy: %.4f | Macro-F1: %.4f"%(acc,bal,mf1))
print(classification_report(ete,pred,target_names=le.classes_,digits=3))
