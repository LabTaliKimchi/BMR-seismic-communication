import os, time, numpy as np, pandas as pd
ND="/sessions/loving-lucid-newton/mnt/Moles/Data/no_clicks"
OUT="/sessions/loving-lucid-newton/mnt/outputs/rerun/cache/noise"
files=sorted([f for f in os.listdir(ND) if f.endswith(".xlsx") and not f.startswith("~$") and not f.upper().startswith("R6")])
start=time.time()
for f in files:
    tag=f.replace(".xlsx","")
    p=f"{OUT}/{tag}.npy"
    if os.path.exists(p): continue
    if time.time()-start>34: print("time budget reached"); break
    df=pd.read_excel(os.path.join(ND,f),header=None)
    wav=df.values[:, :41].astype(float)      # 41 waveform cols
    center=41//2; half=23//2; s=center-half; e=s+23   # cols 9..31 (23)
    W=wav[:, s:e]
    np.save(p, W)
    print(f"cached {tag}: {W.shape} ({time.time()-start:.1f}s)",flush=True)
print("cached files:", sorted(os.listdir(OUT)))
