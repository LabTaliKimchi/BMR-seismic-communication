import importlib.util, os, time
spec=importlib.util.spec_from_file_location("seqmod","sequence_identity_R1R5.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
t=time.time()
df=mod.build_sequence_table(mod.CLICKS_DIR, mod.META_DIR)
print("built %d sequences in %.1fs -> %s"%(len(df),time.time()-t, os.path.join(mod.OUT_DIR,"sequence_table_file_by_channel.csv")))
print("moles:",sorted(df['mole_id'].unique()))
print(df['mole_id'].value_counts().to_string())
print("n_drums describe:\n",df['n_drums'].describe().to_string())
