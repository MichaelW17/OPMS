from pathlib import Path
import json,csv
import numpy as np
P=Path(__file__).resolve().parent;rows=[]
for link in 'ABCD':
 for folder in sorted((P/'native_paths'/link).glob('*_far')):
  z=np.load(folder/'validation_powers.npz');eta=.1 if link in 'AB' else .2
  for kind in ['straight','curve']:
   power=z[kind];assert power.shape==(1000,33)
   mu=power.mean(0);s=-5*np.log10(eta/mu+power.var(0)/mu**2)
   rows.append(dict(link=link,pair=folder.name,path=kind,barrier_db=float(max(0,min(s[0],s[-1])-min(s)))))
assert len(rows)==32 and all(r['barrier_db']<.5 for r in rows if r['path']=='curve')
out=P.parents[1]/'results/path_barriers.csv'
with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Verified all 16 optimized paths: barriers below0.5dB at33positions.')
