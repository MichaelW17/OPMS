"""Recompute all submission Fig3 scores from archived modal powers."""
from pathlib import Path
import csv,json
import numpy as np
R=Path(__file__).resolve().parent
rows=list(csv.DictReader((R/'data/fig3_six_conditions/gsm_hg_endpoint_scores.csv').open(encoding='utf-8-sig')))
total=0
for tag,label in [('gsm_m3','GSM, M=3'),('9hg_m3','9-HG, M=3'),('9hg_m10','9-HG, M=10'),('64hg_m3','64-HG, M=3'),('gsm','GSM, M=10'),('opms','64-HG, M=10')]:
 for link in 'ABCD':
  path=R/'data/fig3_six_conditions'/tag/f'{link}_flux.npz' if tag not in ['gsm','opms'] else R/'data/multistart'/f'{link}_{tag}_flux.npz'
  power=np.load(path)['modal_flux'].astype(np.float64).mean(-1)
  assert power.shape==(1024,1000) and np.isfinite(power).all() and power.min()>=0
  mu=power.mean(-1);eta=.1 if link in 'AB' else .2
  score=-5*np.log10(eta/mu+power.var(-1)/mu**2)
  expected=np.array([float(x['snr_db']) for x in rows if x['link']==link and x['condition']==label])
  error=float(max(abs(score-expected)));assert error<3e-4,(tag,link,error)
  total+=1024;print(link,label,'max SNR error:',error)
assert total==24576
print('Verified all 24576 endpoints.')
