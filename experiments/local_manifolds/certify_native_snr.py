"""Exact rational Bernstein interval SNR certificate with ten denominators."""
from certify_native_geometry import *
from math import lcm
import time

def main():
 results=[];modes=[(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]
 for link in 'ABCD':
  start=time.time();z=np.load(P/link/'patches.npz');Q=z['low_validation_features'];src=np.load(P/link/'source.npz');d=np.load(P/link/'curvature.npz')['low_directions'];fac=np.concatenate([src['x'][None],d]);qs=[source_poly(fac[:,m,:,None]) for m in range(10)]
  vals=np.stack([Q[:,:,0,0],2*Q[:,:,0,1],2*Q[:,:,0,2],Q[:,:,1,1],2*Q[:,:,1,2],Q[:,:,2,2]],axis=-1).reshape(1000,60)
  # All binary floating-point coefficients share a power-of-two denominator.
  fractions=[[rat(x) for x in row] for row in vals];den=max(x.denominator for row in fractions for x in row)
  ints=np.array([[x.numerator*(den//x.denominator) for x in row] for row in fractions],dtype=object)
  sums=ints.sum(0);mom=ints.T@ints;mean=[{k:F(int(sums[m*6+i]),1000*den) for i,k in enumerate(modes)} for m in range(10)]
  cov={}
  for m in range(10):
   for n in range(m,10):
    poly={}
    for i,a in enumerate(modes):
     for j,b in enumerate(modes):
      k=(a[0]+b[0],a[1]+b[1]);v=F(int(mom[m*6+i,n*6+j]),1000*den**2)-mean[m][a]*mean[n][b];poly[k]=poly.get(k,F(0))+v
    cov[m,n]=poly
  eta=rat(.1 if link in 'AB' else .2);mu0=100*sum(mean[m][(0,0)]/qs[m][(0,0)] for m in range(10));var0=10000*sum((1 if m==n else 2)*cov[m,n].get((0,0),F(0))/(qs[m][(0,0)]*qs[n][(0,0)]) for m in range(10) for n in range(m,10));f0=eta/mu0+var0/mu0**2
  lo=F(978,1000);hi=F(1023,1000);assert lo**100>F(1,10) and hi**100<10
  terms=[]
  for m in range(10):
   for n in range(m,10):
    mult=1 if m==n else 2;denpoly=mul(qs[m],qs[n]);base=scale(cov[m,n],10000*mult)
    noise=scale(mul(mean[m],qs[n]),10*eta) if m==n else scale(add(mul(mean[m],qs[n]),mul(mean[n],qs[m])),10*eta)
    base=add(base,noise);mm=scale(mul(mean[m],mean[n]),10000*mult)
    terms.append((denpoly,add(base,scale(mm,-lo*f0)),add(scale(mm,hi*f0),scale(base,-1))))
  attempts=[];ok=False
  for cells in [8,16,32,64,128]:
   ok=True;margin=None
   for ix in range(cells):
    if not ok:break
    for iy in range(cells):
     r=F(1,10);box=(-r+2*r*ix/cells,-r+2*r*(ix+1)/cells,-r+2*r*iy/cells,-r+2*r*(iy+1)/cells);lower=[F(0),F(0)]
     for dp,a,b in terms:
      db=bounds(dp,box)
      if db[0]<=0:ok=False;break
      lower[0]+=divide(bounds(a,box),db)[0];lower[1]+=divide(bounds(b,box),db)[0]
     value=min(lower);margin=value if margin is None else min(margin,value)
     if not ok or value<=0:ok=False;break
   attempts.append(dict(cells=cells,passed=ok,margin=float(margin) if margin is not None else None));print(json.dumps(dict(link=link,attempt=attempts[-1])),flush=True)
   if ok:break
  row=dict(link=link,box=[-.1,.1],tolerance_db=.05,passed=ok,attempts=attempts,elapsed_seconds=time.time()-start,method='exact rational response moments and sum-of-rational Bernstein interval inequalities; distinct per-frame quadratic denominators',scope='stored1000-atmosphere response model')
  results.append(row);(P/'snr_certificates.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)

if __name__=='__main__':main()
