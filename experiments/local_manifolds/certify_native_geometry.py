"""Exact rational interval certificate for the per-frame normalized CSD map."""
from pathlib import Path
import sys,json
import numpy as np
import scipy.linalg as la
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
from certify_patches import F,rat,add,scale,mul,diff,source_poly,minimum_bernstein

def bounds(p,box):return minimum_bernstein(p,box),-minimum_bernstein(scale(p,-1),box)
def divide(a,b):
 assert b[0]>0
 v=[x/y for x in a for y in b];return min(v),max(v)

def main():
 results=[]
 for link in 'ABCD':
  src=np.load(P/link/'source.npz');c=np.load(P/link/'curvature.npz');fac=np.concatenate([src['x'][None],c['low_directions']]);field=np.einsum('mpi,ami->amp',src['bases'],fac)
  # Nominate two real physical-CSD entries; only nomination uses floating point.
  rng=np.random.default_rng(2026103109);pairs=rng.integers(0,field.shape[-1],size=(1024,2));der=[]
  for i,j in pairs:
   v=np.zeros(2,complex)
   for m in range(10):
    den=np.vdot(fac[0,m],fac[0,m]).real;n=field[0,m,i]*field[0,m,j].conjugate()
    v+=(field[1:,m,i]*field[0,m,j].conjugate()+field[0,m,i]*field[1:,m,j].conjugate())/den-n*2*np.real(fac[1:,m].conj()@fac[0,m])/den**2
   der.append(v)
  der=np.array(der);_,_,piv=la.qr(np.r_[der.real,der.imag].T,pivoting=True);chosen=piv[:2]
  den=[source_poly(fac[:,m,:,None]) for m in range(10)];nums=[]
  for idx in chosen:
   imag=idx>=1024;i,j=pairs[idx%1024];ns=[]
   for m in range(10):
    small=field[:,m,[i,j]].reshape(3,2,1);ns.append(source_poly(small,0,1,bool(imag)))
   nums.append(ns)
  A=[]
  for ns in nums:
   A.append([sum(((diff(n,j).get((0,0),F(0))*q[(0,0)]-n.get((0,0),F(0))*diff(q,j).get((0,0),F(0)))/q[(0,0)]**2 for n,q in zip(ns,den)),F(0)) for j in range(2)])
  det=A[0][0]*A[1][1]-A[0][1]*A[1][0];assert det
  inv=[[A[1][1]/det,-A[0][1]/det],[-A[1][0]/det,A[0][0]/det]]
  tops=[]
  for i in range(2):
   row=[]
   for j in range(2):
    row.append([add(mul(diff(n,j),q),scale(mul(n,diff(q,j)),-1)) for n,q in zip([add(scale(nums[0][m],inv[i][0]),scale(nums[1][m],inv[i][1])) for m in range(10)],den)])
   tops.append(row)
  ok=False;attempts=[]
  for cells in [8,16,32]:
   ok=True;margin=None
   for ix in range(cells):
    if not ok:break
    for iy in range(cells):
     r=F(1,10);box=(-r+2*r*ix/cells,-r+2*r*(ix+1)/cells,-r+2*r*iy/cells,-r+2*r*(iy+1)/cells)
     db=[bounds(mul(q,q),box) for q in den]
     if min(x[0] for x in db)<=0:ok=False;break
     jac=[]
     for i in range(2):
      jr=[]
      for j in range(2):
       vals=[divide(bounds(n,box),q) for n,q in zip(tops[i][j],db)];jr.append((sum(v[0] for v in vals),sum(v[1] for v in vals)))
      jac.append(jr)
     off=max(abs((jac[0][1][0]+jac[1][0][0])/2),abs((jac[0][1][1]+jac[1][0][1])/2))
     lower=min(jac[0][0][0],jac[1][1][0],jac[0][0][0]*jac[1][1][0]-off**2)
     margin=lower if margin is None else min(margin,lower)
     if lower<=0:ok=False;break
   attempts.append(dict(cells=cells,passed=ok,lower_margin=float(margin) if margin is not None else None))
   if ok:break
  row=dict(link=link,box=[-.1,.1],passed=ok,method='exact rational Bernstein interval bounds on the symmetric projected CSD Jacobian, with ten separate normalization denominators',attempts=attempts,scope='stored numerical field and coordinate model, no propagation rounding enclosure')
  results.append(row);(P/'geometry_certificates.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)

if __name__=='__main__':main()
