"""Local CSD geometry at the highest-validated-SNR original M10 endpoints.

Every frame retains its own original waist and unit launch power before the
ten-frame average. No common-waist substitution or global-factor normalization.
"""
from pathlib import Path
import sys,json,time,os,argparse,hashlib
import numpy as np
import scipy.linalg as la
import torch
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from propagation import UnifiedWOS,LinkConfig
from sources import SMSParameters,synthesize_sms,hermite_gaussian_1d

def save(p,obj):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2),encoding='utf-8');t.replace(p)
def log(**x):
 x.update(updated_at=time.time(),pid=os.getpid());save(HERE/'progress.json',x);print(json.dumps(x),flush=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snr(p,eta):
 mu=p.mean(0);return -5*np.log10(eta/mu+p.var(0)/mu**2)

@torch.no_grad()
def prepare(link,cfg):
 folder=HERE/link;folder.mkdir(exist_ok=True)
 if (folder/'source.npz').exists():return
 ev=UnifiedWOS(LinkConfig(**cfg['links'][link]));metrics=json.loads((ROOT/f'data/multistart/{link}_opms_metrics.json').read_text())
 run=int(np.argmax([x['snr_db'] for x in metrics]));ends=torch.load(ROOT/f'data/multistart/{link}_opms_endpoints.pt',map_location='cpu')
 model=SMSParameters(10,8,1,.08,ev.device);model.load_state_dict(ends[run]['model_state'])
 amp,ph,w=model.physical();cc=torch.polar(amp,ph).cpu().numpy().astype(np.complex128)
 original=synthesize_sms(ev,model).cpu().numpy();mask=ev.transmitter_aperture.bool().cpu().numpy()
 one=hermite_gaussian_1d(ev,w,8).to(torch.complex64)
 bases=[];transforms=[];xs=[];zs=[];errors=[];conditions=[]
 for m in range(10):
  raw=(one[m,:,None,:,None]*one[m,None,:,None,:]).reshape(64,512,512)*ev.curvature_phase(1.)
  raw=raw[:,mask].cpu().numpy().astype(np.complex128).T
  basis,r=la.qr(raw,mode='economic');x=r@cc[m];x/=la.norm(x)
  complement=la.null_space(x.conj()[None]);z=np.concatenate([complement,1j*complement],axis=1)
  assert np.max(abs(basis.conj().T@basis-np.eye(64)))<1e-12
  assert np.max(abs(x.conj()@z))<1e-12
  err=la.norm(np.sqrt(1000)*basis@x-original[m,mask])/la.norm(original[m,mask]);assert err<2e-5,err
  bases.append(basis);transforms.append(r);xs.append(x);zs.append(z);errors.append(float(err));conditions.append(float(np.linalg.cond(r)))
 bases=np.array(bases);x=np.array(xs);z=np.array(zs)
 union=bases.transpose(1,0,2).reshape(mask.sum(),640);cross=(union.conj().T@union).reshape(10,64,10,64).transpose(0,2,1,3)
 np.savez_compressed(folder/'source.npz',bases=bases,raw_to_orthonormal=np.array(transforms),x=x,Z=z,cross=cross,mask=mask,waists=w.cpu().numpy(),original_fields=original[:,mask])
 save(folder/'selection.json',dict(link=link,run=run,original_validated_snr_db=metrics[run]['snr_db'],original_record=metrics[run],waists_m=w.cpu().tolist(),source_field_relative_errors=errors,basis_conditions=conditions,endpoint_file_sha256=digest(ROOT/f'data/multistart/{link}_opms_endpoints.pt'),metrics_file_sha256=digest(ROOT/f'data/multistart/{link}_opms_metrics.json')))
 log(stage='source_prepared',link=link,run=run,source_error=max(errors))

@torch.no_grad()
def response_bank(link,cfg,kind,N,batch=4):
 folder=HERE/link;dest=folder/f'{kind}_response.npy';state=folder/f'{kind}_state.pt'
 if (folder/f'{kind}_complete.json').exists():return
 ev=UnifiedWOS(LinkConfig(**cfg['links'][link]));src=np.load(folder/'source.npz');mask=src['mask'];seed=2026103101+100000*'ABCD'.index(link)+(50000 if kind=='validation' else 0)
 gen=ev.generator(seed);done=0
 if state.exists():
  st=torch.load(state,map_location='cpu');done=st['completed'];gen.set_state(st['generator_state'])
 arr=np.lib.format.open_memmap(dest,mode='r+' if dest.exists() else 'w+',dtype=np.complex64,shape=(N,10,64,64))
 fields=torch.zeros((10,64,512,512),device='cuda',dtype=torch.complex64)
 fields[:,:,torch.tensor(mask,device='cuda')]=torch.tensor(src['bases'].transpose(0,2,1),device='cuda',dtype=torch.complex64)
 started=time.time();initial=done
 for start in range(done,N,batch):
  bs=min(batch,N-start);screens=ev.screens(bs,gen);phase=torch.exp(1j*screens)
  for m in range(10):
   f=fields[m].unsqueeze(0).expand(bs,-1,-1,-1)
   for k in range(ev.config.wos_steps):f=ev.propagate(f*phase[:,k,None]*ev.grid_truncation)
   rec=f[:,:,ev.receiver_aperture.bool()]
   gram=rec.conj()@rec.transpose(1,2)
   arr[start:start+bs,m]=gram.cpu().numpy()
  if start==0:
   # Independent direct propagation of all original normalized endpoint fields.
   orig=torch.zeros((10,512,512),device='cuda',dtype=torch.complex64)
   orig[:,torch.tensor(mask,device='cuda')]=torch.tensor(src['original_fields'],device='cuda')
   f=orig.unsqueeze(0).expand(bs,-1,-1,-1)
   for k in range(ev.config.wos_steps):f=ev.propagate(f*phase[:,k,None]*ev.grid_truncation)
   direct=(f.abs().square()*ev.receiver_aperture).sum((-1,-2)).double().mean(1).cpu().numpy()
   predicted=100*np.einsum('mi,smij,mj->s',src['x'].conj(),arr[:bs].astype(np.complex128),src['x']).real
   error=float(np.max(abs(direct-predicted)/np.maximum(direct,1e-30)));assert error<3e-4,error
   save(folder/f'{kind}_direct_check.json',dict(relative_power_error=error,direct=direct.tolist(),quadratic=predicted.tolist(),seed=seed))
  arr.flush();torch.save(dict(completed=start+bs,generator_state=gen.get_state()),state)
  if (start+bs)%16==0 or start+bs==N:
   elapsed=time.time()-started;log(stage='responses',link=link,bank=kind,completed=start+bs,total=N,seed=seed,elapsed_seconds=elapsed,seconds_per_atmosphere=elapsed/max(1,start+bs-initial))
 del arr
 save(folder/f'{kind}_complete.json',dict(samples=N,seed=seed,shape=[N,10,64,64],sha256=digest(dest)))

def metric(x,Z,cross):
 overlaps=np.einsum('mi,mnij,nj->mn',x.conj(),cross,x);norm=float(np.sum(abs(overlaps)**2));G=np.empty((1260,1260))
 for m in range(10):
  for n in range(10):
   c=cross[m,n];k=Z[m].conj().T@c@Z[n]
   a=x[n].conj()@cross[n,m]@Z[m];b=x[m].conj()@c@Z[n]
   G[m*126:(m+1)*126,n*126:(n+1)*126]=np.real(np.outer(a,b)+overlaps[m,n]*k.conj())/norm
 return (G+G.T)/2

def full_hessian(R,x,Z,eta):
 N=len(R);J=np.empty((N,1260));p=np.zeros(N)
 # Linear derivatives at the center in per-frame power-normalized coordinates.
 for m in range(10):
  rr=np.asarray(R[:,m],dtype=np.complex128);rx=rr@x[m]
  p+=100*np.einsum('i,si->s',x[m].conj(),rx).real
  J[:,126*m:126*(m+1)]=200*np.real(rx@Z[m].conj())
 mu=p.mean();nu=np.mean(p*p);F=eta/mu+nu/mu**2-1
 mg=J.mean(0);ng=(2*p[:,None]*J).mean(0)
 w=2*p/(N*mu**2)-(2*nu/mu**3+eta/mu**2)/N;fg=J.T@w
 H=2/(N*mu**2)*(J.T@J)-2/mu**3*(np.outer(ng,mg)+np.outer(mg,ng))+(6*nu/mu**4+2*eta/mu**3)*np.outer(mg,mg)
 for m in range(10):
  rw=np.einsum('s,sij->ij',w,np.asarray(R[:,m],dtype=np.complex128));center=float((x[m].conj()@rw@x[m]).real)
  H[126*m:126*(m+1),126*m:126*(m+1)]+=200*(np.real(Z[m].conj().T@rw@Z[m])-center*np.eye(126))
 scale=5/np.log(10);H=scale*(H/F-np.outer(fg,fg)/F**2)
 return (H+H.T)/2,scale*fg/F,p

def modal_features(R,factors):
 Q=np.empty((len(R),10,3,3),dtype=np.float64)
 for m in range(10):Q[:,m]=np.einsum('ai,sij,bj->sab',factors[:,m].conj(),np.asarray(R[:,m],dtype=np.complex128),factors[:,m]).real
 return (Q+Q.transpose(0,1,3,2))/2
def chart_power(Q,S,a):
 t=np.r_[1.,a];den=np.einsum('a,mab,b->m',t,S,t)
 return 100*np.sum(np.einsum('a,smab,b->sm',t,Q,t)/den[None],axis=1)
def chart_state(factors,a):
 y=np.einsum('a,ami->mi',np.r_[1.,a],factors);return y/np.linalg.norm(y,axis=1,keepdims=True)
def csd_inner(a,b,cross):return float(np.sum(abs(np.einsum('mi,mnij,nj->mn',a.conj(),cross,b))**2))

def analyze(link,cfg):
 folder=HERE/link
 if (folder/'analysis_complete.json').exists():return
 log(stage='full_hessian',link=link)
 src=np.load(folder/'source.npz');x,Z,cross=src['x'],src['Z'],src['cross'];eta=cfg['links'][link]['duale_b']
 R=np.load(folder/'optimization_response.npy',mmap_mode='r');val=np.load(folder/'validation_response.npy',mmap_mode='r')
 G=metric(x,Z,cross);H,g,p=full_hessian(R,x,Z,eta)
 ge,gv=la.eigh(G);keep=ge>ge[-1]*1e-10
 # Numerical distinguishability threshold; not asserted to be an exact kernel.
 B=gv[:,keep]/np.sqrt(ge[keep]);ev,V=la.eigh(B.T@H@B);order=np.argsort(abs(ev));vectors=B@V[:,order]
 low=np.einsum('miq,amq->ami',Z,vectors[:,:2].T.reshape(2,10,126))
 high=np.einsum('miq,amq->ami',Z,vectors[:,-2:].T.reshape(2,10,126))
 np.savez_compressed(folder/'curvature.npz',H=H,G=G,gradient=g,metric_eigenvalues=ge,curvatures=ev,low_directions=low,high_directions=high,low_vectors=vectors[:,:2])
 rows=[];saved={};center=float(snr(chart_power(modal_features(val,x[None].repeat(3,0)),np.einsum('ami,bmi->mab',x[None].repeat(3,0).conj(),x[None].repeat(3,0)).real,[0,0]),eta))
 for label,D in [('low',low),('high_control',high)]:
  factors=np.concatenate([x[None],D]);S=np.einsum('ami,bmi->mab',factors.conj(),factors).real;Q=modal_features(R,factors);VQ=modal_features(val,factors)
  # Autodiff check of the two-dimensional restricted Hessian and gradient.
  a=torch.zeros(2,dtype=torch.float64,requires_grad=True);qt=torch.tensor(Q);st=torch.tensor(S)
  def objective(a):
   t=torch.cat([torch.ones(1,dtype=torch.float64),a]);pp=100*(torch.einsum('a,smab,b->sm',t,qt,t)/torch.einsum('a,mab,b->m',t,st,t)).sum(1);mu=pp.mean();return 5*torch.log10(eta/mu+pp.var(unbiased=False)/mu**2)
  hh=torch.autograd.functional.hessian(objective,a).numpy();vv=vectors[:,:2] if label=='low' else vectors[:,-2:]
  expected=vv.T@H@vv;err=float(np.linalg.norm(hh-expected)/max(1.,np.linalg.norm(expected)));assert err<1e-5,(link,err)
  saved[f'{label}_optimization_features']=Q;saved[f'{label}_validation_features']=VQ;saved[f'{label}_source_gram']=S
  for radius in [.02,.05,.1]:
   coords=np.array([(a,b) for a in np.linspace(-radius,radius,9) for b in np.linspace(-radius,radius,9)])
   powers=np.stack([chart_power(VQ,S,a) for a in coords],axis=1);scores=snr(powers,eta);dist=[]
   base=csd_inner(x,x,cross)
   for a in coords:
    y=chart_state(factors,a);norm=csd_inner(y,y,cross);inner=csd_inner(y,x,cross);dist.append(float(np.sqrt(max(0,norm+base-2*inner)/(norm+base))))
   assert np.isfinite(powers).all() and powers.min()>=0
   row=dict(link=link,plane=label,halfwidth=radius,center_snr_db=float(scores[40]),max_abs_snr_change_db=float(np.max(abs(scores-scores[40]))),max_csd_distance=max(dist),fraction_within_0p05_db=float(np.mean(abs(scores-scores[40])<.05)),restricted_hessian_relative_error=err)
   rows.append(row);saved[f'{label}_r{radius}_powers']=powers;saved[f'{label}_r{radius}_csd_distances']=np.array(dist)
 log(stage='neighbor_curvature',link=link)
 angles=[]
 # Repeat full physical-direction analysis at both +/-0.05 neighbors.
 for sign in [-1,1]:
  y=chart_state(np.concatenate([x[None],low]),[sign*.05,0]);zz=[]
  for m in range(10):
   comp=la.null_space(y[m].conj()[None]);zz.append(np.concatenate([comp,1j*comp],axis=1))
  zz=np.array(zz);gg=metric(y,zz,cross);hh,_,_=full_hessian(R,y,zz,eta);e,b=la.eigh(gg);k=e>e[-1]*1e-10;bb=b[:,k]/np.sqrt(e[k]);ee,v=la.eigh(bb.T@hh@bb)
  saved[f'neighbor_{sign}_curvatures']=ee
  angles.append(dict(sign=sign,numerically_resolved_directions=int(k.sum()),two_smallest_absolute_curvatures=np.sort(abs(ee))[:2].tolist()))
 np.savez_compressed(folder/'patches.npz',**saved)
 summary=dict(link=link,coordinate_directions_before_metric_filter=1260,numerically_resolved_csd_directions=int(keep.sum()),metric_cutoff_relative=1e-10,metric_counts={str(t):int(np.sum(ge>ge[-1]*t)) for t in [1e-8,1e-10,1e-12]},selected_curvatures=ev[order[:2]].tolist(),optimization_center_snr_db=float(snr(p,eta)),independent_validation_center_snr_db=center,neighbor_spectra=angles,rows=rows)
 save(folder/'analysis_complete.json',summary);log(stage='link_complete',**summary)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--links',nargs='+',default=list('ABCD'));args=ap.parse_args()
 global HERE
 if args.smoke:HERE=HERE/'smoke';HERE.mkdir(exist_ok=True)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=json.loads((ROOT/'data/multistart/configuration.json').read_text());N=16 if args.smoke else 4800;NV=16 if args.smoke else 1000
 protocol=dict(version=1,links=args.links,center='highest original validated SNR among1024 per link, all64HG M10',waists='all ten frozen at their original endpoint values',normalization='each frame P0=1000 separately; then average M10 powers',optimization_atmospheres=N,independent_validation_atmospheres=NV,seed=2026103101,validation_seed_offset=50000,link_seed_stride=100000,parameterization='per-frame orthonormal HG coordinates with radial and phase freedoms removed:1260 real coordinates; physical CSD metric numerical cutoff1e-10, sensitivity1e-8/1e-12',directions='two smallest absolute generalized negative-SNR Hessian curvatures; high-curvature controls',radii=[.02,.05,.1],grid=[9,9],neighbors=[-.05,.05],original_protocol=cfg,source_hashes={name:digest(ROOT/name) for name in ['sources.py','propagation.py']},proof='continuous geometry and SNR certificates are a subsequent stage; never infer them from81points')
 if (HERE/'protocol.json').exists():assert json.loads((HERE/'protocol.json').read_text())==protocol,'Protocol differs'
 else:save(HERE/'protocol.json',protocol)
 save(HERE/'process.json',dict(pid=os.getpid(),executable=sys.executable,argv=sys.argv,started_at=time.time()))
 for link in args.links:prepare(link,cfg)
 if args.prepare_only:return
 for link in args.links:
  response_bank(link,cfg,'optimization',N);response_bank(link,cfg,'validation',NV);analyze(link,cfg)
 save(HERE/'numerical_complete.json',dict(links=args.links,complete=True));log(stage='numerical_complete',links=args.links)

if __name__=='__main__':
 with threadpool_limits(limits=4):main()
