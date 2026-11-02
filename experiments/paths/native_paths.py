"""Paths in original 64HG/M10/frame-specific-waist family."""
import sys,json,time
from pathlib import Path
import numpy as np,torch
from scipy.optimize import linear_sum_assignment
P=Path(__file__).resolve().parent;ROOT=P.parents[1];sys.path.insert(0,str(ROOT))
from propagation import UnifiedWOS,LinkConfig
from sources import hermite_gaussian_1d,SMSParameters,synthesize_sms
def save(path,value): path.write_text(json.dumps(value,indent=2),encoding="utf-8")
def progress(**value): save(P/"progress.json",value)

def source(ev,c,raw_waist):
 shape=c.shape[:-2];m=c.shape[-2];c=c.reshape(-1,64);raw_waist=raw_waist.reshape(-1);w=.003+(.08-.003)*torch.sigmoid(raw_waist);one=hermite_gaussian_1d(ev,w,8).to(torch.complex64);field=torch.einsum('mij,mix,mjy->mxy',c.reshape(-1,8,8),one,one)*ev.curvature_phase(1.);field=ev.normalize_each_frame(field);return field.reshape(*shape,m,512,512)

def powers(ev,fields,screens):
 # one atmosphere shared by every pseudo-mode in each integration window
 f=fields.unsqueeze(0).expand(len(screens),-1,-1,-1)
 for k in range(5):f=ev.propagate(f*torch.exp(1j*screens[:,None,k])*ev.grid_truncation)
 return (f.abs().square()*ev.receiver_aperture).sum((-1,-2)).double().mean(1)

@torch.no_grad()
def evaluate_paths(ev,sets,seed,path):
 if path.exists():return
 gen=ev.generator(seed);out={label:[] for label in sets};T=33
 for s in range(1000):
  screens=ev.screens(1,gen)
  for label,(cs,ws) in sets.items():
   pp=[]
   for i in range(0,T,8):
    f=source(ev,cs[i:i+8],ws[i:i+8]);flat=f.flatten(0,1);result=powers_per_frame(ev,flat,screens)[0].reshape(-1,10).mean(1);pp.append(result.cpu().numpy())
   out[label].append(np.concatenate(pp))
  if s%100==0:progress(stage='native_validation',file=str(path),atmosphere=s)
 np.savez_compressed(path,**{label:np.array(v) for label,v in out.items()})

def powers_per_frame(ev,f,screens):
 f=f.unsqueeze(0).expand(len(screens),-1,-1,-1)
 for k in range(5):f=ev.propagate(f*torch.exp(1j*screens[:,None,k])*ev.grid_truncation)
 return (f.abs().square()*ev.receiver_aperture).sum((-1,-2)).double()

def main():
 torch.set_num_threads(4);cfg=json.loads((ROOT/'data/multistart/configuration.json').read_text());selection=json.loads((P/'selection.json').read_text())
 for li,link in enumerate('ABCD'):
  ev=UnifiedWOS(LinkConfig(**cfg['links'][link]));eps=torch.load(ROOT/f'data/multistart/{link}_opms_endpoints.pt',map_location='cpu');pairs=selection[link]['OPMS']['pairs']
  for pi,pair in enumerate(pairs):
   
   if pair['group'] != 'far': continue
   folder=P/'native_paths'/link/f'pair{pi}_{pair["group"]}';folder.mkdir(parents=True,exist_ok=True)
   if (folder/'complete.json').exists():continue
   models=[];fields=[];cs=[];ws=[]
   for key in ['run_i','run_j']:
    model=SMSParameters(10,8,0,.08,ev.device);model.load_state_dict(eps[pair[key]]['model_state']);models.append(model)
    with torch.no_grad():
     fields.append(synthesize_sms(ev,model));a,ph,w=model.physical();cc=torch.polar(a,ph);cc/=torch.linalg.vector_norm(cc,dim=-1,keepdim=True);cs.append(cc);ws.append(model.raw_waists.detach().clone())
   overlap=fields[0].flatten(1).conj()@fields[1].flatten(1).T;ii,jj=linear_sum_assignment(-overlap.abs().square().cpu().numpy());perm=torch.tensor(jj,device='cuda');phase=torch.exp(-1j*torch.angle(overlap[torch.arange(10,device='cuda'),perm]));a,b=cs[0],cs[1][perm]*phase[:,None];wa,wb=ws[0],ws[1][perm];ts=torch.linspace(0,1,33,device='cuda');linear_c=(1-ts[:,None,None])*a+ts[:,None,None]*b;linear_w=(1-ts[:,None])*wa+ts[:,None]*wb
   with torch.no_grad():
    check=source(ev,a,wa);err=float(torch.linalg.vector_norm(check-fields[0])/torch.linalg.vector_norm(fields[0]));assert err<2e-5
    checkb=source(ev,b,wb);reference=fields[1][perm]*phase[:,None,None];errb=float(torch.linalg.vector_norm(checkb-reference)/torch.linalg.vector_norm(reference));assert errb<2e-5
   checkpoint=folder/'bend.pt'
   if pair['group']=='far':
    if checkpoint.exists():
     state=torch.load(checkpoint,map_location='cuda');mid=state['c'];wmid=state['w']
    else:
     mid=torch.nn.Parameter(torch.view_as_real((a+b)/2).clone());wmid=torch.nn.Parameter(((wa+wb)/2).clone());opt=torch.optim.Adam([mid,wmid],lr=.02);gen=ev.generator(2026102591+li*100000+pi*1000);rng=np.random.default_rng(2026102593+li*100000+pi*1000)
     for step in range(150):
      opt.zero_grad();screens=ev.screens(16,gen)
      for t in rng.uniform(.05,.95,2):
       cc=(1-t)**2*a+2*t*(1-t)*torch.view_as_complex(mid)+t*t*b;ww=(1-t)**2*wa+2*t*(1-t)*wmid+t*t*wb;src=source(ev,cc,ww);p=powers(ev,src,screens);mu=p.mean();loss=torch.sqrt(ev.config.duale_b/mu+p.var(unbiased=False)/mu**2)/2;loss.backward()
      torch.nn.utils.clip_grad_norm_([mid,wmid],10);opt.step()
      if step%25==0:progress(stage='native_curve_train',link=link,pair=pi,step=step+1)
     mid=torch.view_as_complex(mid).detach();wmid=wmid.detach();torch.save(dict(c=mid.cpu(),w=wmid.cpu()),checkpoint)
    curve_c=(1-ts[:,None,None])**2*a+2*ts[:,None,None]*(1-ts[:,None,None])*mid+ts[:,None,None]**2*b;curve_w=(1-ts[:,None])**2*wa+2*ts[:,None]*(1-ts[:,None])*wmid+ts[:,None]**2*wb
   else:curve_c,curve_w=linear_c,linear_w
   torch.save(dict(endpoint_a=a.cpu(),endpoint_b=b.cpu(),waist_a=wa.cpu(),waist_b=wb.cpu(),linear_c=linear_c.cpu(),linear_w=linear_w.cpu(),curve_c=curve_c.cpu(),curve_w=curve_w.cpu(),t=ts.cpu()),folder/'path_parameters.pt');save(folder/'definition.json',dict(**pair,frame_permutation=jj.tolist(),initial_field_relative_error=err,end_field_relative_error=errb,train_steps=150 if pair['group']=='far' else 0))
   evaluate_paths(ev,dict(straight=(linear_c,linear_w),curve=(curve_c,curve_w)),2026102592+li*100000,folder/'validation_powers.npz');save(folder/'complete.json',dict(complete=True))
 save(P/'native_complete.json',dict(complete=True))
if __name__=='__main__':main()
