"""Optimize DAD-51 from Sobol starts and report top-three screening SNRs."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy import special
import torch
from torch.utils.checkpoint import checkpoint
from propagation import UnifiedWOS
from coherent_multistart import configuration, CASES
from structured_beams import bessel_mode_orders

ROOT=Path(__file__).resolve().parent

class RadialBessel(torch.autograd.Function):
    @staticmethod
    def forward(ctx,beta,radius):
        x=float(beta.detach())*radius.detach().cpu().double().numpy()
        orders=np.arange(26)[:,None]
        values=special.jv(orders,x[None,:])
        derivative=special.jvp(orders,x[None,:])*radius.detach().cpu().double().numpy()[None,:]
        ctx.save_for_backward(torch.as_tensor(derivative,device=beta.device,dtype=beta.dtype))
        return torch.as_tensor(values,device=beta.device,dtype=beta.dtype)
    @staticmethod
    def backward(ctx,gradient):
        return (gradient*ctx.saved_tensors[0]).sum(),None

class DADSource(torch.nn.Module):
    def __init__(self,e,branch,point):
        super().__init__();self.e=e;self.branch=branch
        self.raw=torch.nn.Parameter(torch.logit(point.clamp(1e-5,1-1e-5)).to(e.device))
        self.mask=e.transmitter_aperture.bool()
        r=e.rr[self.mask]
        radius_ids=torch.round((r/e.config.delta)**2).long()
        unique,self.inverse=torch.unique(radius_ids,sorted=True,return_inverse=True)
        self.radius=e.config.delta*torch.sqrt(unique.float())
        self.orders=torch.tensor(bessel_mode_orders(51),device=e.device)
        phi=torch.atan2(e.yy[self.mask],e.xx[self.mask])
        self.azimuth=torch.exp(1j*self.orders[:,None]*phi[None,:])
        self.r2=r.square()
    def physical(self):
        q=torch.sigmoid(self.raw)
        return dict(transverse_wavenumber_rad_m=60+300*q[0],envelope_radius_m=self.e.config.transmitter_radius*torch.exp(np.log(1000)*q[1]),alpha=(-q[2] if self.branch=='dark' else q[2]))
    def forward(self):
        p=self.physical();radial=RadialBessel.apply(p['transverse_wavenumber_rad_m'],self.radius)
        radial=radial[self.orders.abs()][:,self.inverse]
        sign=torch.where(self.orders<0,(-1.)**self.orders.abs(),torch.ones_like(self.orders,dtype=torch.float32))
        f=radial*sign[:,None]*torch.exp(-self.r2/p['envelope_radius_m']**2)[None,:]
        # Rescale before normalizing to avoid underflow for high-order Bessel fields.
        f=f/f.abs().amax(dim=1,keepdim=True).detach().clamp_min(torch.finfo(f.dtype).tiny)
        f=f.to(self.e.complex_dtype)*self.azimuth
        fields=torch.zeros((51,*self.e.rr.shape),device=self.e.device,dtype=self.e.complex_dtype)
        fields[:,self.mask]=f
        fields=self.e.normalize_each_frame(fields)
        w=1+(-1.)**self.orders.abs()*p['alpha'];w=w/w.sum()
        return fields,w

def flux_with_screens(e,source,screens):
    field=source.unsqueeze(0).expand(screens.shape[0],-1,-1,-1)
    for step in range(e.config.wos_steps):
        field=field*torch.exp(1j*screens[:,None,step])*e.grid_truncation
        field=e.propagate(field)
    return (field.abs()*e.receiver_aperture).square().sum(dim=(-2,-1))

def train_loss(e,source,weights,generator,batch_size,chunk_size):
    pieces=[]
    for start in range(0,batch_size,chunk_size):
        screens=e.screens(min(chunk_size,batch_size-start),generator)
        pieces.append(checkpoint(lambda f,s:flux_with_screens(e,f,s),source,screens,use_reentrant=False))
    power=torch.cat(pieces).double()@weights.double()
    mean=power.mean();si=(power.square().mean()/mean.square()-1).clamp_min(0)
    loss=torch.sqrt(e.config.duale_b/mean+si)
    return loss,10*torch.log10(1/loss)

@torch.inference_mode()
def screen(e,source,weights,samples,batch,seed):
    generator=e.generator(seed);pieces=[]
    for start in range(0,samples,batch):
        flux=e.modal_flux_batch(source,min(batch,samples-start),generator)
        pieces.append((flux.double()@weights.double()).cpu())
    return torch.cat(pieces).numpy()

def snr(power,coefficient):
    mean=power.mean(axis=-1)
    si=np.maximum((power**2).mean(axis=-1)/mean**2-1,0)
    return 10*np.log10(1/np.sqrt(coefficient/mean+si))

def summarize(folder,e,records,args):
    chosen=sorted(records,key=lambda r:r['screening_snr_db'],reverse=True)[:3]
    powers=[np.load(folder/f"run_{r['run']:03d}_screening.npz")['power'] for r in chosen]
    indices=np.random.default_rng(args.seed).integers(0,args.screening_samples,size=(2000,args.screening_samples))
    boot=np.mean([snr(p[indices],e.config.duale_b) for p in powers],axis=0)
    low,high=np.quantile(boot,(.025,.975))
    # These intervals describe atmospheric resampling conditional on endpoint selection.
    batches={str(size):sum(r['training_batch_size']==size for r in records) for size in sorted({r['training_batch_size'] for r in records})}
    return dict(starts=args.starts,steps=args.steps,training_batch_size_counts=batches,screening_samples=args.screening_samples,additional_validation_samples=0,top_three=chosen,mean_top_three_screening_snr_db=float(np.mean([r['screening_snr_db'] for r in chosen])),ci95_low_db=float(low),ci95_high_db=float(high),interval='percentile bootstrap of selected screening powers, conditional on selection; not independent validation')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    parser.add_argument('--starts',type=int,default=128)
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--chunk-size',type=int,default=4)
    parser.add_argument('--lr',type=float,default=.02)
    parser.add_argument('--screening-samples',type=int,default=1000)
    parser.add_argument('--screening-batch',type=int,default=4)
    parser.add_argument('--seed',type=int,default=2026101901)
    parser.add_argument('--output',type=Path,default=ROOT/'work/dad_sobol128_screening')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    config={k:(str(v.resolve()) if isinstance(v,Path) else v) for k,v in vars(args).items()}
    path=args.output/'config.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous!=config:
            changed={k for k in set(previous)|set(config) if previous.get(k)!=config.get(k)}
            assert changed=={'batch_size'},'Only an explicitly recorded training batch change can resume existing runs'
            history=args.output/'configuration_history.json'
            transitions=json.loads(history.read_text()) if history.exists() else []
            transitions.append(dict(previous=previous,current=config,reason=f'Author requested {args.batch_size} atmospheres per update; incompatible trajectories are archived separately.'))
            history.write_text(json.dumps(transitions,indent=2))
            for completed in args.output.glob('*/*/run_*.json'):
                record=json.loads(completed.read_text())
                record.setdefault('training_batch_size',previous['batch_size'])
                completed.write_text(json.dumps(record,indent=2))
    path.write_text(json.dumps(config,indent=2))
    for case in args.cases:
        ci=CASES.index(case);e=UnifiedWOS(configuration(case));summaries={}
        for bi,branch in enumerate(('dark','antidark')):
            folder=args.output/case/branch;folder.mkdir(parents=True,exist_ok=True)
            points=torch.quasirandom.SobolEngine(3,scramble=True,seed=args.seed+ci*100+bi).draw(args.starts)
            records=[]
            for run,point in enumerate(points):
                target=folder/f'run_{run:03d}.json'
                if target.exists():
                    cached=json.loads(target.read_text())
                    assert cached['training_batch_size']==args.batch_size,'Move incompatible completed runs to a separate archive before resuming'
                    records.append(cached);continue
                t=time.time();model=DADSource(e,branch,point)
                optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
                generator=e.generator(args.seed+ci*1000000+bi*100000+run)
                for step in range(args.steps):
                    optimizer.zero_grad(set_to_none=True)
                    fields,weights=model()
                    loss,value=train_loss(e,fields,weights,generator,args.batch_size,args.chunk_size)
                    loss.backward()
                    if not torch.isfinite(loss) or not torch.isfinite(model.raw.grad).all():raise FloatingPointError((case,branch,run,step))
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                    if (step+1)%50==0:print(case,branch,f'run={run+1}/{args.starts}',f'step={step+1}/{args.steps}',flush=True)
                fields,weights=model()
                power=screen(e,fields,weights,args.screening_samples,args.screening_batch,args.seed+10000000+ci)
                record=dict(run=run,training_batch_size=args.batch_size,initial_sobol_point=point.tolist(),raw_endpoint=model.raw.detach().cpu().tolist(),parameters={k:float(v.detach()) for k,v in model.physical().items()},screening_snr_db=float(snr(power,e.config.duale_b)),seconds=time.time()-t)
                np.savez_compressed(folder/f'run_{run:03d}_screening.npz',power=power)
                target.write_text(json.dumps(record,indent=2));records.append(record)
                print(case,branch,f'{run+1}/{args.starts}',record['screening_snr_db'],f'{time.time()-t:.1f}s',flush=True)
                del model,fields,weights,loss
                torch.cuda.empty_cache()
            summary=summarize(folder,e,records,args)
            (folder/'summary.json').write_text(json.dumps(summary,indent=2));summaries[branch]=summary
        winner=max(summaries,key=lambda b:summaries[b]['mean_top_three_screening_snr_db'])
        (args.output/case/'summary.json').write_text(json.dumps(dict(branches=summaries,selected_branch=winner),indent=2))

if __name__=='__main__':main()
