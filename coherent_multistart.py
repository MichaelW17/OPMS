"""Sobol multistart Adam optimization of coherent benchmark source families."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy import special
from scipy.optimize import brentq
import torch
from propagation import LinkConfig, UnifiedWOS
from dad_pooled_validation import pooled_metrics

ROOT=Path(__file__).resolve().parent
CASES=('small_rx','long_rx20','strong_small_rx')
FAMILIES=('CB','Pin','Sharp','IOPB')

class Airy(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ai,derivative,_,_=special.airy(x.detach().cpu().double().numpy())
        ctx.save_for_backward(torch.as_tensor(derivative,device=x.device,dtype=x.dtype))
        return torch.as_tensor(ai,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx,gradient):
        return gradient*ctx.saved_tensors[0]

class BesselJ0(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        values=x.detach().cpu().double().numpy()
        ctx.save_for_backward(torch.as_tensor(-special.j1(values),device=x.device,dtype=x.dtype))
        return torch.as_tensor(special.j0(values),device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx,gradient):
        return gradient*ctx.saved_tensors[0]

class Source(torch.nn.Module):
    def __init__(self,evaluator,family,point):
        super().__init__()
        self.e=evaluator
        self.family=family
        self.raw=torch.nn.Parameter(torch.logit(point.clamp(1e-5,1-1e-5)).to(evaluator.device))
        self.mask=evaluator.transmitter_aperture.bool()
        self.radius=evaluator.rr[self.mask]
        self.gamma_min=.4
        if family=='IOPB':
            def constraint(gamma):
                return evaluator.k*evaluator.config.transmitter_radius**(2-gamma)*(3*evaluator.config.delta)**gamma/(gamma*2.404825557695773)-5*evaluator.config.distance
            if constraint(.4)>0:self.gamma_min=brentq(constraint,.4,.95)+1e-6

    def physical(self):
        q=torch.sigmoid(self.raw)
        if self.family=='CB':
            return dict(waist_m=.003+.097*q[0],normalized_curvature=2*q[1])
        if self.family=='Sharp':
            omega=.0075+.0125*q[0]
            # Match the three-pixel radial-scale restriction of the archived search.
            limit=torch.minimum(q.new_tensor(20), (omega/(3*self.e.config.delta))**2+self.e.config.transmitter_radius/omega)
            return dict(omega_m=omega,alpha=4*q[1],r0_over_omega=limit*q[2],normalized_curvature=2*q[3])
        radius=self.e.config.transmitter_radius
        envelope=radius*torch.exp(np.log(2000)*q[0])*.5
        if self.family=='Pin':
            return dict(envelope_radius_m=envelope,design_distance_m=self.e.config.distance*(.5+2.5*q[1]))
        gamma=self.gamma_min+(.95-self.gamma_min)*q[2]
        # Parameterize only profiles whose first Bessel zero is resolved on the grid.
        min_design=self.e.k*radius**(2-gamma)*(3*self.e.config.delta)**gamma/(gamma*2.404825557695773)
        lower=torch.maximum(min_design,q.new_tensor(.5*self.e.config.distance))
        upper=q.new_tensor(5*self.e.config.distance)
        if lower.detach()>upper:
            raise ValueError('No resolved IOPB profile for these bounds')
        return dict(envelope_radius_m=envelope,design_distance_m=lower+(upper-lower)*q[1],gamma=gamma)

    def forward(self):
        p=self.physical();r=self.radius;e=self.e
        if self.family=='CB':
            f=torch.exp(-r.square()/p['waist_m']**2).to(e.complex_dtype)
            f=f*torch.exp(-.5j*e.k*p['normalized_curvature']*r.square()/e.config.distance)
        elif self.family=='Sharp':
            x=p['r0_over_omega']-r/p['omega_m']
            # Subtracting a constant exponent preserves the normalized field.
            f=Airy.apply(x)*torch.exp(p['alpha']*(x-x.max().detach()))
            f=f.to(e.complex_dtype)*torch.exp(-.5j*e.k*p['normalized_curvature']*r.square()/e.config.distance)
        elif self.family=='Pin':
            beta=e.config.transmitter_radius/(4*p['design_distance_m']**2)
            f=torch.exp(-r.square()/p['envelope_radius_m']**2)*torch.exp(-4j/3*e.k*torch.sqrt(beta)*r**1.5)
        else:
            gamma=p['gamma']
            coefficient=e.config.transmitter_radius**(2-gamma)/(gamma*p['design_distance_m'])
            f=torch.exp(-r.square()/p['envelope_radius_m']**2)*BesselJ0.apply(e.k*coefficient*r**gamma)
        f=f/f.abs().max().detach().clamp_min(torch.finfo(e.real_dtype).tiny)
        field=torch.zeros_like(e.rr,dtype=e.complex_dtype)
        field[self.mask]=f.to(e.complex_dtype)
        return e.normalize_each_frame(field.unsqueeze(0))

def configuration(case):
    return LinkConfig(**json.loads((ROOT/f'logs/ol_sms_pcb_3param/{case}_M10/summary.json').read_text())['link'])

@torch.inference_mode()
def evaluate(e,source,samples,batch,seed):
    generator=e.generator(seed)
    return torch.cat([e.modal_flux_batch(source,min(batch,samples-start),generator).cpu() for start in range(0,samples,batch)]).flatten().double().numpy()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    parser.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES))
    parser.add_argument('--starts',type=int,default=128)
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--lr',type=float,default=.02)
    parser.add_argument('--selection-samples',type=int,default=1000)
    parser.add_argument('--validation-samples',type=int,default=2000)
    parser.add_argument('--validation-batch',type=int,default=4)
    parser.add_argument('--independent-validation',action='store_true',help='Optionally validate the selected endpoints on a second ensemble.')
    parser.add_argument('--seed',type=int,default=2026101901)
    parser.add_argument('--output',type=Path,default=ROOT/'work/coherent_sobol128_screening')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    settings={k:(str(v.resolve()) if isinstance(v,Path) else v) for k,v in vars(args).items()}
    config_path=args.output/'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text())==settings,'Use a fresh output directory when changing settings'
    config_path.write_text(json.dumps(settings,indent=2))
    started=time.time()
    for case in args.cases:
        e=UnifiedWOS(configuration(case));ci=CASES.index(case)
        for family in args.families:
            fi=FAMILIES.index(family);dim={'CB':2,'Pin':2,'Sharp':4,'IOPB':3}[family]
            points=torch.quasirandom.SobolEngine(dim,scramble=True,seed=args.seed+ci*100+fi).draw(args.starts)
            out=args.output/case/family;out.mkdir(parents=True,exist_ok=True)
            records=[]
            for run,point in enumerate(points):
                target=out/f'run_{run:03d}.json'
                if target.exists():
                    records.append(json.loads(target.read_text()));continue
                model=Source(e,family,point)
                optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
                generator=e.generator(args.seed+1000000*ci+100000*fi+run)
                t=time.time();history=[]
                for step in range(args.steps):
                    optimizer.zero_grad(set_to_none=True)
                    loss,snr,_,_=e.loss_from_source(model(),args.batch_size,generator)
                    if not torch.isfinite(loss):raise FloatingPointError((case,family,run,step))
                    loss.backward()
                    if not torch.isfinite(model.raw.grad).all():raise FloatingPointError('Nonfinite source gradient')
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                    optimizer.step()
                    if (step+1)%25==0:history.append(dict(step=step+1,training_snr_db=float(snr.detach())))
                power=evaluate(e,model(),args.selection_samples,args.validation_batch,args.seed+10000000+ci)
                mean=power.mean();si=max(np.mean(power**2)/mean**2-1,0)
                score=float(10*np.log10(1/np.sqrt(e.config.duale_b/mean+si)))
                record=dict(run=run,initial_sobol_point=point.tolist(),raw_endpoint=model.raw.detach().cpu().tolist(),parameters={k:float(v.detach()) for k,v in model.physical().items()},selection_snr_db=score,seconds=time.time()-t,history=history)
                np.savez_compressed(out/f'run_{run:03d}_selection.npz',power=power)
                target.write_text(json.dumps(record,indent=2));records.append(record)
                print(case,family,f'{run+1}/{args.starts}',f'selection={score:.5f} dB',f'{time.time()-t:.1f}s',flush=True)
            chosen=sorted(records,key=lambda r:r['selection_snr_db'],reverse=True)[:3]
            if not args.independent_validation:
                indices=np.random.default_rng(args.seed+ci).integers(0,args.selection_samples,size=(2000,args.selection_samples))
                scores=[]
                for record in chosen:
                    with np.load(out/f"run_{record['run']:03d}_selection.npz") as archive:power=archive['power']
                    sample=power[indices];mean=sample.mean(axis=1)
                    si=np.maximum(np.mean(sample**2,axis=1)/mean**2-1,0)
                    scores.append(10*np.log10(1/np.sqrt(e.config.duale_b/mean+si)))
                low,high=np.quantile(np.mean(scores,axis=0),(.025,.975))
                summary=dict(case=case,family=family,starts=args.starts,steps=args.steps,screening_samples=args.selection_samples,additional_validation_samples=0,top_three=chosen,mean_top_three_screening_snr_db=float(np.mean([v['selection_snr_db'] for v in chosen])),ci95_low_db=float(low),ci95_high_db=float(high),interval='percentile bootstrap of selected screening powers, conditional on selection')
                (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
                continue
            final=[]
            validation_powers=[]
            for record in chosen:
                model=Source(e,family,points[record['run']])
                with torch.no_grad():model.raw.copy_(torch.tensor(record['raw_endpoint'],device=e.device))
                power=evaluate(e,model(),args.validation_samples,args.validation_batch,args.seed+20000000+ci)
                validation_powers.append(power)
                metrics=pooled_metrics(power,e.config.duale_b,args.seed+ci,2000)
                np.savez_compressed(out/f"top_run_{record['run']:03d}_validation.npz",power=power)
                final.append(dict(run=record['run'],parameters=record['parameters'],selection_snr_db=record['selection_snr_db'],**metrics))
            summary=dict(case=case,family=family,starts=args.starts,steps=args.steps,selection='top three independently screened endpoints',top_three=final,mean_top_three_validated_snr_db=float(np.mean([v['snr_db'] for v in final])))
            # Resample common atmosphere indices across the three frozen endpoints.
            indices=np.random.default_rng(args.seed+ci).integers(0,args.validation_samples,size=(2000,args.validation_samples))
            bootstrap_scores=[]
            for power in validation_powers:
                samples=power[indices];means=samples.mean(axis=1)
                si=np.maximum(np.mean(samples**2,axis=1)/means**2-1,0)
                bootstrap_scores.append(10*np.log10(1/np.sqrt(e.config.duale_b/means+si)))
            low,high=np.quantile(np.mean(bootstrap_scores,axis=0),(.025,.975))
            summary.update(mean_top_three_ci95_low_db=float(low),mean_top_three_ci95_high_db=float(high),interval='percentile bootstrap of the mean of the selected endpoints, conditional on screening selection')
            (out/'summary.json').write_text(json.dumps(summary,indent=2))
            print(json.dumps(summary),flush=True)
    print('Completed in',time.time()-started,'seconds',flush=True)

if __name__=='__main__':main()
