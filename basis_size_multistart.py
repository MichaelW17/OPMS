"""Repeat the archived OPMS multistart protocol with a smaller HG product basis."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multistart as study

ROOT = Path(__file__).resolve().parent

def analyze(output):
    rows=[]; distributions={}; endpoint_rows=[]
    for link in 'ABCD':
        files=sorted((output/link/'opms').glob('run_*_screening.json'))
        if len(files)!=1024:
            return False
        values=np.array([json.loads(p.read_text())['snr_db'] for p in files])
        frames=json.loads((output/'configuration.json').read_text())['arguments']['frames']
        for label,v in [(f'9 HG, M={frames}',values),('64 HG, M=10',np.array([r['snr_db'] for r in json.loads((ROOT/f'data/multistart/{link}_opms_metrics.json').read_text())]))]:
            reference=np.sort(v)[-100:].mean()
            reference64=np.mean(np.sort([r['snr_db'] for r in json.loads((ROOT/f'data/multistart/{link}_opms_metrics.json').read_text())])[-100:])
            rows.append(dict(link=link,basis=label,starts=1024,reference_snr_db=reference,threshold_snr_db=reference-.5,successes=int((v>=reference-.5).sum()),success_rate=float((v>=reference-.5).mean()),median_snr_db=float(np.median(v)),mean_snr_db=float(v.mean()),success_rate_against_64hg_reference=float((v>=reference64-.5).mean())))
            distributions[link,label]=v
            endpoint_rows.extend(dict(link=link,basis=label,run=i,validated_snr_db=float(value),successful=bool(value>=reference-.5),reference_snr_db=float(reference)) for i,value in enumerate(v))
    with (output/'success_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (output/'endpoint_scores.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(endpoint_rows[0]));writer.writeheader();writer.writerows(endpoint_rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for j,(label,color) in enumerate([(f'9 HG, M={frames}','#3478A8'),('64 HG, M=10','#D46A3A')]):
        positions=np.arange(4)+(j-.5)*.32
        violin=axes[0].violinplot([distributions[l,label] for l in 'ABCD'],positions=positions,widths=.28,showmedians=True,showextrema=False)
        for body in violin['bodies']:body.set_facecolor(color);body.set_alpha(.65)
        violin['cmedians'].set_color(color)
        axes[1].bar(positions,[next(r['success_rate'] for r in rows if r['link']==l and r['basis']==label)*100 for l in 'ABCD'],width=.28,color=color,label=label)
    for ax in axes:ax.set_xticks(range(4),list('ABCD'));ax.set_xlabel('Link');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Validated SNR (dB)');axes[1].set_ylabel('Success rate (%)');axes[1].set_ylim(0,105);axes[1].legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.02))
    for ax,label in zip(axes,['(a)','(b)']):ax.text(.02,.98,label,va='top',transform=ax.transAxes)
    for ext in ('png','pdf'):fig.savefig(output/f'basis9_vs64.{ext}',dpi=300)
    plt.close(fig)
    return True

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-side',type=int,default=3,choices=[3])
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--links',nargs='+',choices=list('ABCD'),default=list('ABCD'))
    parser.add_argument('--frames',type=int,choices=[3,10],default=10)
    parser.add_argument('--output',type=Path)
    options=parser.parse_args()
    archived=json.loads((ROOT/'data/multistart/configuration.json').read_text())
    args=SimpleNamespace(**archived['arguments'])
    args.frames=options.frames
    args.output=(options.output or ROOT/('work/opms_hg9_sobol1024' if args.frames==10 else 'work/opms_hg9_m3_sobol1024')).resolve();args.output.mkdir(parents=True,exist_ok=True)
    if options.smoke:
        args.starts=2;args.steps=2;args.screening_samples=16
    study.N_SIDE=options.n_side
    protocol=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},n_side=options.n_side,hg_modes=options.n_side**2,parameter_count=args.frames*2*options.n_side**2,links=archived['links'],success_definition='Own-family top-100 mean minus 0.5 dB; also report fraction reaching archived 64-HG reference minus 0.5 dB.',seed_definition='Same link/training/screening seed formulas as archived multistart.py; Sobol dimension changes with basis size.',additional_winner_validation=False)
    config=args.output/'configuration.json'
    if config.exists():assert json.loads(config.read_text())==protocol,'Incompatible experiment configuration'
    else:study.write_json(config,protocol)
    completed=0
    for link in options.links:
        position=list('ABCD').index(link)
        evaluator=study.UnifiedWOS(study.link_table()[link],device=args.device)
        link_seed=args.seed+10000000*(position+1)+100000*ord(link)
        pcb=study.pcb_sobol_points(args.starts,link_seed+11)
        points=study.opms_sobol_points(args.starts,args.frames,link_seed+22)
        assert points.shape==(args.starts,args.frames*18)
        for run in range(args.starts):
            folder=args.output/link/'opms';folder.mkdir(parents=True,exist_ok=True)
            metrics_file=folder/f'run_{run:03d}_screening.json'
            flux_file=folder/f'run_{run:03d}_screening_flux.pt'
            study.write_json(args.output/'progress.json',dict(status='running',link=link,run=run,completed=completed,total=len(options.links)*args.starts,stage='training',updated=time.time()))
            endpoint=study.train_candidate('OPMS',run,evaluator,pcb,points,args,link_seed)
            assert endpoint['model_state']['raw_amplitudes'].shape==(args.frames,9)
            if not (metrics_file.exists() and flux_file.exists()):
                study.write_json(args.output/'progress.json',dict(status='running',link=link,run=run,completed=completed,total=len(args.links)*args.starts,stage='endpoint validation',updated=time.time()))
                metrics,flux=study.validate_endpoint('OPMS',endpoint,evaluator,args.screening_samples,args,link_seed,link_seed+700000,bootstrap_repeats=400)
                assert flux.shape==(args.screening_samples,args.frames)
                torch.save(flux,flux_file);study.write_json(metrics_file,metrics)
            completed+=1
            print(link,f'{run+1}/{args.starts}',json.loads(metrics_file.read_text())['snr_db'],flush=True)
        del evaluator
        torch.cuda.empty_cache()
    if not options.smoke and set(options.links)==set('ABCD'):assert analyze(args.output)
    study.write_json(args.output/'progress.json',dict(status='complete',completed=completed,total=completed,smoke=options.smoke,updated=time.time()))

if __name__=='__main__':main()
