"""Validate the selected DAD-51 branches using pooled powers and bootstrap CIs."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import torch
from propagation import LinkConfig, UnifiedWOS
from structured_beams import build_structured_source

ROOT=Path(__file__).resolve().parent
CASES=('small_rx','long_rx20','strong_small_rx')

def pooled_metrics(power, coefficient, seed, repeats):
    def snr(samples):
        mean=samples.mean(axis=-1)
        si=np.maximum(np.mean(samples**2,axis=-1)/mean**2-1,0)
        return 10*np.log10(1/np.sqrt(coefficient/mean+si))
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(power),size=(repeats,len(power)))
    low,high=np.quantile(snr(power[indices]),(.025,.975))
    return dict(snr_db=float(snr(power)),ci95_low_db=float(low),ci95_high_db=float(high),mean_power=float(power.mean()),scintillation_index=float(np.mean(power**2)/power.mean()**2-1))

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=2000)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--bootstrap-repeats',type=int,default=2000)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    out=ROOT/'logs/ol_dad_m51_pooled_validation'
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for ci,case in enumerate(CASES):
        archived=json.loads((ROOT/f'logs/ol_dad_sms_m51/{case}_M51/m51_comparison.json').read_text())
        common=json.loads((ROOT/f'logs/ol_all_beams_pooled_validation/{case}_M10/all_beams_pooled.json').read_text())
        seed=common['screen_seed_common']
        evaluator=UnifiedWOS(LinkConfig(**archived['link']))
        for branch in ('dark','antidark'):
            target=out/f'{case}_{branch}.json'
            raw_path=out/f'{case}_{branch}_powers.npz'
            if target.exists() and raw_path.exists() and not args.force:
                record=json.loads(target.read_text())
                assert record['atmospheric_samples']==args.samples
                assert record['batch_size']==args.batch_size
                assert record['bootstrap_repeats']==args.bootstrap_repeats
            else:
                candidate=archived[branch]['selected']['candidate']
                source,weights=build_structured_source(evaluator,candidate)
                generator=evaluator.generator(seed)
                chunks=[]
                for start in range(0,args.samples,args.batch_size):
                    size=min(args.batch_size,args.samples-start)
                    chunks.append(evaluator.modal_flux_batch(source,size,generator).cpu())
                    if (start+size)%200==0:print(case,branch,start+size,'/',args.samples,flush=True)
                flux=torch.cat(chunks).double().numpy()
                weights=weights.double().numpy()
                power=flux@weights
                np.savez_compressed(raw_path,modal_flux=flux,weights=weights,integrated_power=power)
                record=dict(case=case,link=f'L{ci+1}',branch=branch,mode_count=51,candidate=candidate,atmospheric_samples=args.samples,batch_size=args.batch_size,screen_seed=seed,bootstrap_repeats=args.bootstrap_repeats,bootstrap_seed=2026101801+ci,estimator='SNR from pooled linear powers; percentile bootstrap interval',**pooled_metrics(power,evaluator.config.duale_b,2026101801+ci,args.bootstrap_repeats))
                target.write_text(json.dumps(record,indent=2)+'\n')
                del source
                torch.cuda.empty_cache()
            rows.append(record)
            print(case,branch,record['snr_db'],record['ci95_low_db'],record['ci95_high_db'],flush=True)
    for case in CASES:
        group=[r for r in rows if r['case']==case]
        winner=max(group,key=lambda r:r['snr_db'])
        for r in group:r['selected_for_figure']=r is winner
    fields=('link','case','branch','mode_count','snr_db','ci95_low_db','ci95_high_db','atmospheric_samples','bootstrap_repeats','selected_for_figure')
    with (ROOT/'data/controls/dad_m51_branches.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
    path=ROOT/'data/controls/fig3_performance.csv'
    with path.open() as f:table=list(csv.DictReader(f))
    for r in table:
        if r['method']=='DAD':
            selected=next(v for v in rows if v['case']==r['case'] and v['selected_for_figure'])
            r.update(snr_p_db=selected['snr_db'],ci95_low_db=selected['ci95_low_db'],ci95_high_db=selected['ci95_high_db'])
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)

if __name__=='__main__':main()
