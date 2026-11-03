"""Check supplementary tables against raw powers and repetition-level records."""
from pathlib import Path
import csv
import json
import numpy as np
import torch
from scipy import stats
from supplement_links import CASE_LABELS, CASE_ORDER

ROOT=Path(__file__).resolve().parent

def main():
    report={}
    with (ROOT/'data/controls/weighted_hg_comparison.csv').open() as f:
        for row in csv.DictReader(f):
            assert row['link']==CASE_LABELS[row['case']]
            m=int(row['mode_count'])
            sub='ol_weighted_hg_m64' if m==64 else 'ol_weighted_hg_pseudomode_validation_20260901'
            path=ROOT/f"logs/{sub}/{row['case']}_M10/weighted_hg_result.pt"
            d=torch.load(path,map_location='cpu',weights_only=False)
            flux=d['modal_flux'].double().numpy()
            w=d['weights'].double().numpy()
            assert flux.shape==(2000,m) and np.isclose(w.sum(),1)
            summary=d['summary']
            if m==64:
                expected=[[p,q] for p in range(8) for q in range(8)]
                assert summary['holdout']['mode_indices']==expected
                assert summary['optimization']['runs']==8
                assert summary['optimization']['steps']==300
                assert summary['optimization']['train_bs']==6
                assert summary['optimization']['mc_blocks']==2
                assert summary['selection']['samples']==128
                candidates=summary['all_candidates']
                assert len(candidates)==8
                for candidate in candidates:
                    assert candidate['history'][-1]['step']==300
                    assert candidate['selection_metrics']['mode_indices']==expected
                selected=int(np.argmax([c['selection_metrics']['snrp_legacy_db_10log'] for c in candidates]))
                assert selected==summary['selection']['selected_run']
                report[f"{row['link']}_wHG_M64_rectangular_modes_verified"]=True
            power=flux@w
            eta=summary['link']['duale_b']
            si=max(np.mean(power**2)/np.mean(power)**2-1,0)
            value=10*np.log10(1/np.sqrt(eta/np.mean(power)+si))
            error=abs(value-float(row['snr_db']))
            assert error<1e-9,(path,error)
            report[f"{row['link']}_wHG_M{m}_raw_power_error_db"]=error
            if m==64:
                seed=summary['random_streams']['bootstrap_seed']
                rng=np.random.default_rng(seed)
                ids=rng.integers(0,len(power),size=(2000,len(power)))
                samples=power[ids]
                mean=samples.mean(axis=1)
                variance_ratio=np.maximum(np.mean(samples**2,axis=1)/mean**2-1,0)
                bootstrap=10*np.log10(1/np.sqrt(eta/mean+variance_ratio))
                low,high=np.quantile(bootstrap,[.025,.975])
                interval_error=max(abs(low-float(row['ci95_low_db'])),abs(high-float(row['ci95_high_db'])))
                assert interval_error<1e-9,(path,interval_error)
                report[f"{row['link']}_wHG_M64_bootstrap_error_db"]=float(interval_error)
    with (ROOT/'data/controls/propagation_step_sensitivity.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==72
    max_error=0
    for row in rows:
        path=ROOT/f"logs/ol_s7_step_sensitivity/{row['case']}_{row['method']}_M{row['M']}_K{row['K']}.json"
        d=json.loads(path.read_text())
        v=np.asarray(d['per_repeat_snr_db'])
        assert len(v)==16 and d['atmospheres_per_repeat']==1000
        half=stats.t.ppf(.975,15)*stats.sem(v)
        for name,value in [('mean_snr_db',v.mean()),('ci95_low_db',v.mean()-half),('ci95_high_db',v.mean()+half)]:
            error=abs(value-float(row[name]));assert error<1e-10
            max_error=max(max_error,error)
    report['step_sensitivity_records']=len(rows)
    report['step_sensitivity_max_error_db']=max_error
    with (ROOT/'data/controls/dad_m51_branches.csv').open() as f:
        rows=list(csv.DictReader(f))
    for case in {r['case'] for r in rows}:
        group=[r for r in rows if r['case']==case]
        selected=[r for r in group if r['selected_for_figure']=='True']
        assert len(selected)==1
        assert float(selected[0]['snr_db'])==max(float(r['snr_db']) for r in group)
    from benchmark_screening import audit
    audited=audit()
    for row in rows:
        record=next(r for r in audited if r['case']==row['case'] and r['method']=='DAD_'+row['branch'])
        for key in ('snr_db','ci95_low_db','ci95_high_db'):
            assert abs(record[key]-float(row[key]))<1e-10
    report['sobol_screening_groups_verified']=len(audited)
    report['dad_branches_verified']=len(rows)
    report['supplement_link_order']=list(CASE_ORDER)
    (ROOT/'results/supplement_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
