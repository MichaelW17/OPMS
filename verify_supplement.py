"""Check supplementary tables against raw powers and repetition-level records."""
from pathlib import Path
import csv
import json
import numpy as np
import torch
from scipy import stats

ROOT=Path(__file__).resolve().parent

def main():
    report={}
    with (ROOT/'data/controls/weighted_hg_comparison.csv').open() as f:
        for row in csv.DictReader(f):
            m=int(row['mode_count'])
            sub='ol_weighted_hg_m64' if m==64 else 'ol_weighted_hg_pseudomode_validation_20260901'
            path=ROOT/f"logs/{sub}/{row['case']}_M10/weighted_hg_result.pt"
            d=torch.load(path,map_location='cpu',weights_only=False)
            flux=d['modal_flux'].double().numpy()
            w=d['weights'].double().numpy()
            assert flux.shape==(2000,m) and np.isclose(w.sum(),1)
            power=flux@w
            si=max(np.mean(power**2)/np.mean(power)**2-1,0)
            value=10*np.log10(1/np.sqrt(.2/np.mean(power)+si))
            error=abs(value-float(row['snr_db']))
            assert error<1e-9,(path,error)
            report[f"{row['link']}_wHG_M{m}_raw_power_error_db"]=error
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
    from dad_pooled_validation import pooled_metrics
    for row in rows:
        path=ROOT/f"logs/ol_dad_m51_pooled_validation/{row['case']}_{row['branch']}"
        record=json.loads(path.with_suffix('.json').read_text())
        with np.load(str(path)+'_powers.npz') as archive:
            flux=archive['modal_flux'];weights=archive['weights'];power=archive['integrated_power']
            assert flux.shape==(2000,51)
            assert np.allclose(flux@weights,power,rtol=0,atol=1e-10)
        metrics=pooled_metrics(power,.2,record['bootstrap_seed'],record['bootstrap_repeats'])
        for key in ('snr_db','ci95_low_db','ci95_high_db'):
            assert abs(metrics[key]-float(row[key]))<1e-10
    report['dad_branches_verified']=len(rows)
    (ROOT/'results/supplement_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
