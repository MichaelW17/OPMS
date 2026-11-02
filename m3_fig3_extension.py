"""M=3 GSM and 64-HG extension of the archived Fig. 3 protocol.

Reuse the original physical sources, Sobol initialization, optimizer and validation.
Each endpoint is validated immediately; no extra winner experiment is performed.
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import argparse
import hashlib
import json
import os
import sys
import time
import numpy as np
import torch
import multistart as study

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'work/fig3_m3_extension'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seeds(link, run, base):
    value = base + 10000000 * (list('ABCD').index(link) + 1) + 100000 * ord(link)
    return dict(link_seed=value, gsm_sobol=value+11, opms_sobol=value+22,
                training_atmosphere=value+100000*run+202,
                gsm_training_source=value+100000*run+101,
                opms_training_source=value+100000*run+111,
                validation_source=value+700000+301,
                validation_atmosphere=value+700000+401)


def independent_score(power, eta):
    p = power.double().numpy()
    assert np.isfinite(p).all() and (p >= 0).all()
    integrated = p.mean(axis=1)
    mean = integrated.mean()
    si = max(np.mean(integrated**2)/mean**2 - 1, 0.)
    return float(-5*np.log10(eta/mean+si))


def check_endpoint(endpoint, link, kind, run, args):
    for key, value in dict(link=link, kind=kind, run=run, steps=args.steps,
                           batch_size=16, frames=3).items():
        assert endpoint[key] == value, (key, endpoint[key], value)
    state = endpoint['model_state']
    assert all(torch.isfinite(v).all() for v in state.values())
    if kind == 'OPMS':
        for key, shape in [('raw_amplitudes',(3,64)),('raw_phases',(3,63)),('raw_waists',(3,))]:
            assert tuple(state[key].shape) == shape
        assert sum(v.numel() for v in state.values()) == 384
    else:
        assert sum(v.numel() for v in state.values()) == 3


def paths(out, link, kind, run):
    stem = out / link / kind.lower() / f'run_{run:03d}'
    return Path(str(stem)+'.pt'), Path(str(stem)+'_screening_flux.pt'), Path(str(stem)+'_screening.json'), Path(str(stem)+'_seeds.json')


def verify(out, args, links, kinds):
    rows=[]
    for link in links:
        for kind in kinds:
            for run in range(args.starts):
                ep, fp, mp, sp = paths(out,link,kind,run)
                check_endpoint(torch.load(ep,map_location='cpu',weights_only=False),link,kind,run,args)
                power=torch.load(fp,map_location='cpu',weights_only=False)
                assert tuple(power.shape)==(args.screening_samples,3)
                score=independent_score(power,study.link_table()[link].duale_b)
                assert abs(score-json.loads(mp.read_text())['snr_db']) < 1e-9
                assert json.loads(sp.read_text()) == seeds(link,run,args.seed)
                rows.append(dict(link=link,kind=kind,run=run,snr_db=score,
                                 sha256={p.name:sha(p) for p in (ep,fp,mp,sp)}))
            print(f'Verified {link}/{kind}: {args.starts} endpoints',flush=True)
    target=out/('verification.json' if set(links)==set('ABCD') and set(kinds)=={'GSM','OPMS'} else 'verification_subset.json')
    study.write_json(target,dict(verified_at=time.time(),records=rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--links',nargs='+',choices=list('ABCD'),default=list('ABCD'))
    parser.add_argument('--kinds',nargs='+',choices=['GSM','OPMS'],default=['GSM','OPMS'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--verify',action='store_true')
    options=parser.parse_args()
    torch.set_num_threads(4)
    study.N_SIDE=8
    archived=json.loads((ROOT/'data/multistart/configuration.json').read_text())
    args=SimpleNamespace(**archived['arguments'])
    args.frames=3
    args.output=options.output.resolve()
    if options.smoke:
        assert args.output != DEFAULT.resolve(), 'Smoke must use a separate folder'
        args.starts=2;args.steps=2;args.screening_samples=16
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  links=archived['links'],n_side=8,hg_modes=64,
                  parameter_counts=dict(OPMS=384,GSM=3),
                  source_files_sha256={name:sha(ROOT/name) for name in ['multistart.py','sources.py','propagation.py']},
                  validation='1000 unseen atmospheres per endpoint, common atmospheric seeds; independent GSM sequence per atmosphere; fixed OPMS sequence',
                  success='Own condition/link top100 mean SNR minus 0.5 dB',
                  additional_winner_validation=False,smoke=options.smoke)
    assert {k:asdict(v) for k,v in study.link_table().items()} == archived['links']
    config=args.output/'configuration.json'
    if config.exists():
        assert json.loads(config.read_text())==protocol,'Protocol/source mismatch; do not reuse incompatible results'
    else:
        study.write_json(config,protocol)
    if options.verify:
        verify(args.output,args,options.links,options.kinds)
        return
    study.write_json(args.output/'process.json',dict(pid=os.getpid(),executable=sys.executable,
                     argv=sys.argv,started_at=time.time(),links=options.links,kinds=options.kinds))
    total=args.starts*len(options.links)*len(options.kinds);completed=0
    try:
        for link in options.links:
            evaluator=study.UnifiedWOS(study.link_table()[link],device=args.device)
            link_seed=seeds(link,0,args.seed)['link_seed']
            pcb=study.pcb_sobol_points(args.starts,link_seed+11)
            opms=study.opms_sobol_points(args.starts,3,link_seed+22)
            assert opms.shape==(args.starts,384)
            for kind in options.kinds:
                for run in range(args.starts):
                    ep,fp,mp,sp=paths(args.output,link,kind,run)
                    state=dict(status='running',link=link,kind=kind,run=run,
                               completed=completed,total=total,updated_at=time.time(),stage='training')
                    study.write_json(args.output/'progress.json',state)
                    endpoint=study.train_candidate(kind,run,evaluator,pcb,opms,args,link_seed)
                    check_endpoint(endpoint,link,kind,run,args)
                    if not (fp.exists() and mp.exists()):
                        state.update(stage='validation',updated_at=time.time())
                        study.write_json(args.output/'progress.json',state)
                        metrics,power=study.validate_endpoint(kind,endpoint,evaluator,args.screening_samples,
                                      args,link_seed,link_seed+700000,bootstrap_repeats=400)
                        assert power.shape==(args.screening_samples,3)
                        assert abs(independent_score(power,evaluator.config.duale_b)-metrics['snr_db'])<1e-9
                        temporary=fp.with_suffix('.tmp.pt');torch.save(power,temporary);temporary.replace(fp)
                        study.write_json(mp,metrics)
                    power=torch.load(fp,map_location='cpu',weights_only=False)
                    assert power.shape==(args.screening_samples,3)
                    score=independent_score(power,evaluator.config.duale_b)
                    assert abs(score-json.loads(mp.read_text())['snr_db'])<1e-9
                    study.write_json(sp,seeds(link,run,args.seed))
                    completed+=1
                    print(f'{link}/{kind} {run+1}/{args.starts} SNR={score:.6f}',flush=True)
            del evaluator
            torch.cuda.empty_cache()
        verify(args.output,args,options.links,options.kinds)
        study.write_json(args.output/'progress.json',dict(status='complete',completed=completed,total=total,updated_at=time.time()))
    except Exception as exc:
        study.write_json(args.output/'progress.json',dict(status='error',completed=completed,total=total,
                         updated_at=time.time(),error=repr(exc)))
        raise


if __name__=='__main__':main()
