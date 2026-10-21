"""Audit archived Sobol screening powers and rebuild the structured benchmark."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
CASES = ('small_rx', 'long_rx20', 'strong_small_rx')
ARCHIVE = ROOT / 'logs/benchmark_sobol128_screening'

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def score(power):
    mean = power.mean(axis=-1)
    si = np.maximum((power**2).mean(axis=-1) / mean**2 - 1, 0)
    return 10 * np.log10(1 / np.sqrt(.2 / mean + si))

def audit():
    summaries = []
    for ci, case in enumerate(CASES):
        for family in ('CB', 'Pin', 'Sharp', 'IOPB', 'DAD_dark', 'DAD_antidark'):
            folder = ARCHIVE / case / family
            config = json.loads((folder / 'config.json').read_text())
            records = json.loads((folder / 'records.json').read_text())
            assert config['starts'] == 128 and config['steps'] == 300 and config['batch_size'] == 16
            assert [r['run'] for r in records] == list(range(128))
            with np.load(folder / 'screening_powers.npz') as data:
                powers = data['power']
            assert powers.shape == (128, 1000) and np.isfinite(powers).all() and (powers > 0).all()
            dad = family.startswith('DAD')
            key = 'screening_snr_db' if dad else 'selection_snr_db'
            if dad:
                assert all(r['training_batch_size'] == 16 for r in records)
            else:
                assert all(r['history'][-1]['step'] == 300 for r in records)
            scores = score(powers)
            assert np.allclose(scores, [r[key] for r in records], rtol=0, atol=1e-10)
            chosen = np.argsort(-scores, kind='stable')[:3]
            seed = 2026101901 if dad else 2026101901 + ci
            indices = np.random.default_rng(seed).integers(0, 1000, size=(2000, 1000))
            boot = np.mean([score(powers[i][indices]) for i in chosen], axis=0)
            lo, hi = np.quantile(boot, [.025, .975])
            saved = json.loads((folder / 'summary.json').read_text())
            center = float(scores[chosen].mean())
            assert abs(center - saved['mean_top_three_screening_snr_db']) < 1e-10
            assert np.allclose([lo, hi], [saved['ci95_low_db'], saved['ci95_high_db']], rtol=0, atol=1e-10)
            assert chosen.tolist() == [r['run'] for r in saved['top_three']]
            summaries.append(dict(link=f'L{ci+1}', case=case, method=family, starts=128, steps=300,
                training_batch_size=16, mode_count=51 if dad else 1, screening_samples=1000,
                additional_validation_samples=0, selected_runs=','.join(map(str, chosen)),
                snr_db=center, ci95_low_db=float(lo), ci95_high_db=float(hi), bootstrap_repeats=2000))
    return summaries

def main():
    summaries = audit()
    table = ROOT / 'data/controls/fig3_performance.csv'
    with table.open() as stream:
        rows = list(csv.DictReader(stream))
    branches = []
    for case in CASES:
        dad = [r for r in summaries if r['case'] == case and r['method'].startswith('DAD')]
        winner = max(dad, key=lambda r:r['snr_db'])
        for row in dad:
            branches.append(dict(**row, branch=row['method'].split('_')[1], selected_for_figure=row is winner))
        for record in [r for r in summaries if r['case'] == case and not r['method'].startswith('DAD')] + [winner]:
            method = 'DAD' if record['method'].startswith('DAD') else record['method']
            target = next(r for r in rows if r['case'] == case and r['method'] == method)
            target.update(snr_p_db=record['snr_db'], ci95_low_db=record['ci95_low_db'], ci95_high_db=record['ci95_high_db'])
    write_csv(table, rows)
    write_csv(ROOT / 'data/controls/dad_m51_branches.csv', branches)
    write_csv(ROOT / 'data/controls/coherent_sobol128_summary.csv', [r for r in summaries if not r['method'].startswith('DAD')])
    write_csv(ROOT / 'data/controls/benchmark_sobol128_summary.csv', summaries)
    report = dict(dad_trajectories=768, coherent_trajectories=1536, screening_samples_per_endpoint=1000,
                  training_batch_size=16, groups_verified=18, additional_validation_samples=0)
    (ROOT / 'results/benchmark_screening_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))

if __name__ == '__main__':
    main()
