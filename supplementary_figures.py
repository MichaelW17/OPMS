"""Draw the three supplementary figures from the released numerical tables."""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent
CASES = ('small_rx', 'long_rx20', 'strong_small_rx')
COLORS = ('#286890', '#c56a36', '#38866b')
MARKERS = 'os^'

def read(name):
    with (ROOT / 'data/controls' / name).open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))

def save(fig, output, name):
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'{name}.{ext}', dpi=400, bbox_inches='tight')
    plt.close(fig)

def draw(output):
    output = Path(output)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False})
    rows = read('fig5_frame_budget.csv')
    fig, ax = plt.subplots(figsize=(5.24, 3.0), layout='constrained')
    for j, case in enumerate(CASES):
        for method, style in [('SMS', '-'), ('GSM', '--')]:
            group = sorted((r for r in rows if r['case'] == case and r['method'] == method), key=lambda r:int(r['frame_count']))
            ax.plot([int(r['frame_count']) for r in group], [float(r['snr_p_db']) for r in group], style, marker=MARKERS[j], ms=4, color=COLORS[j])
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 4, 10, 20, 51], ['1', '2', '4', '10', '20', '51'])
    ax.set_xlabel(r'Modal truncation order, $M$')
    ax.set_ylabel('Validated SNR (dB)')
    ax.grid(alpha=.18)
    handles = [Line2D([0],[0],color=COLORS[j],marker=MARKERS[j],ls=style,ms=4,label=rf'$\mathit{{L}}{j+1}$ {method}') for method, style in [('OPMS','-'),('GSM','--')] for j in range(3)]
    ax.legend(handles=handles,frameon=False,ncol=2,loc='lower right',handlelength=3.5,columnspacing=1.1,fontsize=8)
    save(fig,output,'figS1_modal_truncation')

    rows=read('fig6_basis_performance.csv')
    fig,ax=plt.subplots(figsize=(4.7,2.5),layout='constrained')
    for j,r in enumerate(rows):
        v=float(r['lg_minus_hg_snr_p_db'])
        ax.errorbar(j,v,yerr=[[v-float(r['ci95_low_db'])],[float(r['ci95_high_db'])-v]],fmt='o',ms=5,color=COLORS[j],capsize=4)
    ax.axhline(0,color='.45',lw=.8,ls='--')
    ax.set_xticks(range(3),[rf'$\mathit{{L}}{j+1}$' for j in range(3)])
    ax.set_xlim(-.6,2.6)
    ax.set_ylabel('LG − HG validated SNR (dB)')
    ax.grid(axis='y',alpha=.2)
    save(fig,output,'figS2_basis_comparison')

    rows=read('fig3_performance.csv')
    methods=['SMS','GSM','CB','Sharp','Pin','IOPB','DAD']
    fig,ax=plt.subplots(figsize=(5.24,3.05),layout='constrained')
    for j,case in enumerate(CASES):
        group=[next(r for r in rows if r['case']==case and r['method']==m) for m in methods]
        v=np.array([float(r['snr_p_db']) for r in group])
        lo=np.array([float(r['ci95_low_db']) for r in group])
        hi=np.array([float(r['ci95_high_db']) for r in group])
        ax.errorbar(np.arange(7)+(j-1)*.18,v,yerr=[v-lo,hi-v],fmt=MARKERS[j],ms=4,capsize=2,color=COLORS[j])
    ax.set_xticks(range(7),['OPMS' if m=='SMS' else m for m in methods])
    ax.set_xlabel('Beam type')
    ax.set_ylabel('Validated SNR (dB)')
    ax.grid(axis='y',alpha=.2)
    handles=[Line2D([0],[0],color=COLORS[j],marker=MARKERS[j],ls='None',ms=5,label=rf'$\mathit{{L}}{j+1}$') for j in range(3)]
    ax.legend(handles=handles,frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.01),handlelength=0,columnspacing=1.5)
    save(fig,output,'figS3_structured_beams')

    # wHG remains a numerical comparison in the text, rather than a plotted family.
    for name in ('weighted_hg_comparison.csv','dad_m51_branches.csv','propagation_step_sensitivity.csv'):
        (output/name).write_bytes((ROOT/'data/controls'/name).read_bytes())

