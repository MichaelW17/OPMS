from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def main():
 grid=[]
 cfg=json.loads((HERE/'protocol.json').read_text())
 for link in 'ABCD':
  power=np.load(HERE/link/'patches.npz')['low_r0.1_powers'];mu=power.mean(0);eta=cfg['original_protocol']['links'][link]['duale_b'];scores=-5*np.log10(eta/mu+power.var(0)/mu**2)
  grid.extend(dict(link=link,delta_snr_db=float(x-scores[40])) for x in scores)
 out=ROOT/'results';out.mkdir(exist_ok=True)
 plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'axes.linewidth':.6,'pdf.fonttype':42})
 fig=plt.figure(figsize=(3.4,3.30));side=1.08;gap=.18;left=.64;bottom=.46
 colorlimit=max(abs(float(r['delta_snr_db'])) for r in grid);colorlimit=max(.01,float(np.ceil(colorlimit/.01)*.01))
 for i,link in enumerate('ABCD'):
  row,col=divmod(i,2);ax=fig.add_axes([(left+col*(side+gap))/3.4,(bottom+(1-row)*(side+gap))/3.30,side/3.4,side/3.30])
  vals=np.array([r['delta_snr_db'] for r in grid if r['link']==link]).reshape(9,9).T
  edges=np.linspace(-.1125,.1125,10);im=ax.pcolormesh(edges,edges,vals,cmap='RdBu_r',vmin=-colorlimit,vmax=colorlimit,shading='flat',rasterized=True)
  ax.set_aspect('equal');ax.set_xticks([-.1,0,.1]);ax.set_yticks([-.1,0,.1]);ax.tick_params(labelbottom=row==1,labelleft=col==0,length=2,pad=2);ax.text(.035,.97,link,transform=ax.transAxes,va='top',fontsize=11)
 cax=fig.add_axes([left/3.4,2.99/3.30,(2*side+gap)/3.4,.06/3.30]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-colorlimit,0,colorlimit]);cb.ax.tick_params(length=2,pad=1);cb.ax.xaxis.set_label_position('top');cb.set_label('SNR change from center (dB)',labelpad=3,fontsize=11)
 fig.text((left+side+gap/2)/3.4,.11/3.30,r'Beam parameter coordinate $u$',ha='center',fontsize=11);fig.text(.08/3.4,(bottom+side+gap/2)/3.30,r'Beam parameter coordinate $v$',va='center',rotation=90,fontsize=11)
 for ext in ['png','pdf']:fig.savefig(out/f'Fig5_native_local_manifolds.{ext}',dpi=600)
 plt.close(fig)

if __name__=='__main__':main()
