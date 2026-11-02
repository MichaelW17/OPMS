"""Independent numerical audit and compact figure; no continuous-proof claim."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_native_local import snr,chart_state,csd_inner

def main():
 cfg=json.loads((HERE/'protocol.json').read_text());rows=[];grid=[];audit=[]
 for link in 'ABCD':
  folder=HERE/link;done=json.loads((folder/'analysis_complete.json').read_text());src=np.load(folder/'source.npz');curv=np.load(folder/'curvature.npz');patch=np.load(folder/'patches.npz');selection=json.loads((folder/'selection.json').read_text())
  x=src['x'];D=curv['low_directions'];fac=np.concatenate([x[None],D]);cross=src['cross'];eta=cfg['original_protocol']['links'][link]['duale_b']
  radius=.1;coords=np.array([(a,b) for a in np.linspace(-radius,radius,9) for b in np.linspace(-radius,radius,9)])
  Y=np.array([chart_state(fac,a) for a in coords]);power=np.zeros((1000,81));R=np.load(folder/'validation_response.npy',mmap_mode='r')
  # Recompute from full 64x64 response matrices, rather than small-chart features.
  for m in range(10):
   rr=np.asarray(R[:,m],dtype=np.complex128)
   ry=rr@Y[:,m].T
   power+=100*np.einsum('ki,sik->sk',Y[:,m].conj(),ry).real
  archived=patch['low_r0.1_powers'];err=float(np.max(abs(power-archived)/np.maximum(abs(power),1e-30)));assert err<1e-10,err
  scores=snr(power,eta);delta=scores-scores[40];base=csd_inner(x,x,cross);dist=[]
  for y in Y:
   norm=csd_inner(y,y,cross);inner=csd_inner(y,x,cross);dist.append(np.sqrt(max(0,norm+base-2*inner)/(norm+base)))
  assert np.max(abs(np.array(dist)-patch['low_r0.1_csd_distances']))<1e-10
  for i,((u,v),s,d) in enumerate(zip(coords,scores,dist)):
   grid.append(dict(link=link,u=float(u),v=float(v),snr_db=float(s),delta_snr_db=float(s-scores[40]),csd_distance=float(d)))
  rows.append(dict(link=link,original_endpoint_run=selection['run'],original_validated_snr_db=selection['original_validated_snr_db'],new_validation_center_snr_db=float(scores[40]),max_abs_snr_change_db=float(max(abs(delta))),max_csd_distance=float(max(dist)),resolved_local_directions=done['numerically_resolved_csd_directions'],geometric_certificate='pending',continuous_snr_certificate='pending'))
  audit.append(dict(link=link,recomputed_power_max_relative_error=err,validation_samples=1000,positions=81,optimization_response_sha256=json.loads((folder/'optimization_complete.json').read_text())['sha256'],validation_response_sha256=json.loads((folder/'validation_complete.json').read_text())['sha256']))
 out=HERE/'analysis';out.mkdir(exist_ok=True)
 for name,data in [('summary.csv',rows),('grid.csv',grid)]:
  with (out/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 (out/'independent_numerical_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
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
 text=['# 正文原始OPMS终点附近的局部结构','', '每条链路使用正文64HG M10的1024个终点中原validated SNR最高的一个，不重新优化中心。十帧waist分别保持原值；每帧分别归一化到功率1000，随后取十帧平均。','', '两个方向由新4800个优化用大气的完整Hessian及实际CSD变化尺度确定。独立的新1000个大气用于网格评价，不参与选择中心或方向。','', '| 链路 | 原终点编号（从0计） | 新验证中心SNR dB | 81点最大绝对变化 dB | 最大CSD距离 |','|---|---:|---:|---:|---:|']
 for r in rows:text.append(f"| {r['link']} | {r['original_endpoint_run']} | {r['new_validation_center_snr_db']:.4f} | {r['max_abs_snr_change_db']:.4f} | {r['max_csd_distance']:.4f} |")
 text+=['','以上是半宽0.1的9×9网格结果。它反映实际光源变化和性能变化的幅度，不将近零曲率自动等同于驻点或严格最优解。','', '几何与连续SNR认证须另查本目录的认证记录。在新证明完成前，仅称离散数值证据；不得移用共同35mm waist实验的认证或0.05dB结论。','', '控制方向的有限CSD距离可能与小曲率方向不同，不将相同坐标幅度下的大曲率对照称为匹配有限CSD距离的公平对照。','']
 (out/'report_zh.md').write_text('\n'.join(text),encoding='utf-8');print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=4):main()
