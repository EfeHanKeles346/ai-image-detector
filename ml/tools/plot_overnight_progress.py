"""Render the measured overnight checkpoints; reads only committed aggregate evidence."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2];E=ROOT/'evidence'
CONDITIONS=['clean','assigned_transport','q75'];IDS=[64,67,68,69,70,71,73,74,76,77]


def main():
    paths=[E/f'e{i}_fit.json' for i in IDS]+[E/'e70_development.json',E/'e71_development.json']
    fits={i:json.loads((E/f'e{i}_fit.json').read_text()) for i in IDS}
    train=[[fits[64]['train_gate']['comparisons'][c]['real']['old_ai_rate'] for c in CONDITIONS]]
    for i in IDS:
        r=fits[i]
        values=[r['real_population_gates'][c]['rates']['old_real'] if i>=76 else r['train_gate']['comparisons'][c]['real']['new_ai_rate'] for c in CONDITIONS]
        train.append(values)
        assert r['train_gate']['new_ai_miss_views']==0
    train=np.asarray(train)*100;assert train.shape==(11,3)
    dev={i:json.loads((E/f'e{i}_development.json').read_text()) for i in [70,71]}
    real=[];miss=[]
    for experiment in [43,70,71]:
        rr=[];mm=[]
        for cond in ['publisher_original','social_q75']:
            report=dev[70 if experiment==43 else experiment]['reports'][cond]
            rr.append(report['old' if experiment==43 else 'new']['binary_rates']['pooled_real_false_ai']*100)
            mm.append(0 if experiment==43 else sum(x['new_ai_misses'] for x in report['transitions'].values()))
        real.append(rr);miss.append(mm)
    real=np.asarray(real);miss=np.asarray(miss)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','axes.labelcolor':'#334155','text.color':'#172033','xtick.color':'#334155','ytick.color':'#334155'})
    fig=plt.figure(figsize=(14,10),facecolor='#fafbfc');grid=fig.add_gridspec(2,2,height_ratios=[1.1,1],hspace=.46,wspace=.22)
    top=fig.add_subplot(grid[0,:]);left=fig.add_subplot(grid[1,0]);right=fig.add_subplot(grid[1,1])
    palette=['#2563a6','#e6a03a','#168877'];labels=['Temiz','Atanmış aktarım','TRAIN Q75']
    x=np.arange(len(train));width=.24
    for j in range(3):top.bar(x+(j-1)*width,train[:,j],width,color=palette[j],label=labels[j],zorder=3)
    top.axhline(10,color='#b43a3a',linestyle='--',linewidth=1.5,label='Hedef: en fazla %10',zorder=4)
    top.set_xticks(x,['E43']+[f'E{i}'+('*' if i>=76 else '') for i in IDS]);top.set_ylim(0,20)
    top.set_ylabel('Eski REAL grubunda yanlış AI alarmı (%)')
    top.set_title('Eğitim verisi • aynı 7.035 REAL görüntüde ölçüm',loc='left',pad=16)
    for idx in [5,6]:
        top.axvspan(idx-.44,idx+.44,color='#fae5e5',zorder=0)
        top.text(idx,18,'DEV\nelendi',ha='center',va='center',fontsize=9,color='#a12b38')
    top.legend(loc='upper left',ncol=4,frameon=False,bbox_to_anchor=(0,1.02))
    x=np.arange(3);width=.32
    for j,label in enumerate(['Orijinal','Sosyal Q75']):
        color=palette[0 if j==0 else 2]
        bars=left.bar(x+(j-.5)*width,real[:,j],width,color=color,label=label,zorder=3)
        left.bar_label(bars,labels=[f'{v:.1f}' for v in real[:,j]],padding=3,fontsize=10)
        bars=right.bar(x+(j-.5)*width,miss[:,j],width,color=color,label=label,zorder=3)
        right.bar_label(bars,labels=[str(v) for v in miss[:,j]],padding=3,fontsize=11)
    left.axhline(10,color='#b43a3a',linestyle='--',linewidth=1.5)
    left.set_title('Ayrı E66 geliştirme verisi • yanlış alarm',loc='left',pad=15)
    left.set_ylabel('REAL yanlış AI alarmı (%)');left.set_ylim(0,57)
    right.set_title('Aynı geliştirme verisi • yeni kaçırılan AI',loc='left',pad=15)
    right.set_ylabel('E43 yakalarken adayın kaçırdığı görüntü');right.set_ylim(0,5.5);right.set_yticks(range(6))
    right.text(.98,.95,'Kabul koşulu: 0 yeni kayıp',transform=right.transAxes,ha='right',va='top',color='#a12b38')
    for ax in [left,right]:ax.set_xticks(x,['E43 referans','E70','E71']);ax.legend(frameon=False,loc='upper left')
    for ax in [top,left,right]:ax.set_facecolor('white');ax.grid(axis='y',color='#dfe5ed',linewidth=.7,zorder=0);ax.set_axisbelow(True)
    fig.suptitle('Hedef henüz geçilmedi',x=.065,ha='left',y=.97,fontsize=23,fontweight='bold')
    fig.text(.065,.93,'Düşük eğitim hatası tek başına başarı değil. E70/E71 geliştirmede; E77 eğitim kontrolünde elendi.',fontsize=12,color='#475569')
    fig.subplots_adjust(left=.075,right=.975,top=.85,bottom=.16)
    fig.text(.075,.095,'* E76/E77 daha geniş TRAIN grubunda eğitildi; üst panel eski REAL grubunu sabit tutar. E73 ve sonrası tüm TRAIN AI skorlarını korur.',fontsize=9,color='#475569')
    fig.text(.075,.072,'E66: koşul başına 160 REAL (yalnızca 10 bağımlı sahne) + 160 AI. Daha önce kullanılmış geliştirme verisi; bağımsız final değildir.',fontsize=9,color='#475569')
    fig.text(.075,.049,'13 Eylül 2026 • E49 açılmadı • E43 ve servis değişmedi • E79 özellik çıkarımı sürüyor; E80 için henüz sonuç yok.',fontsize=9,color='#475569')
    outputs=[]
    for ext in ['png','svg']:
        path=E/f'overnight_progress_2026-09-13.{ext}';fig.savefig(path,dpi=150,facecolor=fig.get_facecolor());outputs.append(path)
    plt.close(fig)
    receipt={'state':'aggregate_evidence_plot','inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'outputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
             'new_model_scores':0,'e49_read':False,'train_real_rates_percent':train.tolist(),
             'dev_real_rates_percent':real.tolist(),'dev_new_ai_misses':miss.tolist()}
    (E/'overnight_progress_plot.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'outputs':[str(p) for p in outputs],'new_model_scores':0}))


if __name__=='__main__':main()
