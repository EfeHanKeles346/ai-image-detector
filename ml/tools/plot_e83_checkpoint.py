"""Plot frozen E83 TRAIN and consumed DEVELOPMENT aggregate results only."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];E=ROOT/'evidence'
IDS=[71,77,80,81,83];CONDITIONS=['clean','assigned_transport','q75']

def main():
    paths=[E/f'e{i}_fit.json' for i in IDS]+[E/f'e{i}_development.json' for i in [70,71,83]]
    reports={p.name:json.loads(p.read_text()) for p in paths}
    train_real=[];train_ai=[]
    for i in [43,*IDS]:
        r=reports[f'e{71 if i==43 else i}_fit.json'];rr=[];aa=[]
        for c in CONDITIONS:
            comparison=r['train_gate']['comparisons'][c]
            rr.append(comparison['real']['old_ai_rate'] if i==43 else
                r['real_population_gates'][c]['rates']['old_real'] if i>=77 else comparison['real']['new_ai_rate'])
            aa.append(comparison['ai']['old_ai_rate'] if i==43 else comparison['ai']['new_ai_rate'])
        train_real.append(rr);train_ai.append(aa)
    dev_real=[];dev_ai=[];dev_miss=[]
    for i in [43,70,71,83]:
        d=reports[f'e{70 if i==43 else i}_development.json'];rr=[];aa=[];mm=[]
        for c in ['publisher_original','social_q75']:
            cell=d['reports'][c];rates=cell['old' if i==43 else 'new']['binary_rates']
            rr.append(rates['pooled_real_false_ai']);aa.append(rates['pooled_ai_recall'])
            mm.append(0 if i==43 else sum(v['new_ai_misses'] for v in cell['transitions'].values()))
        dev_real.append(rr);dev_ai.append(aa);dev_miss.append(mm)
    values=[np.asarray(v)*100 for v in [train_real,train_ai,dev_real,dev_ai]]
    assert all(np.isfinite(v).all() and np.all((v>=0)&(v<=100)) for v in values)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,10),facecolor='#fafbfc')
    palette=['#2563a6','#e6a03a','#168877']
    titles=['TRAIN: gerçek fotoğrafta yanlış AI alarmı','TRAIN: AI yakalama oranı',
            'Ayrı geliştirme: gerçek fotoğrafta yanlış alarm','Ayrı geliştirme: AI yakalama oranı']
    for k,(ax,v,title) in enumerate(zip(axes.flat,values,titles)):
        n=v.shape[1];x=np.arange(len(v));width=.23 if n==3 else .30
        labels=['Temiz','Atanmış aktarım','TRAIN Q75'] if k<2 else ['Orijinal','Sosyal Q75']
        for j in range(n):
            bars=ax.bar(x+(j-(n-1)/2)*width,v[:,j],width,color=palette[j if n==3 else j*2],label=labels[j],zorder=3)
            if k!=1:ax.bar_label(bars,labels=[f'{a:.2f}' if k==0 and a<1 else f'{a:.1f}' for a in v[:,j]],padding=3,fontsize=9)
        ax.set_xticks(x,['E43',*[f'E{i}' for i in IDS]] if k<2 else ['E43','E70','E71','E83'])
        ax.set_title(title,loc='left',fontweight='bold',pad=18);ax.set_ylabel('%')
        ax.grid(axis='y',color='#e2e8f0',zorder=0);ax.set_axisbelow(True);ax.set_facecolor('white')
        ax.legend(frameon=False,ncol=n,fontsize=9,loc='upper left',bbox_to_anchor=(0,1.02))
        if k%2==0:
            ax.axhline(10,color='#b43a3a',linestyle='--',linewidth=1.2)
            ax.set_ylim(0,max(20,float(v.max())+10))
        else:ax.set_ylim(0,112)
    for idx,counts in enumerate(dev_miss):
        axes[1,1].text(idx,8,f'Yeni kayıp: {counts[0]}/{counts[1]}',ha='center',fontsize=8,color='#9f2937')
    passed=reports['e83_development.json']['passes_limited_dev_screen']
    fig.suptitle('E83: eğitim geçti • geliştirme '+('geçti' if passed else 'geçmedi'),x=.075,ha='left',fontsize=22,fontweight='bold',y=.97)
    fig.text(.075,.925,'Aynı karar eşikleri. AI koruması yeni kayıp sayısıyla da denetlenir; toplam oran tek başına yeterli değil.',fontsize=11,color='#475569')
    fig.subplots_adjust(left=.075,right=.975,top=.85,bottom=.16,hspace=.42,wspace=.22)
    fig.text(.075,.10,'TRAIN karşılaştırması aynı 7.035 REAL / 4.595 AI üzerinde; E77 ve sonrası ayrıca 511 MIDD REAL ile eğitildi.',fontsize=9,color='#475569')
    fig.text(.075,.077,'E83 eğitim sonuçları öğrenilmiş veriye aittir. E66: koşul başına 160 REAL (10 bağımlı sahne) + 160 AI; tüketilmiş geliştirme.',fontsize=9,color='#475569')
    fig.text(.075,.054,'13 Eylül 2026 • Bağımsız final sonucu değildir • E49 açılmadı • E43 ve servis değişmedi.',fontsize=9,color='#475569')
    outputs=[]
    for ext in ['png','svg']:
        path=E/f'e83_progress_2026-09-13.{ext}';fig.savefig(path,dpi=150,facecolor=fig.get_facecolor());outputs.append(path)
    plt.close(fig)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    result={'state':'E83_aggregate_checkpoint_plot','inputs':{str(p.relative_to(ROOT)):sha(p) for p in paths},
        'script_sha256':sha(Path(__file__)),'outputs':{str(p.relative_to(ROOT)):sha(p) for p in outputs},
        'train_real_rates':train_real,'train_ai_rates':train_ai,'dev_real_rates':dev_real,'dev_ai_rates':dev_ai,
        'dev_new_ai_misses':dev_miss,'new_model_scores':0,'e49_read':False}
    (E/'e83_progress_plot.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'outputs':[str(p) for p in outputs]}))

if __name__=='__main__':main()
