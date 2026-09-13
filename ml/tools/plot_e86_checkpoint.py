"""Plot complete, already locked consumed DEV reports; no image or model reads."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'evidence'
CONDITIONS = ['publisher_original', 'social_q75']


def main():
    paths = [EVIDENCE / f'e{i}_development.json' for i in [83, 86]]
    reports = [json.loads(p.read_text()) for p in paths]
    cells = [[reports[0]['reports'][c]['old'] for c in CONDITIONS]]
    cells += [[r['reports'][c]['new'] for c in CONDITIONS] for r in reports]
    misses = [[0, 0]] + [[sum(v['new_ai_misses'] for v in r['reports'][c]['transitions'].values())
                           for c in CONDITIONS] for r in reports]
    rates = [np.asarray([[v['binary_rates'][key] * 100 for v in row] for row in cells])
             for key in ['pooled_real_false_ai', 'pooled_ai_recall']]
    if any(not np.isfinite(v).all() or np.any((v < 0) | (v > 100)) for v in rates):
        raise ValueError('finite complete rate reports required')
    for row in cells:
        for cell in row:
            counts = cell['binary_metrics']['counts']
            if counts != {'total': 320, 'succeeded': 320, 'failed': 0,
                          'real_succeeded': 160, 'ai_succeeded': 160}:
                raise ValueError('complete matched 320-view condition required')
    outputs = [EVIDENCE / f'e86_progress_2026-09-13.{ext}' for ext in ['png', 'svg']]
    receipt = EVIDENCE / 'e86_progress_plot.json'
    if any(p.exists() for p in [*outputs, receipt]):
        raise FileExistsError('preserve existing E86 plot artifacts')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), facecolor='#fafbfc')
    palette = ['#2563a6', '#168877']
    for i, (ax, values) in enumerate(zip(axes, rates)):
        for j, label in enumerate(['Orijinal', 'Sosyal Q75']):
            bars = ax.bar(np.arange(3) + (j - .5) * .32, values[:, j], .32,
                          label=label, color=palette[j], zorder=3)
            ax.bar_label(bars, labels=[f'{v:.2f}' for v in values[:, j]], padding=4, fontsize=10)
        ax.set_xticks(np.arange(3), ['E43 referans', 'E83', 'E86'])
        ax.set_ylabel('%'); ax.set_facecolor('white')
        ax.set_axisbelow(True); ax.grid(axis='y', color='#e2e8f0')
        ax.legend(frameon=False, ncol=2, loc='upper left', bbox_to_anchor=(0, 1.03))
        ax.set_title(['Gerçek fotoğrafta yanlış AI alarmı', 'AI yakalama oranı'][i],
                     loc='left', fontweight='bold', pad=28)
        ax.set_ylim(0, max(55, float(values.max()) + 12) if i == 0 else 112)
        if i == 0:
            ax.axhline(10, color='#b43a3a', linestyle='--', linewidth=1.2)
            ax.text(2.45, 11, 'Hedef ≤%10', ha='right', color='#9f2937', fontsize=9)
        else:
            for k, pair in enumerate(misses):
                ax.text(k, 8, f'Yeni AI kaybı\n{pair[0]} / {pair[1]}', ha='center',
                        fontsize=9, color='#9f2937',
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .95})
    passed = reports[1]['passes_limited_dev_screen']
    fig.suptitle('E86: tüketilmiş geliştirme ' + ('geçti' if passed else 'geçmedi'),
                 x=.075, y=.96, ha='left', fontsize=22, fontweight='bold')
    fig.text(.075, .895, 'Aynı 160 gerçek + 160 AI fotoğraf; aynı karar eşikleri. Her fotoğrafın iki koşulu karşılaştırılıyor.',
             fontsize=11, color='#475569')
    fig.subplots_adjust(left=.075, right=.975, top=.77, bottom=.27, wspace=.23)
    checks = reports[1]['checks']
    status = ' · '.join(f'{label}: sayısal {"geçti" if checks[c]["absolute_gates_passed"] else "kaldı"}, '
                        f'AI koruması {"geçti" if checks[c]["zero_new_ai_misses"] else "kaldı"}'
                        for label, c in zip(['Orijinal', 'Q75'], CONDITIONS))
    fig.text(.075, .175, status, fontsize=10, color='#334155')
    fig.text(.075, .13, 'Yeni kayıp: E43’ün yakaladığı, adayın kaçırdığı AI sayısı (orijinal / Q75). Kurtarılan diğer AI bunu telafi etmez.',
             fontsize=9, color='#475569')
    fig.text(.075, .09, 'SIDD gerçekleri 10 bağımlı sahneden gelir. Bu E66 karşılaştırması tüketilmiştir; bağımsız final sonucu değildir.',
             fontsize=9, color='#475569')
    for path in outputs:
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result = {'state': 'E86_consumed_DEV_aggregate_plot',
              'inputs': {str(p.relative_to(ROOT)): sha(p) for p in paths},
              'script_sha256': sha(Path(__file__)),
              'outputs': {str(p.relative_to(ROOT)): sha(p) for p in outputs},
              'real_FPR_percent': rates[0].tolist(), 'ai_recall_percent': rates[1].tolist(),
              'new_ai_misses_vs_E43': misses, 'conditions': CONDITIONS,
              'new_model_scores': 0, 'image_reads': 0, 'independent_final': False}
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'outputs': [str(p) for p in outputs]}))


if __name__ == '__main__':
    main()
