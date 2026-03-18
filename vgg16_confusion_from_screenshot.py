import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

OUTPUT      = "/Users/akshxunfiltered/DermoXAI/dermograph_output/"
MODEL_NAME  = "vgg16"
CLASS_NAMES = ['Melanoma','Nevi','Basal Cell Carcinoma',
               'Actinic Keratosis','Benign Keratosis',
               'Dermatofibroma','Vascular']

# ── Reconstruct confusion matrix from screenshot ──────────────────────
# Rows = Actual, Cols = Predicted
# Values from confusion matrix screenshot (percentages × support)
support = [112, 731, 133, 125, 133, 9, 12]

# CM percentages from screenshot (row by row)
cm_pct = np.array([
    [58.9, 24.1,  3.6,  1.8, 11.6,  0.0,  0.0],  # Melanoma
    [ 4.7, 88.2,  1.8,  0.8,  4.4,  0.0,  0.1],  # Nevi
    [ 0.8,  0.8, 75.9, 18.8,  2.3,  0.8,  0.8],  # BCC
    [ 0.8,  1.6, 15.2, 73.6,  8.8,  0.0,  0.0],  # Actinic Keratosis
    [12.0,  9.8,  2.3,  6.8, 69.2,  0.0,  0.0],  # Benign Keratosis
    [ 0.0, 22.2, 33.3,  0.0,  0.0, 44.4,  0.0],  # Dermatofibroma
    [ 8.3,  0.0,  8.3,  0.0,  0.0,  0.0, 83.3],  # Vascular
])

# Convert percentages → counts
cm = np.zeros((7,7), dtype=int)
for i in range(7):
    for j in range(7):
        cm[i,j] = round(cm_pct[i,j] / 100.0 * support[i])

# Fix rounding so rows sum to support
for i in range(7):
    diff = support[i] - cm[i].sum()
    if diff != 0:
        cm[i, i] += diff  # add/subtract from diagonal

test_acc = 0.8048
test_f1  = 0.7102
test_auc = 0.9601

# ── PLOT 1: Percentage confusion matrix ───────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor('#0a0e1a')
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='YlOrRd',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
            linewidths=0.5, linecolor='#1e2d4a',
            annot_kws={'size':10, 'weight':'bold'})
ax.set_facecolor('#0f1525')
ax.set_xlabel('Predicted Label', color='white', fontsize=12)
ax.set_ylabel('True Label',      color='white', fontsize=12)
ax.set_title(f'VGG16 — Confusion Matrix (%)\n'
             f'Acc={test_acc*100:.2f}%  |  F1={test_f1:.4f}  |  AUC={test_auc:.4f}',
             color='white', fontsize=13, fontweight='bold', pad=15)
plt.setp(ax.get_xticklabels(), rotation=35, ha='right', color='white', fontsize=9)
plt.setp(ax.get_yticklabels(), rotation=0,  color='white', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT}vgg16_confusion_matrix_pct.png',
            dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
plt.show()
print("✓ Saved: vgg16_confusion_matrix_pct.png")

# ── PLOT 2: Count matrix with ✓/✗ ────────────────────────────────────
annot = np.empty_like(cm, dtype=object)
for i in range(7):
    for j in range(7):
        count = cm[i, j]
        pct   = cm_pct[i, j]
        if i == j:
            annot[i,j] = f'✓ {count}\n({pct:.1f}%)'
        else:
            annot[i,j] = f'✗ {count}\n({pct:.1f}%)' if count > 0 else '0'

fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor('#0a0e1a')
sns.heatmap(cm, annot=annot, fmt='', cmap='RdYlGn',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
            linewidths=0.5, linecolor='#1e2d4a',
            annot_kws={'size':9, 'weight':'bold'})
ax.set_facecolor('#0f1525')
ax.set_xlabel('Predicted Label', color='white', fontsize=12)
ax.set_ylabel('True Label',      color='white', fontsize=12)
ax.set_title(f'VGG16 — Confusion Matrix (Counts)\n'
             f'✓ = Correct  |  ✗ = Misclassified  |  Total = {sum(support):,}',
             color='white', fontsize=13, fontweight='bold', pad=15)
plt.setp(ax.get_xticklabels(), rotation=35, ha='right', color='white', fontsize=9)
plt.setp(ax.get_yticklabels(), rotation=0,  color='white', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT}vgg16_confusion_matrix_counts.png',
            dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
plt.show()
print("✓ Saved: vgg16_confusion_matrix_counts.png")

# ── PLOT 3: TP/FP/FN breakdown ────────────────────────────────────────
TP = cm.diagonal()
FP = cm.sum(axis=0) - TP
FN = cm.sum(axis=1) - TP
TN = cm.sum() - (TP + FP + FN)

precision_per = np.where((TP+FP)>0, TP/(TP+FP), 0)
recall_per    = np.where((TP+FN)>0, TP/(TP+FN), 0)
f1_per        = np.where((precision_per+recall_per)>0,
                          2*precision_per*recall_per/(precision_per+recall_per), 0)

x     = np.arange(7)
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.patch.set_facecolor('#0a0e1a')
for ax in axes:
    ax.set_facecolor('#0f1525')
    ax.spines[['top','right','left','bottom']].set_color('#1e2d4a')
    ax.tick_params(colors='#94a3b8')

# TP / FP / FN counts
axes[0].bar(x-width, TP, width, label='TP (Correct)',     color='#22c55e', edgecolor='white', lw=0.5)
axes[0].bar(x,       FP, width, label='FP (False Alarm)', color='#ffc849', edgecolor='white', lw=0.5)
axes[0].bar(x+width, FN, width, label='FN (Missed)',      color='#FF4444', edgecolor='white', lw=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels(CLASS_NAMES, rotation=35, ha='right', color='white', fontsize=8)
axes[0].set_title('TP / FP / FN per Class', color='white', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count', color='#94a3b8')
axes[0].legend(facecolor='#0f1525', edgecolor='#1e2d4a', labelcolor='white')
for i,(tp,fp,fn) in enumerate(zip(TP,FP,FN)):
    axes[0].text(i-width, tp+1, str(tp), ha='center', color='white', fontsize=8, fontweight='bold')
    axes[0].text(i,       fp+1, str(fp), ha='center', color='white', fontsize=8, fontweight='bold')
    axes[0].text(i+width, fn+1, str(fn), ha='center', color='white', fontsize=8, fontweight='bold')

# Precision / Recall / F1
axes[1].bar(x-width, precision_per, width, label='Precision', color='#00e5cc', edgecolor='white', lw=0.5)
axes[1].bar(x,       recall_per,    width, label='Recall',    color='#a855f7', edgecolor='white', lw=0.5)
axes[1].bar(x+width, f1_per,        width, label='F1',        color='#ffc849', edgecolor='white', lw=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(CLASS_NAMES, rotation=35, ha='right', color='white', fontsize=8)
axes[1].set_title('Precision / Recall / F1 per Class', color='white', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Score', color='#94a3b8')
axes[1].set_ylim(0, 1.15)
axes[1].legend(facecolor='#0f1525', edgecolor='#1e2d4a', labelcolor='white')
for i,(p,r,f) in enumerate(zip(precision_per,recall_per,f1_per)):
    axes[1].text(i-width, p+0.02, f'{p:.2f}', ha='center', color='white', fontsize=8, fontweight='bold')
    axes[1].text(i,       r+0.02, f'{r:.2f}', ha='center', color='white', fontsize=8, fontweight='bold')
    axes[1].text(i+width, f+0.02, f'{f:.2f}', ha='center', color='white', fontsize=8, fontweight='bold')

plt.suptitle('VGG16 — Per-Class TP/FP/FN & Precision/Recall/F1',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT}vgg16_tp_fp_fn.png',
            dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
plt.show()
print("✓ Saved: vgg16_tp_fp_fn.png")

# ── Print TP/FP/FN table ──────────────────────────────────────────────
print(f"\n{'Class':<25} {'Sup':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>5} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("-"*72)
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:<25} {support[i]:>4} {TP[i]:>4} {FP[i]:>4} {FN[i]:>4} {TN[i]:>5} "
          f"{precision_per[i]:>6.3f} {recall_per[i]:>6.3f} {f1_per[i]:>6.3f}")
print("-"*72)
print(f"\n✓ All 3 plots saved to: {OUTPUT}")
