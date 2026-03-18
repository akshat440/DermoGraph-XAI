"""
DermoGraph-XAI | Script 04
ABCDE CLINICAL RISK SCORING ENGINE
Uses hmnist pixel data to compute real asymmetry & color scores.
Outputs: 04_abcde_risk_analysis.png, 04_abcde_scores.json
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
import json, os, warnings
warnings.filterwarnings('ignore')

INPUT_META = "HAM10000_metadata.csv"
INPUT_PIX  = "hmnist_28_28_L.csv"   # 28×28 grayscale pixel data
OUTPUT_DIR = "dermograph_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FULL_NAMES = {
    'nv':'Melanocytic Nevi','mel':'Melanoma','bkl':'Benign Keratosis',
    'bcc':'Basal Cell Carcinoma','akiec':'Actinic Keratosis',
    'vasc':'Vascular Lesion','df':'Dermatofibroma'
}
COLORS = {
    'nv':'#00e5c8','mel':'#ff4d6d','bkl':'#ffc23e',
    'bcc':'#7c5cfc','akiec':'#ff8c42','vasc':'#4ecdc4','df':'#a8dadc'
}
CLASS_MAP = {0:'akiec',1:'bcc',2:'bkl',3:'df',4:'mel',5:'nv',6:'vasc'}
MALIGNANT = {'mel','bcc','akiec'}

print("📦 Loading pixel data (28×28 grayscale)...")
pix = pd.read_csv(INPUT_PIX)
meta = pd.read_csv(INPUT_META)

label_col = 'label' if 'label' in pix.columns else pix.columns[-1]
pixel_cols = [c for c in pix.columns if c != label_col]
print(f"   Pixel data shape: {pix.shape} | Classes: {pix[label_col].nunique()}")

# ─── ABCDE SCORE ENGINE ───────────────────────────────────────
def compute_asymmetry(img):
    """A: IoU between image and its vertical/horizontal flips"""
    img_h = np.fliplr(img)
    img_v = np.flipud(img)
    threshold = img.mean()
    mask  = img  > threshold
    mask_h= img_h> threshold
    mask_v= img_v> threshold
    inter_h = (mask & mask_h).sum()
    union_h = (mask | mask_h).sum()
    inter_v = (mask & mask_v).sum()
    union_v = (mask | mask_v).sum()
    iou_h = inter_h / union_h if union_h > 0 else 1.0
    iou_v = inter_v / union_v if union_v > 0 else 1.0
    asym = min(2, round((1 - iou_h) * 2 + (1 - iou_v)))
    return int(asym), round(1 - iou_h, 3), round(1 - iou_v, 3)

def compute_border(img):
    """B: Compactness — high for irregular borders"""
    threshold = img.mean()
    mask = (img > threshold).astype(np.uint8)
    area = mask.sum()
    if area == 0: return 0, 0.0
    # approximate perimeter via gradient edges
    grad_x = np.abs(np.diff(mask, axis=1, prepend=mask[:,:1]))
    grad_y = np.abs(np.diff(mask, axis=0, prepend=mask[:1,:]))
    perimeter = (grad_x + grad_y).sum()
    compactness = (perimeter**2) / (4 * np.pi * area) if area > 0 else 1.0
    B = min(2, max(0, int(compactness - 1)))
    return int(B), round(compactness, 3)

def compute_color_variance(img):
    """C: Color heterogeneity via std of pixel values in lesion region"""
    threshold = img.mean()
    lesion = img[img > threshold]
    if len(lesion) == 0: return 0, 0.0
    std = lesion.std()
    # normalize to 0-2 range
    C = min(2, int(std / 30))
    return int(C), round(float(std), 3)

def compute_diameter_proxy(img):
    """D: If lesion area > 40% of image, flag as large"""
    threshold = img.mean()
    mask = img > threshold
    area_pct = mask.sum() / img.size
    D = 1 if area_pct > 0.40 else 0
    return int(D), round(float(area_pct), 3)

# ─── COMPUTE SCORES PER SAMPLE ───────────────────────────────
print("🔬 Computing ABCDE scores for all samples...")
records = []
sample_size = min(2000, len(pix))  # use up to 2000 samples for speed
pix_sample = pix.sample(n=sample_size, random_state=42).reset_index(drop=True)

for idx, row in pix_sample.iterrows():
    label_id = int(row[label_col])
    dx = CLASS_MAP.get(label_id, 'nv')
    pixels = row[pixel_cols].values.astype(np.float32).reshape(28, 28)

    A, asym_h, asym_v = compute_asymmetry(pixels)
    B, compact = compute_border(pixels)
    C, color_std = compute_color_variance(pixels)
    D, area_pct = compute_diameter_proxy(pixels)
    E = 0  # no temporal data in hmnist; placeholder
    total = A + B + C + D + E

    records.append({
        'dx': dx,
        'A_score': A, 'B_score': B, 'C_score': C,
        'D_score': D, 'E_score': E, 'total_score': total,
        'asymmetry_h': asym_h, 'asymmetry_v': asym_v,
        'compactness': compact, 'color_std': color_std,
        'area_pct': area_pct,
        'malignant': dx in MALIGNANT
    })
    if idx % 400 == 0:
        print(f"   Processed {idx}/{sample_size}...")

scores_df = pd.DataFrame(records)
print(f"✅ Scored {len(scores_df)} lesion images")

# ─── PLOT ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 15), facecolor='#020409')
gs  = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38,
               left=0.06, right=0.97, top=0.88, bottom=0.07)

ax_tot  = fig.add_subplot(gs[0, :2])
ax_rad  = fig.add_subplot(gs[0, 2])
ax_a    = fig.add_subplot(gs[1, 0])
ax_b    = fig.add_subplot(gs[1, 1])
ax_susp = fig.add_subplot(gs[1, 2])

for ax in [ax_tot, ax_rad, ax_a, ax_b, ax_susp]:
    ax.set_facecolor('#080d14')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a2535')

fig.suptitle('DermoGraph-XAI  ·  ABCDE Clinical Risk Score Engine',
             fontsize=22, fontweight='bold', color='#e8f0fe',
             fontfamily='monospace', y=0.95)
fig.text(0.5, 0.915,
         f'Computed from {sample_size} real pixel samples (hmnist_28×28_L)  ·  A=Asymmetry  B=Border  C=Color  D=Diameter',
         ha='center', fontsize=10.5, color='#8097b5', fontfamily='monospace')

# ─── TOTAL SCORE VIOLIN/BOX per class ────────────────────────
dx_order = sorted(scores_df['dx'].unique(),
                  key=lambda d: scores_df[scores_df['dx']==d]['total_score'].mean())
positions = range(1, len(dx_order)+1)
bp = ax_tot.boxplot(
    [scores_df[scores_df['dx']==dx]['total_score'].values for dx in dx_order],
    patch_artist=True, vert=True,
    medianprops=dict(color='#020409', linewidth=2.5),
    whiskerprops=dict(color='#8097b5', linewidth=1.2),
    capprops=dict(color='#8097b5', linewidth=1.2),
    flierprops=dict(marker='+', markersize=3, markerfacecolor='#8097b5', alpha=0.5)
)
for patch, dx in zip(bp['boxes'], dx_order):
    patch.set_facecolor(COLORS[dx]); patch.set_alpha(0.8)

# scatter jitter overlay
for i, dx in enumerate(dx_order, 1):
    vals = scores_df[scores_df['dx']==dx]['total_score'].values
    jitter = np.random.uniform(-0.18, 0.18, len(vals))
    ax_tot.scatter(i + jitter, vals, color=COLORS[dx], alpha=0.3,
                   s=8, edgecolors='none', zorder=3)

ax_tot.set_xticks(list(positions))
ax_tot.set_xticklabels([FULL_NAMES[dx] for dx in dx_order],
                       rotation=15, ha='right', fontsize=9, color='#8097b5')
ax_tot.set_title('ABCDE Total Score Distribution per Class (Box + Scatter)',
                 color='#e8f0fe', fontsize=13, fontweight='bold', pad=12)
ax_tot.set_ylabel('ABCDE Total Score (0–8)', color='#8097b5', fontsize=10)
ax_tot.tick_params(colors='#8097b5', labelsize=9)
ax_tot.grid(axis='y', color='#1a2535', linewidth=0.7, linestyle='--')
ax_tot.set_axisbelow(True)
ax_tot.axhline(3, color='#ff4d6d', linestyle='--', linewidth=1.5, alpha=0.6)
ax_tot.text(len(dx_order)+0.2, 3.1, '≥3: Suspected', color='#ff4d6d',
            fontsize=8.5, va='bottom', fontfamily='monospace')
ax_tot.axhline(5, color='#ffc23e', linestyle='--', linewidth=1.5, alpha=0.6)
ax_tot.text(len(dx_order)+0.2, 5.1, '≥5: High Risk', color='#ffc23e',
            fontsize=8.5, va='bottom', fontfamily='monospace')
for i, (dx, pos) in enumerate(zip(dx_order, positions)):
    mean_v = scores_df[scores_df['dx']==dx]['total_score'].mean()
    ax_tot.text(pos, -0.55, f'μ={mean_v:.2f}', ha='center',
                color=COLORS[dx], fontsize=8, fontfamily='monospace')

# ─── RADAR CHART: avg A,B,C,D per class ──────────────────────
ax_rad.remove()
ax_rad = fig.add_subplot(gs[0, 2], polar=True)
ax_rad.set_facecolor('#080d14')
criteria = ['A\nAsymmetry','B\nBorder','C\nColor','D\nDiameter']
N = len(criteria)
angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

for dx in ['mel','nv','bcc']:
    sub = scores_df[scores_df['dx']==dx]
    vals = [
        sub['A_score'].mean() / 2,
        sub['B_score'].mean() / 2,
        sub['C_score'].mean() / 2,
        sub['D_score'].mean(),
    ]
    vals += vals[:1]
    ax_rad.plot(angles, vals, color=COLORS[dx], linewidth=2, label=FULL_NAMES[dx])
    ax_rad.fill(angles, vals, color=COLORS[dx], alpha=0.12)

ax_rad.set_xticks(angles[:-1])
ax_rad.set_xticklabels(criteria, fontsize=9, color='#e8f0fe')
ax_rad.set_ylim(0, 1)
ax_rad.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_rad.set_yticklabels(['0.25','0.5','0.75','1.0'], fontsize=7, color='#8097b5')
ax_rad.grid(color='#1a2535', linewidth=0.8)
ax_rad.set_facecolor('#080d14')
ax_rad.spines['polar'].set_edgecolor('#1a2535')
ax_rad.set_title('ABCDE Radar\nMel vs NV vs BCC', color='#e8f0fe',
                 fontsize=10, fontweight='bold', pad=18)
ax_rad.legend(loc='lower right', bbox_to_anchor=(1.4, -0.1),
              fontsize=7.5, framealpha=0.1, labelcolor='white')

# ─── ASYMMETRY SCORE DISTRIBUTION ────────────────────────────
for dx in FULL_NAMES:
    sub = scores_df[scores_df['dx']==dx]['A_score']
    counts_a = sub.value_counts().sort_index()
    ax_a.plot(counts_a.index, counts_a.values, marker='o',
              color=COLORS[dx], linewidth=1.8, markersize=5,
              label=dx.upper(), alpha=0.85)
ax_a.set_title('Asymmetry Score Frequency per Class', color='#e8f0fe',
               fontsize=11, fontweight='bold', pad=10)
ax_a.set_xlabel('Asymmetry Score (0=Symmetric, 2=Highly Asymmetric)',
                color='#8097b5', fontsize=8.5)
ax_a.set_ylabel('Sample Count', color='#8097b5', fontsize=9)
ax_a.tick_params(colors='#8097b5', labelsize=9)
ax_a.grid(color='#1a2535', linewidth=0.7, linestyle='--')
ax_a.set_axisbelow(True)
ax_a.legend(fontsize=7.5, framealpha=0.1, facecolor='#080d14',
            labelcolor='white', ncol=2)
ax_a.set_xticks([0,1,2])

# ─── BORDER (COMPACTNESS) DISTRIBUTION ───────────────────────
for dx in FULL_NAMES:
    sub = scores_df[scores_df['dx']==dx]['compactness']
    ax_b.hist(sub, bins=20, color=COLORS[dx], alpha=0.4, density=True,
              label=dx.upper(), histtype='stepfilled', linewidth=0)
    ax_b.hist(sub, bins=20, color=COLORS[dx], alpha=0.9, density=True,
              histtype='step', linewidth=1.5)
ax_b.set_title('Border Compactness Distribution (B-score)', color='#e8f0fe',
               fontsize=11, fontweight='bold', pad=10)
ax_b.set_xlabel('Compactness Index (1=Perfect Circle, >1=Irregular)', color='#8097b5', fontsize=8.5)
ax_b.set_ylabel('Density', color='#8097b5', fontsize=9)
ax_b.tick_params(colors='#8097b5', labelsize=9)
ax_b.grid(color='#1a2535', linewidth=0.7, linestyle='--')
ax_b.set_axisbelow(True)
ax_b.axvline(1.0, color='white', linestyle=':', alpha=0.5)
ax_b.legend(fontsize=7, framealpha=0.1, facecolor='#080d14',
            labelcolor='white', ncol=2)

# ─── SUSPICION RATE ──────────────────────────────────────────
susp_rates, high_rates, dx_labels_s = [], [], []
for dx in FULL_NAMES:
    sub = scores_df[scores_df['dx']==dx]['total_score']
    if len(sub) == 0: continue
    susp_rates.append((sub >= 3).mean() * 100)
    high_rates.append((sub >= 5).mean() * 100)
    dx_labels_s.append(dx.upper())

x2 = np.arange(len(dx_labels_s))
ax_susp.bar(x2 - 0.2, susp_rates, 0.38, label='≥3 Suspected',
            color=[COLORS[dx.lower()] for dx in dx_labels_s], alpha=0.7, edgecolor='none')
ax_susp.bar(x2 + 0.2, high_rates, 0.38, label='≥5 High Risk',
            color='#ff4d6d', alpha=0.6, edgecolor='none')
ax_susp.set_xticks(x2)
ax_susp.set_xticklabels(dx_labels_s, fontsize=9, color='#8097b5')
ax_susp.set_title('Clinical Suspicion Rate per Class', color='#e8f0fe',
                  fontsize=11, fontweight='bold', pad=10)
ax_susp.set_ylabel('% of Samples Flagged', color='#8097b5', fontsize=9)
ax_susp.tick_params(colors='#8097b5', labelsize=9)
ax_susp.grid(axis='y', color='#1a2535', linewidth=0.7)
ax_susp.set_axisbelow(True)
ax_susp.legend(fontsize=9, framealpha=0.1, facecolor='#080d14', labelcolor='white')
for i, (sr, hr) in enumerate(zip(susp_rates, high_rates)):
    ax_susp.text(i-0.2, sr+0.8, f'{sr:.0f}%', ha='center',
                 color='#e8f0fe', fontsize=7.5, fontfamily='monospace')
    ax_susp.text(i+0.2, hr+0.8, f'{hr:.0f}%', ha='center',
                 color='#ff4d6d', fontsize=7.5, fontfamily='monospace')
ax_susp.set_ylim(0, max(susp_rates)*1.25)

# ─── SAVE ─────────────────────────────────────────────────────
out_png = os.path.join(OUTPUT_DIR, "04_abcde_risk_analysis.png")
plt.savefig(out_png, dpi=180, bbox_inches='tight', facecolor='#020409')
plt.close()
print(f"✅ Saved: {out_png}")

abcde_json = {
    "samples_scored": sample_size,
    "mean_scores_by_class": {
        dx: {
            "A": round(scores_df[scores_df['dx']==dx]['A_score'].mean(), 3),
            "B": round(scores_df[scores_df['dx']==dx]['B_score'].mean(), 3),
            "C": round(scores_df[scores_df['dx']==dx]['C_score'].mean(), 3),
            "D": round(scores_df[scores_df['dx']==dx]['D_score'].mean(), 3),
            "total": round(scores_df[scores_df['dx']==dx]['total_score'].mean(), 3),
            "pct_suspected_ge3": round((scores_df[scores_df['dx']==dx]['total_score']>=3).mean()*100, 1),
            "pct_high_risk_ge5": round((scores_df[scores_df['dx']==dx]['total_score']>=5).mean()*100, 1),
        }
        for dx in FULL_NAMES if dx in scores_df['dx'].values
    },
    "overall_malignant_mean_total": round(
        scores_df[scores_df['malignant']==True]['total_score'].mean(), 3),
    "overall_benign_mean_total": round(
        scores_df[scores_df['malignant']==False]['total_score'].mean(), 3),
}
out_json = os.path.join(OUTPUT_DIR, "04_abcde_scores.json")
with open(out_json, 'w') as f:
    json.dump(abcde_json, f, indent=2)
print(f"✅ Saved: {out_json}")
print(f"\n🎯 Malignant mean ABCDE score: {abcde_json['overall_malignant_mean_total']}")
print(f"🎯 Benign mean ABCDE score:    {abcde_json['overall_benign_mean_total']}")