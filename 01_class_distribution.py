"""
DermoGraph-XAI | Script 01
CLASS DISTRIBUTION & DATASET OVERVIEW
Outputs: dermograph_output/01_class_distribution.png
         dermograph_output/01_dataset_summary.json
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Always run relative to THIS script's folder ───────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))

INPUT_CSV  = "HAM10000_metadata.csv"
OUTPUT_DIR = "dermograph_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FULL_NAMES = {
    'nv':    'Melanocytic Nevi',
    'mel':   'Melanoma',
    'bkl':   'Benign Keratosis',
    'bcc':   'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis',
    'vasc':  'Vascular Lesion',
    'df':    'Dermatofibroma'
}
COLORS = {
    'nv':    '#00e5c8',
    'mel':   '#ff4d6d',
    'bkl':   '#ffc23e',
    'bcc':   '#7c5cfc',
    'akiec': '#ff8c42',
    'vasc':  '#4ecdc4',
    'df':    '#a8dadc'
}
MALIGNANT = {'mel', 'bcc', 'akiec'}

# ── Load ──────────────────────────────────────────────────────
df     = pd.read_csv(INPUT_CSV)
counts = df['dx'].value_counts()
total  = len(df)

# ── Figure ────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 13), facecolor='#020409')
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
               left=0.06, right=0.97, top=0.88, bottom=0.08)

ax_bar  = fig.add_subplot(gs[0, :2])
ax_pie  = fig.add_subplot(gs[0, 2])
ax_risk = fig.add_subplot(gs[1, 0])
ax_type = fig.add_subplot(gs[1, 1])
ax_stat = fig.add_subplot(gs[1, 2])

for ax in [ax_bar, ax_pie, ax_risk, ax_type, ax_stat]:
    ax.set_facecolor('#080d14')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a2535')

fig.suptitle('DermoGraph-XAI  ·  HAM10000 Clinical Dataset Analysis',
             fontsize=22, fontweight='bold', color='#e8f0fe',
             fontfamily='monospace', y=0.95)
fig.text(0.5, 0.915,
         'n = 10,015 biopsy-confirmed dermoscopy images  ·  7 diagnostic classes',
         ha='center', fontsize=11, color='#8097b5', fontfamily='monospace')

# ── Horizontal Bar ────────────────────────────────────────────
labels = [FULL_NAMES[k] for k in counts.index]
vals   = counts.values
cols   = [COLORS[k] for k in counts.index]

bars = ax_bar.barh(labels, vals, color=cols, height=0.62, edgecolor='none')
ax_bar.set_xlim(0, max(vals) * 1.22)
ax_bar.set_xlabel('Number of Images', color='#8097b5', fontsize=10)
ax_bar.set_title('Image Count per Diagnostic Class', color='#e8f0fe',
                 fontsize=13, fontweight='bold', pad=12)
ax_bar.tick_params(colors='#8097b5', labelsize=9)
ax_bar.grid(axis='x', color='#1a2535', linewidth=0.7, linestyle='--')
ax_bar.set_axisbelow(True)

for bar, v, k in zip(bars, vals, counts.index):
    pct = v / total * 100
    ax_bar.text(v + 60, bar.get_y() + bar.get_height()/2,
                f'{v:,}  ({pct:.1f}%)', va='center',
                color='#e8f0fe', fontsize=9, fontfamily='monospace')
    if k in MALIGNANT:
        ax_bar.text(-180, bar.get_y() + bar.get_height()/2,
                    '⚠', va='center', color='#ff4d6d', fontsize=10)

ax_bar.axvline(counts['nv'] / 7, color='#ff4d6d', linestyle=':', alpha=0.5, linewidth=1)
ax_bar.text(counts['nv']/7 + 80, 0.3, 'Avg if balanced',
            color='#ff4d6d', fontsize=8, alpha=0.7)

# ── Donut ─────────────────────────────────────────────────────
wedge_vals = [counts[k] for k in counts.index]
wedge_cols = [COLORS[k] for k in counts.index]
wedges, texts, autotexts = ax_pie.pie(
    wedge_vals, labels=None, colors=wedge_cols,
    autopct='%1.1f%%', startangle=140,
    wedgeprops=dict(width=0.55, edgecolor='#020409', linewidth=2),
    pctdistance=0.78
)
for at in autotexts:
    at.set_fontsize(7.5)
    at.set_color('#020409')
    at.set_fontweight('bold')

ax_pie.set_title('Class Distribution', color='#e8f0fe',
                 fontsize=13, fontweight='bold', pad=12)
ax_pie.text(0, 0, f'{total:,}\nimages', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#e8f0fe', fontfamily='monospace')

legend_patches = [mpatches.Patch(color=COLORS[k], label=FULL_NAMES[k]) for k in counts.index]
ax_pie.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.28),
              ncol=2, fontsize=7.5, framealpha=0, labelcolor='#8097b5')

# ── Malignant vs Benign ───────────────────────────────────────
mal_count = sum(counts[k] for k in counts.index if k in MALIGNANT)
ben_count = total - mal_count
bars2 = ax_risk.bar(['Malignant\n(mel+bcc+akiec)', 'Benign\n(nv+bkl+vasc+df)'],
                    [mal_count, ben_count],
                    color=['#ff4d6d', '#00e5c8'], width=0.5, edgecolor='none')
ax_risk.set_title('Malignant vs Benign', color='#e8f0fe',
                  fontsize=11, fontweight='bold', pad=10)
ax_risk.set_ylabel('Count', color='#8097b5', fontsize=9)
ax_risk.tick_params(colors='#8097b5', labelsize=9)
ax_risk.grid(axis='y', color='#1a2535', linewidth=0.7)
ax_risk.set_axisbelow(True)
for bar, v in zip(bars2, [mal_count, ben_count]):
    ax_risk.text(bar.get_x() + bar.get_width()/2, v + 80,
                 f'{v:,}\n({v/total*100:.1f}%)',
                 ha='center', va='bottom', color='#e8f0fe',
                 fontsize=10, fontweight='bold', fontfamily='monospace')
ax_risk.set_ylim(0, max(mal_count, ben_count) * 1.2)

# ── Diagnosis Type ────────────────────────────────────────────
type_counts = df['dx_type'].value_counts()
type_colors = ['#7c5cfc', '#00e5c8', '#ffc23e', '#ff8c42']
bars3 = ax_type.bar(type_counts.index, type_counts.values,
                    color=type_colors[:len(type_counts)], width=0.5, edgecolor='none')
ax_type.set_title('Diagnosis Confirmation Type', color='#e8f0fe',
                  fontsize=11, fontweight='bold', pad=10)
ax_type.tick_params(colors='#8097b5', labelsize=9)
ax_type.grid(axis='y', color='#1a2535', linewidth=0.7)
ax_type.set_axisbelow(True)
for bar, v in zip(bars3, type_counts.values):
    ax_type.text(bar.get_x() + bar.get_width()/2, v + 50,
                 f'{v:,}', ha='center', va='bottom',
                 color='#e8f0fe', fontsize=9, fontfamily='monospace')
ax_type.set_ylim(0, max(type_counts.values) * 1.18)

# ── Stats Table ───────────────────────────────────────────────
ax_stat.axis('off')
ax_stat.set_title('Key Dataset Statistics', color='#e8f0fe',
                  fontsize=11, fontweight='bold', pad=10)

stats = [
    ('Total Images',        f'{total:,}'),
    ('Unique Lesions',      f'{df["lesion_id"].nunique():,}'),
    ('Diagnostic Classes',  '7'),
    ('Malignant Classes',   '3'),
    ('Avg Patient Age',     f'{df["age"].mean():.1f} yrs'),
    ('Age Range',           f'{int(df["age"].min())}–{int(df["age"].max())} yrs'),
    ('Male Patients',       f'{(df["sex"]=="male").sum():,} ({(df["sex"]=="male").mean()*100:.1f}%)'),
    ('Female Patients',     f'{(df["sex"]=="female").sum():,} ({(df["sex"]=="female").mean()*100:.1f}%)'),
    ('Melanoma Cases',      f'{counts["mel"]:,} ({counts["mel"]/total*100:.1f}%)'),
    ('Class Imbalance',     f'{counts["nv"]/counts["df"]:.0f}:1 (nv vs df)'),
    ('Histology-confirmed', f'{(df["dx_type"]=="histo").sum():,}'),
    ('Biopsy-confirmed',    f'{df["dx_type"].isin(["histo","confocal"]).sum():,}'),
]

for i, (label, val) in enumerate(stats):
    y = 0.97 - i * 0.082
    ax_stat.text(0.02, y, label, transform=ax_stat.transAxes,
                 fontsize=9, color='#8097b5', va='top')
    ax_stat.text(0.98, y, val, transform=ax_stat.transAxes,
                 fontsize=9, color='#00e5c8', va='top', ha='right',
                 fontweight='bold', fontfamily='monospace')
    # FIXED: use ax.plot() with transform instead of axhline()
    ax_stat.plot([0.01, 0.99], [y - 0.01, y - 0.01],
                 transform=ax_stat.transAxes,
                 color='#1a2535', linewidth=0.5, clip_on=False)

# ── Save ──────────────────────────────────────────────────────
out_png = os.path.join(OUTPUT_DIR, "01_class_distribution.png")
plt.savefig(out_png, dpi=180, bbox_inches='tight', facecolor='#020409')
plt.close()
print(f"✅ Saved: {out_png}")

# ── JSON ──────────────────────────────────────────────────────
summary = {
    "total_images": int(total),
    "unique_lesions": int(df['lesion_id'].nunique()),
    "classes": {
        k: {
            "full_name": FULL_NAMES[k],
            "count": int(v),
            "percentage": round(v/total*100, 2),
            "malignant": k in MALIGNANT
        } for k, v in counts.items()
    },
    "malignant_total": int(mal_count),
    "benign_total": int(ben_count),
    "malignant_pct": round(mal_count/total*100, 2),
    "age_stats": {
        "mean": round(df['age'].mean(), 2),
        "median": float(df['age'].median()),
        "std": round(df['age'].std(), 2),
        "min": int(df['age'].min()),
        "max": int(df['age'].max())
    },
    "sex_distribution": df['sex'].value_counts().to_dict(),
    "diagnosis_type": df['dx_type'].value_counts().to_dict(),
    "class_imbalance_ratio": round(counts['nv']/counts['df'], 1)
}
out_json = os.path.join(OUTPUT_DIR, "01_dataset_summary.json")
with open(out_json, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✅ Saved: {out_json}")
print("\n📊 Quick Stats:")
for k, v in counts.items():
    print(f"   {FULL_NAMES[k]:30s}: {v:5d} images ({v/total*100:.1f}%)")