"""
DermoGraph-XAI | Script 02
AGE ANALYSIS — Risk by decade, violin plots, melanoma age curve
Outputs: 02_age_analysis.png, 02_age_data.json
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import json, os

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
    'nv':'#00e5c8','mel':'#ff4d6d','bkl':'#ffc23e',
    'bcc':'#7c5cfc','akiec':'#ff8c42','vasc':'#4ecdc4','df':'#a8dadc'
}
MALIGNANT = {'mel','bcc','akiec'}

df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=['age'])

bins   = [0,10,20,30,40,50,60,70,80,90]
labels = ['0–10','10–20','20–30','30–40','40–50','50–60','60–70','70–80','80+']
df['age_bin'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

fig = plt.figure(figsize=(20, 14), facecolor='#020409')
gs  = GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38,
               left=0.07, right=0.97, top=0.88, bottom=0.08)

ax_hist  = fig.add_subplot(gs[0, :2])
ax_box   = fig.add_subplot(gs[0, 2])
ax_mel   = fig.add_subplot(gs[1, 0])
ax_risk  = fig.add_subplot(gs[1, 1])
ax_heat  = fig.add_subplot(gs[1, 2])

for ax in [ax_hist, ax_box, ax_mel, ax_risk, ax_heat]:
    ax.set_facecolor('#080d14')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a2535')

fig.suptitle('DermoGraph-XAI  ·  Age Analysis & Risk Distribution',
             fontsize=22, fontweight='bold', color='#e8f0fe',
             fontfamily='monospace', y=0.95)
fig.text(0.5, 0.915, 'HAM10000  ·  Age-stratified clinical intelligence across 7 lesion classes',
         ha='center', fontsize=11, color='#8097b5', fontfamily='monospace')

# ─── AGE HISTOGRAM per class (stacked) ───────────────────────
age_range = np.arange(0, 90, 5)
bottom = np.zeros(len(age_range) - 1)
for dx in ['nv','bkl','df','vasc','akiec','bcc','mel']:
    sub = df[df['dx']==dx]['age']
    counts, edges = np.histogram(sub, bins=age_range)
    ax_hist.bar(edges[:-1], counts, width=4.5, bottom=bottom,
                color=COLORS[dx], label=FULL_NAMES[dx], alpha=0.92, edgecolor='none')
    bottom += counts

ax_hist.set_title('Age Distribution by Diagnostic Class (Stacked)', color='#e8f0fe',
                  fontsize=13, fontweight='bold', pad=12)
ax_hist.set_xlabel('Patient Age (years)', color='#8097b5', fontsize=10)
ax_hist.set_ylabel('Number of Images', color='#8097b5', fontsize=10)
ax_hist.tick_params(colors='#8097b5', labelsize=9)
ax_hist.grid(axis='y', color='#1a2535', linewidth=0.7, linestyle='--')
ax_hist.set_axisbelow(True)
ax_hist.axvline(df['age'].mean(), color='#ffc23e', linestyle='--', linewidth=1.5, alpha=0.7)
ax_hist.text(df['age'].mean()+1, ax_hist.get_ylim()[1]*0.9,
             f'Mean: {df["age"].mean():.1f} yrs',
             color='#ffc23e', fontsize=9, fontfamily='monospace')
ax_hist.legend(loc='upper left', fontsize=7.5, framealpha=0.1,
               facecolor='#080d14', labelcolor='white', ncol=2)

# ─── BOX PLOTS per class ──────────────────────────────────────
dx_order  = sorted(df['dx'].unique(), key=lambda x: df[df['dx']==x]['age'].median())
plot_data = [df[df['dx']==dx]['age'].dropna().values for dx in dx_order]
bp = ax_box.boxplot(plot_data, patch_artist=True, vert=True,
                    medianprops=dict(color='#020409', linewidth=2.5),
                    whiskerprops=dict(color='#8097b5', linewidth=1),
                    capprops=dict(color='#8097b5', linewidth=1),
                    flierprops=dict(marker='o', markersize=2,
                                   markerfacecolor='#8097b5', alpha=0.3))
for patch, dx in zip(bp['boxes'], dx_order):
    patch.set_facecolor(COLORS[dx])
    patch.set_alpha(0.85)
ax_box.set_xticks(range(1, len(dx_order)+1))
ax_box.set_xticklabels([dx.upper() for dx in dx_order], fontsize=8.5, color='#8097b5')
ax_box.set_title('Age Range per Class (Boxplot)', color='#e8f0fe',
                 fontsize=11, fontweight='bold', pad=10)
ax_box.set_ylabel('Age (years)', color='#8097b5', fontsize=9)
ax_box.tick_params(colors='#8097b5', labelsize=9)
ax_box.grid(axis='y', color='#1a2535', linewidth=0.7)
ax_box.set_axisbelow(True)
for i, dx in enumerate(dx_order, 1):
    med = df[df['dx']==dx]['age'].median()
    ax_box.text(i, med + 1.5, f'{med:.0f}', ha='center',
                color='#e8f0fe', fontsize=7.5, fontfamily='monospace')

# ─── MELANOMA RISK CURVE by age ───────────────────────────────
total_by_bin = df.groupby('age_bin', observed=True).size()
mel_by_bin   = df[df['dx']=='mel'].groupby('age_bin', observed=True).size()
risk_rate    = (mel_by_bin / total_by_bin * 100).fillna(0)

x_pos = range(len(labels))
ax_mel.fill_between(x_pos, risk_rate.reindex(labels).fillna(0).values,
                    alpha=0.25, color='#ff4d6d')
ax_mel.plot(x_pos, risk_rate.reindex(labels).fillna(0).values,
            color='#ff4d6d', linewidth=2.5, marker='o',
            markersize=7, markerfacecolor='#020409', markeredgewidth=2,
            markeredgecolor='#ff4d6d')
ax_mel.set_xticks(x_pos)
ax_mel.set_xticklabels(labels, rotation=35, fontsize=8, color='#8097b5')
ax_mel.set_title('Melanoma Risk Rate by Age Decade', color='#e8f0fe',
                 fontsize=11, fontweight='bold', pad=10)
ax_mel.set_ylabel('Melanoma % of Age Group', color='#8097b5', fontsize=9)
ax_mel.tick_params(colors='#8097b5', labelsize=9)
ax_mel.grid(color='#1a2535', linewidth=0.7, linestyle='--')
ax_mel.set_axisbelow(True)
peak_idx = int(np.argmax(risk_rate.reindex(labels).fillna(0).values))
peak_val = risk_rate.reindex(labels).fillna(0).values[peak_idx]
ax_mel.annotate(f'Peak: {peak_val:.1f}%\n{labels[peak_idx]} yrs',
                xy=(peak_idx, peak_val),
                xytext=(peak_idx - 1.5, peak_val + 2),
                color='#ff4d6d', fontsize=8.5, fontfamily='monospace',
                arrowprops=dict(arrowstyle='->', color='#ff4d6d', lw=1.5))

# ─── AGE MEAN per class bar ───────────────────────────────────
age_means = df.groupby('dx')['age'].mean().sort_values()
cols_sorted = [COLORS[dx] for dx in age_means.index]
bars = ax_risk.barh([FULL_NAMES[dx] for dx in age_means.index],
                    age_means.values, color=cols_sorted,
                    height=0.55, edgecolor='none')
ax_risk.set_title('Mean Patient Age per Class', color='#e8f0fe',
                  fontsize=11, fontweight='bold', pad=10)
ax_risk.set_xlabel('Mean Age (years)', color='#8097b5', fontsize=9)
ax_risk.tick_params(colors='#8097b5', labelsize=8)
ax_risk.grid(axis='x', color='#1a2535', linewidth=0.7, linestyle='--')
ax_risk.set_axisbelow(True)
for bar, v, dx in zip(bars, age_means.values, age_means.index):
    ax_risk.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{v:.1f} yrs', va='center',
                 color='#e8f0fe', fontsize=8.5, fontfamily='monospace')
    if dx in MALIGNANT:
        ax_risk.text(1, bar.get_y() + bar.get_height()/2,
                     '⚠', va='center', color='#ff4d6d', fontsize=9)
ax_risk.set_xlim(0, max(age_means.values) * 1.2)

# ─── AGE BIN HEATMAP (dx × age_bin) ──────────────────────────
pivot = df.groupby(['dx', 'age_bin'], observed=True).size().unstack(fill_value=0)
pivot = pivot.reindex(columns=labels, fill_value=0)
pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)  # normalize per class

im = ax_heat.imshow(pivot_norm.values, aspect='auto',
                    cmap='YlOrRd', interpolation='nearest',
                    vmin=0, vmax=pivot_norm.values.max())
ax_heat.set_xticks(range(len(labels)))
ax_heat.set_xticklabels(labels, rotation=45, fontsize=7.5, color='#8097b5')
ax_heat.set_yticks(range(len(pivot_norm.index)))
ax_heat.set_yticklabels([FULL_NAMES[dx] for dx in pivot_norm.index],
                        fontsize=8, color='#8097b5')
ax_heat.set_title('Age Distribution Heatmap (Normalized)', color='#e8f0fe',
                  fontsize=11, fontweight='bold', pad=10)
for i in range(len(pivot_norm.index)):
    for j in range(len(labels)):
        v = pivot_norm.values[i, j]
        ax_heat.text(j, i, f'{v:.2f}', ha='center', va='center',
                     color='black' if v > 0.2 else '#8097b5', fontsize=6.5)
plt.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02).ax.tick_params(colors='#8097b5')

# ─── SAVE ─────────────────────────────────────────────────────
out_png = os.path.join(OUTPUT_DIR, "02_age_analysis.png")
plt.savefig(out_png, dpi=180, bbox_inches='tight', facecolor='#020409')
plt.close()
print(f"✅ Saved: {out_png}")

age_data = {
    "mean_age_by_class": df.groupby('dx')['age'].mean().round(2).to_dict(),
    "median_age_by_class": df.groupby('dx')['age'].median().to_dict(),
    "std_age_by_class": df.groupby('dx')['age'].std().round(2).to_dict(),
    "melanoma_risk_pct_by_decade": {
        k: round(v, 2) for k, v in risk_rate.reindex(labels).fillna(0).to_dict().items()
    },
    "peak_melanoma_age_group": labels[peak_idx],
    "overall_age_mean": round(df['age'].mean(), 2),
    "overall_age_median": float(df['age'].median()),
}
out_json = os.path.join(OUTPUT_DIR, "02_age_data.json")
with open(out_json, 'w') as f:
    json.dump(age_data, f, indent=2)
print(f"✅ Saved: {out_json}")
print(f"\n🎯 Peak melanoma risk age group: {labels[peak_idx]} ({peak_val:.1f}% of all cases in that group)")