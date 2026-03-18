"""
DermoGraph-XAI | Script 03
BODY LOCALIZATION & GENDER ANALYSIS
Outputs: 03_localization_gender.png, 03_localization_data.json
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
    'nv':'Melanocytic Nevi','mel':'Melanoma','bkl':'Benign Keratosis',
    'bcc':'Basal Cell Carcinoma','akiec':'Actinic Keratosis',
    'vasc':'Vascular Lesion','df':'Dermatofibroma'
}
COLORS = {
    'nv':'#00e5c8','mel':'#ff4d6d','bkl':'#ffc23e',
    'bcc':'#7c5cfc','akiec':'#ff8c42','vasc':'#4ecdc4','df':'#a8dadc'
}
MALIGNANT = {'mel','bcc','akiec'}

df = pd.read_csv(INPUT_CSV)
df_known = df[df['sex'].isin(['male','female'])]

fig = plt.figure(figsize=(22, 14), facecolor='#020409')
gs  = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.4,
               left=0.06, right=0.97, top=0.88, bottom=0.07)

ax_loc   = fig.add_subplot(gs[0, :2])
ax_sex   = fig.add_subplot(gs[0, 2])
ax_msex  = fig.add_subplot(gs[1, 0])
ax_lheat = fig.add_subplot(gs[1, 1])
ax_mfrat = fig.add_subplot(gs[1, 2])

for ax in [ax_loc, ax_sex, ax_msex, ax_lheat, ax_mfrat]:
    ax.set_facecolor('#080d14')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a2535')

fig.suptitle('DermoGraph-XAI  ·  Body Localization & Gender Analysis',
             fontsize=22, fontweight='bold', color='#e8f0fe',
             fontfamily='monospace', y=0.95)
fig.text(0.5, 0.915, 'HAM10000  ·  Spatial distribution of lesions across body sites and patient demographics',
         ha='center', fontsize=11, color='#8097b5', fontfamily='monospace')

# ─── LOCALIZATION GROUPED BAR ─────────────────────────────────
loc_order = df['localization'].value_counts().head(9).index.tolist()
loc_order = [l for l in loc_order if l != 'unknown']

x     = np.arange(len(loc_order))
width = 0.11
dx_list = list(COLORS.keys())

for i, dx in enumerate(dx_list):
    vals = [df[(df['dx']==dx) & (df['localization']==l)].shape[0] for l in loc_order]
    ax_loc.bar(x + i*width - 3*width, vals, width, color=COLORS[dx],
               label=FULL_NAMES[dx], alpha=0.88, edgecolor='none')

ax_loc.set_xticks(x)
ax_loc.set_xticklabels([l.replace(' ','\n') for l in loc_order],
                       fontsize=9, color='#8097b5')
ax_loc.set_title('Lesion Count by Body Location & Class', color='#e8f0fe',
                 fontsize=13, fontweight='bold', pad=12)
ax_loc.set_ylabel('Number of Images', color='#8097b5', fontsize=10)
ax_loc.tick_params(colors='#8097b5', labelsize=9)
ax_loc.grid(axis='y', color='#1a2535', linewidth=0.7, linestyle='--')
ax_loc.set_axisbelow(True)
ax_loc.legend(fontsize=7.5, framealpha=0.1, facecolor='#080d14',
              labelcolor='white', ncol=4, loc='upper right')

# ─── GENDER PIE ───────────────────────────────────────────────
sex_counts = df_known['sex'].value_counts()
cols_sex   = ['#7c5cfc','#ff8c42']
wedges, _, auto = ax_sex.pie(
    sex_counts.values, labels=None, colors=cols_sex,
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(width=0.55, edgecolor='#020409', linewidth=2),
    pctdistance=0.75
)
for at in auto:
    at.set_fontsize(11); at.set_color('#020409'); at.set_fontweight('bold')
ax_sex.set_title('Gender Distribution', color='#e8f0fe',
                 fontsize=12, fontweight='bold', pad=12)
ax_sex.text(0, 0, f'{len(df_known):,}\npatients', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#e8f0fe', fontfamily='monospace')
ax_sex.legend(['Male','Female'], loc='lower center', bbox_to_anchor=(0.5,-0.15),
              fontsize=10, framealpha=0, labelcolor='#8097b5', ncol=2)

# ─── MELANOMA GENDER BREAKDOWN ────────────────────────────────
mel_sex = df[df['dx']=='mel']['sex'].value_counts()
mel_sex = mel_sex[mel_sex.index.isin(['male','female'])]
bars = ax_msex.bar(['Male','Female'], [mel_sex.get('male',0), mel_sex.get('female',0)],
                   color=['#7c5cfc','#ff8c42'], width=0.45, edgecolor='none')
ax_msex.set_title('Melanoma Cases by Gender', color='#e8f0fe',
                  fontsize=11, fontweight='bold', pad=10)
ax_msex.set_ylabel('Number of Cases', color='#8097b5', fontsize=9)
ax_msex.tick_params(colors='#8097b5', labelsize=10)
ax_msex.grid(axis='y', color='#1a2535', linewidth=0.7)
ax_msex.set_axisbelow(True)
total_mel = mel_sex.sum()
for bar, sex_label in zip(bars, ['male','female']):
    v = mel_sex.get(sex_label, 0)
    ax_msex.text(bar.get_x()+bar.get_width()/2, v+8,
                 f'{v}\n({v/total_mel*100:.1f}%)',
                 ha='center', va='bottom', color='#e8f0fe',
                 fontsize=10, fontweight='bold', fontfamily='monospace')
ax_msex.set_ylim(0, max(mel_sex.values)*1.25)
ax_msex.text(0.5, 0.05, f'Total melanoma cases: {total_mel}',
             transform=ax_msex.transAxes, ha='center',
             color='#ff4d6d', fontsize=9, fontfamily='monospace')

# ─── LOCATION × CLASS HEATMAP ────────────────────────────────
pivot2 = df.groupby(['localization','dx']).size().unstack(fill_value=0)
pivot2 = pivot2.reindex(columns=list(FULL_NAMES.keys()), fill_value=0)
loc_top = df['localization'].value_counts().head(8).index.tolist()
pivot2  = pivot2.loc[pivot2.index.isin(loc_top)]
pivot2  = pivot2.loc[loc_top]
pivot2_norm = pivot2.div(pivot2.sum(axis=1), axis=0)

im = ax_lheat.imshow(pivot2_norm.values, aspect='auto',
                     cmap='plasma', interpolation='nearest',
                     vmin=0, vmax=pivot2_norm.values.max())
ax_lheat.set_xticks(range(len(FULL_NAMES)))
ax_lheat.set_xticklabels(list(FULL_NAMES.keys()), fontsize=9,
                          color='#8097b5', rotation=30)
ax_lheat.set_yticks(range(len(loc_top)))
ax_lheat.set_yticklabels(loc_top, fontsize=8.5, color='#8097b5')
ax_lheat.set_title('Class Distribution by Location (Normalized)', color='#e8f0fe',
                   fontsize=11, fontweight='bold', pad=10)
for i in range(len(loc_top)):
    for j in range(len(FULL_NAMES)):
        v = pivot2_norm.values[i, j]
        ax_lheat.text(j, i, f'{v:.2f}', ha='center', va='center',
                      color='white' if v > 0.3 else '#8097b5', fontsize=7)
plt.colorbar(im, ax=ax_lheat, shrink=0.85).ax.tick_params(colors='#8097b5')

# ─── M/F RATIO PER CLASS ─────────────────────────────────────
ratios, dxnames, bar_colors = [], [], []
for dx in FULL_NAMES:
    sub = df[df['dx']==dx]
    m = (sub['sex']=='male').sum()
    f = (sub['sex']=='female').sum()
    ratio = m/f if f > 0 else 0
    ratios.append(ratio)
    dxnames.append(FULL_NAMES[dx])
    bar_colors.append(COLORS[dx])

bars2 = ax_mfrat.barh(dxnames, ratios, color=bar_colors, height=0.55, edgecolor='none')
ax_mfrat.axvline(1.0, color='#ffc23e', linestyle='--', linewidth=1.5, alpha=0.7)
ax_mfrat.text(1.02, len(dxnames)-0.3, '  1:1 parity',
              color='#ffc23e', fontsize=8.5, va='top', fontfamily='monospace')
ax_mfrat.set_title('Male:Female Ratio per Class', color='#e8f0fe',
                   fontsize=11, fontweight='bold', pad=10)
ax_mfrat.set_xlabel('Male / Female Ratio', color='#8097b5', fontsize=9)
ax_mfrat.tick_params(colors='#8097b5', labelsize=8)
ax_mfrat.grid(axis='x', color='#1a2535', linewidth=0.7, linestyle='--')
ax_mfrat.set_axisbelow(True)
for bar, ratio in zip(bars2, ratios):
    ax_mfrat.text(ratio+0.01, bar.get_y()+bar.get_height()/2,
                  f'{ratio:.2f}x', va='center',
                  color='#e8f0fe', fontsize=8.5, fontfamily='monospace')

# ─── SAVE ─────────────────────────────────────────────────────
out_png = os.path.join(OUTPUT_DIR, "03_localization_gender.png")
plt.savefig(out_png, dpi=180, bbox_inches='tight', facecolor='#020409')
plt.close()
print(f"✅ Saved: {out_png}")

loc_data = {
    "top_locations": df['localization'].value_counts().head(10).to_dict(),
    "melanoma_by_location": df[df['dx']=='mel']['localization'].value_counts().to_dict(),
    "gender_distribution": sex_counts.to_dict(),
    "mf_ratio_by_class": {dx: round(r,3) for dx,r in zip(FULL_NAMES.keys(), ratios)},
    "melanoma_gender": mel_sex.to_dict(),
}
out_json = os.path.join(OUTPUT_DIR, "03_localization_data.json")
with open(out_json, 'w') as f:
    json.dump(loc_data, f, indent=2)
print(f"✅ Saved: {out_json}")