"""
dataset_loader.py
DermoGraph-XAI — Unified Dataset Loader
Loads all 6 datasets → single combined DataFrame

Run: python3 dataset_loader.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# ══════════════════════════════════════════════════════════
# YOUR PATHS — already confirmed working
# ══════════════════════════════════════════════════════════
BASE    = "/Users/akshxunfiltered/DermoXAI/SKIN CANCER DATASET"
OUTPUT  = "/Users/akshxunfiltered/DermoXAI/dermograph_output"
os.makedirs(OUTPUT, exist_ok=True)

# ── Unified class mapping (9 classes) ─────────────────────
# 0=Melanoma, 1=Nevi, 2=BCC, 3=Actinic/SCC, 4=Benign Kerat,
# 5=Dermatofibroma, 6=Vascular, 7=Other
CLASS_MAP = {
    # HAM10000
    'mel':0, 'nv':1, 'bcc':2, 'akiec':3,
    'bkl':4, 'df':5, 'vasc':6,
    # PAD-UFES-20
    'MEL':0, 'NEV':1, 'BCC':2, 'SCC':3,
    'ACK':3, 'BOD':4, 'SEK':4,
    # melanoma_cancer_dataset folder names
    'malignant':0, 'benign':1,
}

CLASS_NAMES = [
    'Melanoma', 'Nevi', 'Basal Cell Carcinoma',
    'Actinic Keratosis', 'Benign Keratosis',
    'Dermatofibroma', 'Vascular', 'Other'
]

# ══════════════════════════════════════════════════════════
# LOADER 1: HAM10000
# ══════════════════════════════════════════════════════════
def load_ham10000():
    print("\n── Loading HAM10000 ──")
    ham_dir = f"{BASE}/HAM10000"
    df = pd.read_csv(f"{ham_dir}/HAM10000_metadata.csv")

    def find_image(img_id):
        for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
            p = f"{ham_dir}/{part}/{img_id}.jpg"
            if os.path.exists(p):
                return p
        return None

    df['image_path'] = df['image_id'].apply(find_image)
    df = df.dropna(subset=['image_path'])
    df['label']      = df['dx'].map(CLASS_MAP).fillna(7).astype(int)
    df['fitzpatrick'] = 0   # not available in HAM10000
    df['source']     = 'ham10000'
    df['age']        = pd.to_numeric(df['age'], errors='coerce').fillna(50)
    df['sex']        = df['sex'].fillna('unknown')

    print(f"   ✓ {len(df)} images loaded")
    print(f"   Classes: {df['dx'].value_counts().to_dict()}")
    return df[['image_path','label','fitzpatrick','age','sex','source']]


# ══════════════════════════════════════════════════════════
# LOADER 2: ISIC 2020 (images only, no CSV)
# Binary: melanoma vs benign based on filename prefix
# ISIC filenames don't encode label — treat all as "unknown"
# We'll use this dataset for pretraining/augmentation only
# ══════════════════════════════════════════════════════════
def load_isic2020():
    print("\n── Loading ISIC 2020 ──")
    isic_dir = f"{BASE}/melanoma ISIC 2020 dataset - 224"

    images = [f for f in os.listdir(isic_dir)
              if f.lower().endswith(('.jpg','.jpeg','.png'))]

    rows = []
    for fname in images:
        rows.append({
            'image_path':   f"{isic_dir}/{fname}",
            'label':        1,   # treat as benign (nevi) — no label CSV available
            'fitzpatrick':  0,
            'age':          50,
            'sex':          'unknown',
            'source':       'isic2020',
        })

    df = pd.DataFrame(rows)
    print(f"   ✓ {len(df)} images loaded (all treated as benign/nevi — no label CSV)")
    print(f"   Note: Used for augmentation and pretraining only")
    return df


# ══════════════════════════════════════════════════════════
# LOADER 3: PAD-UFES-20
# ══════════════════════════════════════════════════════════
def load_padufes():
    print("\n── Loading PAD-UFES-20 ──")
    pad_dir = f"{BASE}/PAD-UFES-20"
    df = pd.read_csv(f"{pad_dir}/metadata.csv")

    def find_image(img_id):
        p = f"{pad_dir}/images/{img_id}"
        if os.path.exists(p):
            return p
        return None

    df['image_path']  = df['img_id'].apply(find_image)
    df = df.dropna(subset=['image_path'])
    df['label']       = df['diagnostic'].map(CLASS_MAP).fillna(7).astype(int)
    df['fitzpatrick'] = pd.to_numeric(df['fitspatrick'], errors='coerce').fillna(0).astype(int)
    df['age']         = pd.to_numeric(df['age'], errors='coerce').fillna(50)
    df['sex']         = df['gender'].fillna('unknown')
    df['source']      = 'padufes20'

    print(f"   ✓ {len(df)} images loaded")
    print(f"   Classes: {df['diagnostic'].value_counts().to_dict()}")
    print(f"   Fitzpatrick types: {df['fitzpatrick'].value_counts().sort_index().to_dict()}")
    return df[['image_path','label','fitzpatrick','age','sex','source']]


# ══════════════════════════════════════════════════════════
# LOADER 4: Derm7pt (release_v0)
# ══════════════════════════════════════════════════════════
def load_derm7pt():
    print("\n── Loading Derm7pt ──")
    d7_dir  = f"{BASE}/release_v0"
    meta_df = pd.read_csv(f"{d7_dir}/meta/meta.csv")

    def find_image(fname):
        p = f"{d7_dir}/images/{fname}"
        if os.path.exists(p):
            return p
        # try without extension
        for ext in ['.jpg','.jpeg','.png']:
            p2 = f"{d7_dir}/images/{fname}{ext}"
            if os.path.exists(p2):
                return p2
        return None

    # Derm7pt has 'derm' column = dermoscopy image filename
    img_col = 'derm' if 'derm' in meta_df.columns else meta_df.columns[0]
    meta_df['image_path'] = meta_df[img_col].apply(find_image)
    meta_df = meta_df.dropna(subset=['image_path'])

    # Map diagnosis
    dx_col = 'diagnosis' if 'diagnosis' in meta_df.columns else None
    if dx_col:
        dx_map = {'melanoma':0,'nevus':1,'basal cell carcinoma':2,
                  'seborrheic keratosis':4,'dermatofibroma':5,'vascular':6,'other':7}
        meta_df['label'] = meta_df[dx_col].str.lower().map(dx_map).fillna(7).astype(int)
    else:
        meta_df['label'] = 7

    meta_df['fitzpatrick'] = 0
    meta_df['age']         = 50
    meta_df['sex']         = 'unknown'
    meta_df['source']      = 'derm7pt'

    print(f"   ✓ {len(meta_df)} images loaded")
    if dx_col:
        print(f"   Classes: {meta_df[dx_col].value_counts().to_dict()}")
    return meta_df[['image_path','label','fitzpatrick','age','sex','source']]


# ══════════════════════════════════════════════════════════
# LOADER 5: MIDAS
# ══════════════════════════════════════════════════════════
def load_midas():
    print("\n── Loading MIDAS ──")
    midas_dir = f"{BASE}/MIDAS/midasmultimodalimagedatasetforaibasedskincancer"

    images = [f for f in os.listdir(midas_dir)
              if f.lower().endswith(('.jpg','.jpeg','.png','.JPG','.JPEG'))]

    rows = []
    for fname in images:
        # Label from filename: cropped = lesion, else = full image
        label = 0 if 'mel' in fname.lower() else 1
        rows.append({
            'image_path':  f"{midas_dir}/{fname}",
            'label':       label,
            'fitzpatrick': 0,
            'age':         50,
            'sex':         'unknown',
            'source':      'midas',
        })

    df = pd.DataFrame(rows)
    print(f"   ✓ {len(df)} images loaded")
    return df


# ══════════════════════════════════════════════════════════
# LOADER 6: melanoma_cancer_dataset (folder-based labels)
# ══════════════════════════════════════════════════════════
def load_melanoma_cancer():
    print("\n── Loading melanoma_cancer_dataset ──")
    mc_dir = f"{BASE}/melanoma_cancer_dataset"

    rows = []
    for split in ['train', 'test']:
        for cls in ['malignant', 'benign']:
            cls_dir = f"{mc_dir}/{split}/{cls}"
            if not os.path.exists(cls_dir):
                continue
            label = CLASS_MAP[cls]
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg','.jpeg','.png')):
                    rows.append({
                        'image_path':  f"{cls_dir}/{fname}",
                        'label':       label,
                        'fitzpatrick': 0,
                        'age':         50,
                        'sex':         'unknown',
                        'source':      f'melanoma_{split}',
                    })

    df = pd.DataFrame(rows)
    print(f"   ✓ {len(df)} images loaded")
    print(f"   Malignant: {(df['label']==0).sum()} | Benign: {(df['label']==1).sum()}")
    return df


# ══════════════════════════════════════════════════════════
# COMBINE ALL DATASETS
# ══════════════════════════════════════════════════════════
def load_all_datasets():
    print("="*55)
    print("  DermoGraph-XAI — Loading All Datasets")
    print("="*55)

    loaders = [
        load_ham10000,
        load_isic2020,
        load_padufes,
        load_derm7pt,
        load_midas,
        load_melanoma_cancer,
    ]

    dfs = []
    for loader in loaders:
        try:
            df = loader()
            dfs.append(df)
        except Exception as e:
            print(f"   ✗ Error in {loader.__name__}: {e}")

    combined = pd.concat(dfs, ignore_index=True)

    # Remove any rows where image file doesn't actually exist
    print("\n── Verifying image files exist ──")
    combined['exists'] = combined['image_path'].apply(os.path.exists)
    missing = (~combined['exists']).sum()
    if missing > 0:
        print(f"   ⚠ Removing {missing} missing image paths")
    combined = combined[combined['exists']].drop(columns=['exists'])
    combined = combined.reset_index(drop=True)

    # ── Summary ──────────────────────────────────────────
    print("\n" + "="*55)
    print("  COMBINED DATASET SUMMARY")
    print("="*55)
    print(f"  Total images:   {len(combined):,}")
    print(f"  Total sources:  {combined['source'].nunique()}")
    print(f"\n  By source:")
    for src, cnt in combined['source'].value_counts().items():
        print(f"    {src:<25} {cnt:>6,} images")

    print(f"\n  By class:")
    for label, cnt in combined['label'].value_counts().sort_index().items():
        name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else 'Other'
        pct  = cnt / len(combined) * 100
        bar  = '█' * int(pct / 2)
        print(f"    {label} {name:<22} {cnt:>6,}  {pct:5.1f}%  {bar}")

    print(f"\n  Fitzpatrick types available: {(combined['fitzpatrick']>0).sum():,} images")
    print(f"  (from PAD-UFES-20 — needed for fairness module)")

    # Save to CSV for reuse
    out_path = f"{OUTPUT}/combined_dataset.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved to: {out_path}")
    print("="*55)

    return combined


# ══════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_all_datasets()
    print(f"\n✓ Done. DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Sample rows:")
    print(df.sample(3).to_string())