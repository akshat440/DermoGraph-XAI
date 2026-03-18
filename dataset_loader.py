"""
dataset_loader.py
DermoGraph-XAI — Unified Dataset Loader
Run: python3 dataset_loader.py
"""

import os
import pandas as pd

BASE   = "/Users/akshxunfiltered/DermoXAI/SKIN CANCER DATASET"
OUTPUT = "/Users/akshxunfiltered/DermoXAI/dermograph_output"
os.makedirs(OUTPUT, exist_ok=True)

CLASS_MAP = {
    'mel':0, 'nv':1, 'bcc':2, 'akiec':3, 'bkl':4, 'df':5, 'vasc':6,
    'MEL':0, 'NEV':1, 'BCC':2, 'SCC':3, 'ACK':3, 'BOD':4, 'SEK':4,
    'malignant':0, 'benign':1,
}
CLASS_NAMES = ['Melanoma','Nevi','Basal Cell Carcinoma','Actinic Keratosis',
               'Benign Keratosis','Dermatofibroma','Vascular','Other']

def load_ham10000():
    print("\n── Loading HAM10000 ──")
    d = f"{BASE}/HAM10000"
    df = pd.read_csv(f"{d}/HAM10000_metadata.csv")
    def find(iid):
        for p in ['HAM10000_images_part_1','HAM10000_images_part_2']:
            fp = f"{d}/{p}/{iid}.jpg"
            if os.path.exists(fp): return fp
        return None
    df['image_path']  = df['image_id'].apply(find)
    df = df.dropna(subset=['image_path'])
    df['label']       = df['dx'].map(CLASS_MAP).fillna(7).astype(int)
    df['fitzpatrick'] = 0
    df['source']      = 'ham10000'
    df['age']         = pd.to_numeric(df['age'], errors='coerce').fillna(50)
    df['sex']         = df['sex'].fillna('unknown')
    print(f"   ✓ {len(df):,} images | classes: {df['dx'].value_counts().to_dict()}")
    return df[['image_path','label','fitzpatrick','age','sex','source']]

def load_isic2020():
    print("\n── Loading ISIC 2020 ──")
    d = f"{BASE}/melanoma ISIC 2020 dataset - 224"
    imgs = [f for f in os.listdir(d) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    df = pd.DataFrame([{'image_path':f"{d}/{f}",'label':1,'fitzpatrick':0,
                         'age':50,'sex':'unknown','source':'isic2020'} for f in imgs])
    print(f"   ✓ {len(df):,} images")
    return df

def load_padufes():
    print("\n── Loading PAD-UFES-20 ──")
    d = f"{BASE}/PAD-UFES-20"
    df = pd.read_csv(f"{d}/metadata.csv")

    # Build lookup across imgs_part_1, imgs_part_2, imgs_part_3
    lookup = {}
    for part in ['imgs_part_1','imgs_part_2','imgs_part_3']:
        part_dir = f"{d}/images/{part}"
        if not os.path.exists(part_dir): continue
        for fname in os.listdir(part_dir):
            lookup[fname] = f"{part_dir}/{fname}"
    print(f"   Found {len(lookup):,} images in imgs_part_1/2/3")

    df['image_path']  = df['img_id'].map(lookup)
    df = df.dropna(subset=['image_path'])
    df['label']       = df['diagnostic'].map(CLASS_MAP).fillna(7).astype(int)
    df['fitzpatrick'] = pd.to_numeric(df['fitspatrick'], errors='coerce').fillna(0).astype(int)
    df['age']         = pd.to_numeric(df['age'], errors='coerce').fillna(50)
    df['sex']         = df['gender'].fillna('unknown')
    df['source']      = 'padufes20'
    print(f"   ✓ {len(df):,} images | classes: {df['diagnostic'].value_counts().to_dict()}")
    print(f"   Fitzpatrick: {df['fitzpatrick'].value_counts().sort_index().to_dict()}")
    return df[['image_path','label','fitzpatrick','age','sex','source']]

def load_derm7pt():
    print("\n── Loading Derm7pt ──")
    d = f"{BASE}/release_v0"
    meta = pd.read_csv(f"{d}/meta/meta.csv")

    def find(fname):
        if not isinstance(fname, str): return None
        for ext in ['','.jpg','.jpeg','.png']:
            p = f"{d}/images/{fname}{ext}"
            if os.path.exists(p): return p
        return None

    img_col = 'derm' if 'derm' in meta.columns else meta.columns[0]
    meta['image_path'] = meta[img_col].apply(find)
    meta = meta.dropna(subset=['image_path'])

    def map_dx(dx):
        dx = str(dx).lower()
        if 'melanoma' in dx:                                    return 0
        if any(x in dx for x in ['nevus','clark','spitz','blue','congenital','dermal','combined','recurrent']): return 1
        if 'basal cell' in dx:                                  return 2
        if any(x in dx for x in ['seborrheic','lentigo']):      return 4
        if 'dermatofibroma' in dx:                              return 5
        if 'vascular' in dx:                                    return 6
        return 7

    dx_col = 'diagnosis' if 'diagnosis' in meta.columns else None
    meta['label']       = meta[dx_col].apply(map_dx) if dx_col else 7
    meta['fitzpatrick'] = 0
    meta['age']         = 50
    meta['sex']         = 'unknown'
    meta['source']      = 'derm7pt'
    print(f"   ✓ {len(meta):,} images")
    return meta[['image_path','label','fitzpatrick','age','sex','source']]

def load_midas():
    print("\n── Loading MIDAS ──")
    d = f"{BASE}/MIDAS/midasmultimodalimagedatasetforaibasedskincancer"
    imgs = [f for f in os.listdir(d) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    df = pd.DataFrame([{'image_path':f"{d}/{f}",
                         'label': 0 if 'mel' in f.lower() else 1,
                         'fitzpatrick':0,'age':50,'sex':'unknown','source':'midas'}
                        for f in imgs])
    print(f"   ✓ {len(df):,} images")
    return df

def load_melanoma_cancer():
    print("\n── Loading melanoma_cancer_dataset ──")
    d = f"{BASE}/melanoma_cancer_dataset"
    rows = []
    for split in ['train','test']:
        for cls in ['malignant','benign']:
            cls_dir = f"{d}/{split}/{cls}"
            if not os.path.exists(cls_dir): continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg','.jpeg','.png')):
                    rows.append({'image_path':f"{cls_dir}/{fname}",
                                 'label':CLASS_MAP[cls],'fitzpatrick':0,
                                 'age':50,'sex':'unknown','source':f'melanoma_{split}'})
    df = pd.DataFrame(rows)
    print(f"   ✓ {len(df):,} images | malignant:{(df['label']==0).sum()} benign:{(df['label']==1).sum()}")
    return df

def load_all_datasets():
    print("="*55)
    print("  DermoGraph-XAI — Loading All Datasets")
    print("="*55)

    dfs = []
    for loader in [load_ham10000, load_isic2020, load_padufes,
                   load_derm7pt, load_midas, load_melanoma_cancer]:
        try:
            dfs.append(loader())
        except Exception as e:
            print(f"   ✗ {loader.__name__}: {e}")

    combined = pd.concat(dfs, ignore_index=True)

    # Verify files exist
    combined['exists'] = combined['image_path'].apply(os.path.exists)
    missing = (~combined['exists']).sum()
    if missing: print(f"\n   ⚠ Removing {missing} missing paths")
    combined = combined[combined['exists']].drop(columns=['exists']).reset_index(drop=True)

    print("\n" + "="*55)
    print("  COMBINED DATASET SUMMARY")
    print("="*55)
    print(f"  Total images : {len(combined):,}")
    print(f"\n  By source:")
    for src, cnt in combined['source'].value_counts().items():
        print(f"    {src:<25} {cnt:>6,}")
    print(f"\n  By class:")
    for lbl, cnt in combined['label'].value_counts().sort_index().items():
        name = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else 'Other'
        pct  = cnt/len(combined)*100
        print(f"    {lbl} {name:<22} {cnt:>6,}  {pct:5.1f}%  {'█'*int(pct/2)}")
    fitzp = (combined['fitzpatrick']>0).sum()
    print(f"\n  Fitzpatrick labels : {fitzp:,} images")
    out = f"{OUTPUT}/combined_dataset.csv"
    combined.to_csv(out, index=False)
    print(f"  ✓ Saved: {out}")
    print("="*55)
    return combined

if __name__ == "__main__":
    df = load_all_datasets()
    print(f"\n✓ Final shape: {df.shape}")