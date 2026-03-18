"""
preprocessing.py
DermoGraph-XAI — Hair Removal + Augmentation + Train/Val/Test Split

Run: python3 preprocessing.py
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

OUTPUT = "/Users/akshxunfiltered/DermoXAI/dermograph_output"
os.makedirs(OUTPUT, exist_ok=True)

CLASS_NAMES = ['Melanoma','Nevi','Basal Cell Carcinoma','Actinic Keratosis',
               'Benign Keratosis','Dermatofibroma','Vascular','Other']

# ══════════════════════════════════════════════════════════
# 1. HAIR REMOVAL
# Blackhat morphological filter (11×11) + Telea inpainting
# ══════════════════════════════════════════════════════════
def remove_hair(img_bgr: np.ndarray,
                kernel_size: int = 11,
                threshold: int = 10,
                inpaint_radius: int = 3) -> np.ndarray:
    """
    Remove hair artifacts from dermoscopy image.
    Input/Output: BGR numpy array (H, W, 3)
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat= cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    # Dilate mask slightly to cover full hair width
    dk      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask    = cv2.dilate(mask, dk, iterations=1)
    result  = cv2.inpaint(img_bgr, mask, inpaintRadius=inpaint_radius,
                          flags=cv2.INPAINT_TELEA)
    return result


def preprocess_image(image_path: str, size: int = 224) -> np.ndarray:
    """
    Full pipeline for one image:
    Load → Hair removal → Resize → RGB → Normalize [0,1]
    Returns float32 numpy array (H, W, 3)
    """
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((size, size, 3), dtype=np.float32)
    img = remove_hair(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


# ══════════════════════════════════════════════════════════
# 2. VERIFY HAIR REMOVAL — saves before/after samples
# ══════════════════════════════════════════════════════════
def verify_hair_removal(df: pd.DataFrame, n_samples: int = 6):
    """
    Save a before/after grid so you can see hair removal working.
    Picks images from HAM10000 (most likely to have hair artifacts).
    """
    import matplotlib.pyplot as plt

    print("\n── Verifying hair removal ──")
    # Prefer HAM10000 images (they have most hair)
    ham_df = df[df['source'] == 'ham10000'].sample(
        min(n_samples, len(df[df['source']=='ham10000'])), random_state=42)

    fig, axes = plt.subplots(n_samples, 2, figsize=(8, n_samples * 3))
    fig.patch.set_facecolor('#0a0e1a')
    fig.suptitle('Hair Removal — Before vs After', color='white',
                 fontsize=14, fontweight='bold', y=1.01)

    for i, (_, row) in enumerate(ham_df.iterrows()):
        path = row['image_path']
        original = cv2.imread(path)
        if original is None:
            continue
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        removed      = remove_hair(original)
        removed_rgb  = cv2.cvtColor(removed, cv2.COLOR_BGR2RGB)

        axes[i, 0].imshow(original_rgb)
        axes[i, 0].set_title('Original', color='white', fontsize=9)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(removed_rgb)
        axes[i, 1].set_title('Hair Removed', color='#00e5cc', fontsize=9)
        axes[i, 1].axis('off')

    plt.tight_layout()
    out = f"{OUTPUT}/hair_removal_samples.png"
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='#0a0e1a')
    print(f"   ✓ Saved: {out}")
    plt.show()


# ══════════════════════════════════════════════════════════
# 3. TRAIN / VAL / TEST SPLIT
# Stratified by class label
# 80% train / 10% val / 10% test
# ══════════════════════════════════════════════════════════
def make_splits(df: pd.DataFrame,
                train_ratio: float = 0.80,
                val_ratio:   float = 0.10,
                random_state: int  = 42) -> tuple:
    """
    Stratified split → (train_df, val_df, test_df)
    Stratified means each split has same class % as full dataset.
    """
    print("\n── Creating train/val/test splits ──")

    # First split off test (10%)
    train_val, test = train_test_split(
        df, test_size=0.10,
        stratify=df['label'],
        random_state=random_state
    )
    # Then split val from remaining (10% of total = 11.1% of train_val)
    train, val = train_test_split(
        train_val, test_size=0.111,
        stratify=train_val['label'],
        random_state=random_state
    )

    print(f"   Train : {len(train):>6,} images ({len(train)/len(df)*100:.1f}%)")
    print(f"   Val   : {len(val):>6,} images ({len(val)/len(df)*100:.1f}%)")
    print(f"   Test  : {len(test):>6,} images ({len(test)/len(df)*100:.1f}%)")

    # Verify class balance in each split
    print(f"\n   Class distribution check:")
    print(f"   {'Class':<22} {'Train%':>7} {'Val%':>7} {'Test%':>7}")
    print(f"   {'-'*46}")
    for lbl in sorted(df['label'].unique()):
        name  = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else 'Other'
        tp    = (train['label']==lbl).sum()/len(train)*100
        vp    = (val['label']==lbl).sum()/len(val)*100
        tep   = (test['label']==lbl).sum()/len(test)*100
        print(f"   {name:<22} {tp:>6.1f}% {vp:>6.1f}% {tep:>6.1f}%")

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ══════════════════════════════════════════════════════════
# 4. CLASS WEIGHTS (for focal loss in training)
# ══════════════════════════════════════════════════════════
def compute_class_weights(train_df: pd.DataFrame) -> dict:
    """
    Inverse frequency weights: w_c = N / (C × N_c)
    Used in Kaggle training notebook to handle class imbalance.
    """
    print("\n── Computing class weights ──")
    n_total   = len(train_df)
    n_classes = train_df['label'].nunique()
    weights   = {}

    for lbl in sorted(train_df['label'].unique()):
        n_c       = (train_df['label'] == lbl).sum()
        w         = n_total / (n_classes * n_c)
        weights[lbl] = round(w, 4)
        name      = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else 'Other'
        print(f"   Class {lbl} {name:<22} n={n_c:>6,}  weight={w:.4f}")

    return weights


# ══════════════════════════════════════════════════════════
# 5. QUICK SANITY CHECK — load and display 1 image
# ══════════════════════════════════════════════════════════
def sanity_check(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    print("\n── Sanity check: loading 8 random images ──")
    sample = df.sample(8, random_state=42)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor('#0a0e1a')
    axes = axes.flatten()

    for i, (_, row) in enumerate(sample.iterrows()):
        img = preprocess_image(row['image_path'], size=224)
        axes[i].imshow(img)
        name = CLASS_NAMES[row['label']] if row['label'] < len(CLASS_NAMES) else 'Other'
        axes[i].set_title(f"{name}\n{row['source']}", color='white', fontsize=8)
        axes[i].axis('off')

    plt.suptitle('Random Sample — After Preprocessing (Hair Removed + 224×224)',
                 color='white', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = f"{OUTPUT}/sample_images.png"
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='#0a0e1a')
    print(f"   ✓ Saved: {out}")
    plt.show()


# ══════════════════════════════════════════════════════════
# 6. SAVE SPLITS TO CSV
# These CSVs are used by the Kaggle training notebook
# ══════════════════════════════════════════════════════════
def save_splits(train_df, val_df, test_df, weights):
    train_df.to_csv(f"{OUTPUT}/train.csv", index=False)
    val_df.to_csv(f"{OUTPUT}/val.csv",     index=False)
    test_df.to_csv(f"{OUTPUT}/test.csv",   index=False)

    import json
    with open(f"{OUTPUT}/class_weights.json", 'w') as f:
        json.dump(weights, f, indent=2)

    print(f"\n   ✓ Saved train.csv  ({len(train_df):,} rows)")
    print(f"   ✓ Saved val.csv    ({len(val_df):,} rows)")
    print(f"   ✓ Saved test.csv   ({len(test_df):,} rows)")
    print(f"   ✓ Saved class_weights.json")


# ══════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Load the combined CSV saved by dataset_loader.py
    csv_path = f"{OUTPUT}/combined_dataset.csv"
    if not os.path.exists(csv_path):
        print("❌ combined_dataset.csv not found!")
        print("   Run dataset_loader.py first.")
        exit(1)

    print("Loading combined_dataset.csv...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} images\n")

    # Step 1: Verify hair removal visually
    verify_hair_removal(df, n_samples=6)

    # Step 2: Train/val/test split
    train_df, val_df, test_df = make_splits(df)

    # Step 3: Class weights for training
    weights = compute_class_weights(train_df)

    # Step 4: Sanity check — display samples
    sanity_check(df)

    # Step 5: Save splits
    print("\n── Saving splits ──")
    save_splits(train_df, val_df, test_df, weights)

    print("\n" + "="*55)
    print("  ✓ Preprocessing complete!")
    print("  Next: Upload train.csv/val.csv/test.csv to Kaggle")
    print("        and run the training notebook")
    print("="*55)