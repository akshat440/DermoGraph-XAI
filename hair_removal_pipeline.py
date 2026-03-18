"""
hair_removal_pipeline.py
DermoGraph-XAI — Complete Hair Removal Pipeline
Matches Figure 4 from the paper: 7 visualization stages

Stages:
1. Original
2. BRG Mask
3. BW Original + Mask
4. Original + Mask
5. Original + Mask Alt
6. Original + BW Mask
7. BW Mask (final clean segmentation)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
#  CORE HAIR REMOVAL FUNCTION
# ═══════════════════════════════════════════════════════════════

def extract_hair_mask(img_bgr):
    """
    Extract hair mask using Blackhat morphological operation.
    Returns binary mask where white = hair pixels.
    """
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask  = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # Dilate slightly to cover full hair width
    dk       = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask     = cv2.dilate(mask, dk, iterations=1)
    return mask


def remove_hair(img_bgr):
    """
    Full hair removal: extract mask → Telea inpainting.
    Returns cleaned image (no hair).
    """
    mask = extract_hair_mask(img_bgr)
    return cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)


# ═══════════════════════════════════════════════════════════════
#  7 VISUALIZATION STAGES (Figure 4)
# ═══════════════════════════════════════════════════════════════

def generate_all_stages(img_bgr):
    """
    Generate all 7 stages described in Figure 4 of the paper.

    Returns dict with keys:
        original, brg_mask, bw_original_mask, original_mask,
        original_mask_alt, original_bw_mask, bw_mask
    """
    h, w   = img_bgr.shape[:2]
    mask   = extract_hair_mask(img_bgr)
    clean  = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)

    # ── 1. Original ───────────────────────────────────────────
    # Raw dermoscopy image with hair occlusions
    original = img_bgr.copy()

    # ── 2. BRG Mask ───────────────────────────────────────────
    # Color mask (Blue-Red-Green) highlighting edges & contours
    # Shows hair boundaries + lesion contours in color
    edges    = cv2.Canny(mask, 50, 150)
    brg_mask = np.zeros_like(img_bgr)
    # Blue channel: full hair mask
    brg_mask[:, :, 0] = mask
    # Red channel: edge detection of mask
    brg_mask[:, :, 2] = edges
    # Green channel: dilated edges (wider contour)
    dk = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    brg_mask[:, :, 1] = cv2.dilate(edges, dk, iterations=2)

    # ── 3. BW Original + Mask ─────────────────────────────────
    # Grayscale image overlaid with edge detection mask
    # Hair detail still visible in edges
    gray            = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_3ch        = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edges_3ch       = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    bw_original_mask= cv2.addWeighted(gray_3ch, 0.7, edges_3ch, 0.3, 0)

    # ── 4. Original + Mask ────────────────────────────────────
    # Contour line around lesion boundary on original image
    # Intermediate stage: lesion isolated, hair still present
    contours, _     = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
    original_mask   = img_bgr.copy()
    cv2.drawContours(original_mask, contours, -1, (0, 255, 0), 2)

    # ── 5. Original + Mask Alt ────────────────────────────────
    # Stronger contrast around lesion boundary
    # Clearer separation of lesion from hair area
    # Use thicker contour + semi-transparent overlay
    overlay          = img_bgr.copy()
    mask_3ch         = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_colored     = np.zeros_like(img_bgr)
    mask_colored[mask > 0] = [0, 0, 255]   # red hair regions
    original_mask_alt= cv2.addWeighted(overlay, 0.75, mask_colored, 0.25, 0)
    cv2.drawContours(original_mask_alt, contours, -1, (255, 255, 0), 3)

    # ── 6. Original + BW Mask ────────────────────────────────
    # Binary mask fades surrounding areas, lesion prominent
    # Strong contrast between lesion and background
    mask_inv         = cv2.bitwise_not(mask)
    faded            = img_bgr.copy()
    faded[mask_inv > 0] = (faded[mask_inv > 0] * 0.3).astype(np.uint8)
    original_bw_mask = faded

    # ── 7. BW Mask (Final clean segmentation) ─────────────────
    # White = lesion/clean region, Black = removed hair
    # This is what gets fed into ML models
    bw_mask = mask   # binary: hair=white, skin=black

    return {
        'original'          : original,
        'brg_mask'          : brg_mask,
        'bw_original_mask'  : bw_original_mask,
        'original_mask'     : original_mask,
        'original_mask_alt' : original_mask_alt,
        'original_bw_mask'  : original_bw_mask,
        'bw_mask'           : bw_mask,
        'clean'             : clean,   # final inpainted result
    }


# ═══════════════════════════════════════════════════════════════
#  FIGURE 4 VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def plot_figure4(img_bgr, save_path=None, title="Figure 4: Hair Removal Pipeline"):
    """
    Reproduce Figure 4 from the paper — all 7 stages in one plot.
    """
    stages = generate_all_stages(img_bgr)

    labels = [
        'Original',
        'BRG Mask',
        'BW Original + Mask',
        'Original + Mask',
        'Original + Mask Alt',
        'Original + BW Mask',
        'BW Mask',
    ]
    keys = [
        'original',
        'brg_mask',
        'bw_original_mask',
        'original_mask',
        'original_mask_alt',
        'original_bw_mask',
        'bw_mask',
    ]

    fig, axes = plt.subplots(1, 7, figsize=(24, 4))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

    for ax, key, label in zip(axes, keys, labels):
        img = stages[key]
        # Convert BGR → RGB for matplotlib (except BW mask)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img, cmap='gray' if len(stages[key].shape) == 2 else None)
        ax.set_title(label, fontsize=8, fontweight='bold', pad=4)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved Figure 4 to: {save_path}")
    plt.show()
    return stages


# ═══════════════════════════════════════════════════════════════
#  FAST BATCH PREPROCESSING (for Mac overnight / Kaggle)
# ═══════════════════════════════════════════════════════════════

def preprocess_dataset(input_dir, output_dir, size=224, n_workers=1):
    """
    Fast batch hair removal for entire dataset.
    Output: resized + hair-removed JPEGs at output_dir/
    
    Usage (Mac overnight):
        preprocess_dataset(
            input_dir  = '/Users/.../SKIN CANCER DATASET',
            output_dir = '/Users/.../dermograph-preprocessed-224',
        )
    """
    from tqdm import tqdm
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Find all images recursively
    exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    all_imgs = [
        p for p in Path(input_dir).rglob('*')
        if p.suffix in exts
    ]
    print(f"Found {len(all_imgs):,} images in {input_dir}")
    print(f"Output → {output_dir}")
    print(f"Size   → {size}×{size}")

    done = skipped = errors = 0

    for src in tqdm(all_imgs, desc="Hair removal + resize"):
        # Preserve relative path structure in filename
        rel   = src.relative_to(input_dir)
        # Flatten: replace path separators with __ to avoid conflicts
        flat  = str(rel).replace('/', '__').replace('\\', '__')
        dst   = Path(output_dir) / flat

        if dst.exists():
            skipped += 1
            continue

        img = cv2.imread(str(src))
        if img is None:
            errors += 1
            continue

        # Hair removal
        img = remove_hair(img)

        # Resize
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

        # Save
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        done += 1

    print(f"\n✓ Done     : {done:,}")
    print(f"  Skipped  : {skipped:,}")
    print(f"  Errors   : {errors}")
    print(f"\nNow zip and upload to Kaggle as 'dermograph-preprocessed-224'")
    print(f"All future notebooks: mount once, no preprocessing needed!")


# ═══════════════════════════════════════════════════════════════
#  QUICK DEMO
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Demo on a single image
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read: {img_path}")
            sys.exit(1)
        img = cv2.resize(img, (224, 224))
        plot_figure4(img, save_path="figure4_hair_removal.png")

    else:
        # Run full dataset preprocessing
        preprocess_dataset(
            input_dir  = "/Users/akshxunfiltered/DermoXAI/SKIN CANCER DATASET",
            output_dir = "/Users/akshxunfiltered/DermoXAI/dermograph-preprocessed-224",
            size       = 224,
        )