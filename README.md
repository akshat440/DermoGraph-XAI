# DermoGraph-XAI 🔬

> **Multi-Dataset Skin Lesion Classification with Explainable AI**  
> Team 8 | VIT Bhopal | B.Tech Final Year Project

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Datasets-20BEFF.svg)](https://www.kaggle.com/akshat23029)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

DermoGraph-XAI is a comprehensive deep learning framework for automated skin lesion classification across **7 diagnostic categories**. The system benchmarks **12+ CNN and transformer architectures** on a unified multi-dataset corpus and introduces **DermoNet** — a novel hybrid architecture combining Dual-Scale CNNs, Lesion-Aware Attention Gates (LAAG), and Multi-Resolution Transformer Blocks (MRTB).

### 7-Class Classification
| Class | Label | Risk | Description |
|---|---|---|---|
| Melanoma | 0 | 🔴 HIGH | Malignant melanocytic tumor |
| Nevi | 1 | 🟢 LOW | Benign melanocytic nevus |
| Basal Cell Carcinoma | 2 | 🔴 HIGH | Most common skin cancer |
| Actinic Keratosis | 3 | 🟡 MEDIUM | Precancerous lesion / SCC |
| Benign Keratosis | 4 | 🟢 LOW | Seborrheic keratosis |
| Dermatofibroma | 5 | 🟢 LOW | Benign fibrous nodule |
| Vascular | 6 | 🟢 LOW | Vascular lesion |

---

## 📊 Benchmark Results

| Model | Accuracy | F1 Macro | AUC-ROC | Params | Type |
|---|---|---|---|---|---|
| VGG16 | 80.48% | 0.7102 | 0.9601 | 138M | CNN Baseline |
| MobileNetV2 | 83.74% | 0.7334 | 0.9805 | 3.4M | Lightweight CNN |
| ResNet50 | 87.40% | 0.7261 | 0.9823 | 25M | CNN Baseline |
| DenseNet121 | 87.69% | 0.7663 | 0.9866 | 8M | Dense CNN |
| EfficientNet-B0 | 89.37% | 0.7747 | 0.9850 | 5.3M | Efficient CNN |
| EfficientNet-B3 | 90.70% | 0.8234 | 0.9845 | 12M | Efficient CNN |
| MaxViT-T | 91.98% | 0.8325 | 0.9840 | 31M | Transformer |
| **Swin-T** ⭐ | **92.66%** | **0.8493** | **0.9895** | **28M** | **Transformer** |
| Swin-T Fine-tuned | 🔄 | — | — | 28M | Transformer |
| DermoNet (ours) | 🔄 | — | — | ~18M | Novel |

> All baseline models trained on the same 6-dataset corpus (35,084 images) with identical setup for fair comparison.

### Swin-T Per-Class Performance (Current SOTA)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Melanoma | 0.82 | 0.95 | 0.88 | 636 |
| Nevi | 0.98 | 0.94 | 0.96 | 2469 |
| Basal Cell Carcinoma | 0.79 | 0.89 | 0.84 | 133 |
| Actinic Keratosis | 0.84 | 0.77 | 0.80 | 125 |
| Benign Keratosis | 0.80 | 0.80 | 0.80 | 133 |
| Dermatofibroma ⚠️ | 0.80 | 0.89 | 0.84 | 9 |
| Vascular ⚠️ | 0.73 | 0.92 | 0.81 | 12 |

> ⚠️ Dermatofibroma (n=9) and Vascular (n=12) have very limited test samples — Phase 2 fine-tuning targets these specifically.

---

## 📊 Results

### Class Distribution
![Class Distribution](assets/results/01_class_distribution.png)

### Best Model — Swin-T (92.66%)
![MaxViT TP/FP/FN/TN](assets/results/maxvit_t_tp_fp_fn_tn.png)

### Confusion Matrix
![MaxViT Confusion Matrix](assets/results/maxvit_t_cm_viridis.png)

### Training Curves
![MaxViT Curves](assets/results/maxvit_t_curves.png)

### EfficientNet-B0 (89.37%)
![EfficientNet TP/FP/FN/TN](assets/results/efficientnet_b0_tp_fp_fn_tn.png)

### ResNet50 Baseline
![ResNet50 Analysis](assets/results/resnet50_tp_fp_fn_tn.png)

### Hair Removal Pipeline
![Hair Removal](assets/results/hair_removal_samples.png)

---

## 🗃️ Datasets

### Phase 1 — Core Training (6 datasets — 35,084 images)

| Dataset | Images | Classes | Kaggle Link |
|---|---|---|---|
| HAM10000 | 10,015 | 7 | [dermograph-ham-images](https://www.kaggle.com/datasets/akshat23029/dermograph-ham-images) |
| ISIC 2020 | 8,757 | 2 | [dermograph-isic2020](https://www.kaggle.com/datasets/akshat23029/dermograph-isic2020) |
| PAD-UFES-20 | 2,298 | 6 | [dermograph-pad-images](https://www.kaggle.com/datasets/akshat23029/dermograph-pad-images) |
| Melanoma Cancer | 10,605 | 2 | [dermograph-melanoma-cancer](https://www.kaggle.com/datasets/akshat23029/dermograph-melanoma-cancer) |
| MIDAS | 3,411 | 7 | [dermograph-midas](https://www.kaggle.com/datasets/akshat23029/dermograph-midas) |
| Train/Val/Test Splits | — | — | [dermograph-splits](https://www.kaggle.com/datasets/akshat23029/dermograph-splits) |

### Phase 2 — Minority Class Augmentation (2 datasets — 44,277 images)

| Dataset | Images | DF added | VASC added | BCC added | Kaggle Link |
|---|---|---|---|---|---|
| ISIC 2019 | 25,331 | 239 (26×) | 253 (21×) | 3,323 (25×) | [isic-2019-skin-lesion-images](https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification) |
| BCN20000 | 18,946 | ~200 | ~200 | ~2,000 | [bcn20000](https://www.kaggle.com/datasets/mathieubecher/bcn20000) |

```
Phase 1 + Phase 2 combined: ~79,361 images total

Minority class improvement:
  Dermatofibroma  :   9 → 448  samples  (49× increase)
  Vascular Lesion :  12 → 465  samples  (39× increase)
  BCC             : 133 → 5,456 samples  (41× increase)
```

### Extended Datasets (Innovation Modules)

| Dataset | Images | Purpose | Link |
|---|---|---|---|
| FitzPatrick17k | 16,577 | Fairness MTL — skin tone I–VI | [dermograph-fitzpatrick](https://www.kaggle.com/datasets/akshat23029/dermograph-fitzpatrick) |
| Derm7pt | 1,011 | ABCDE Branch — 7-point checklist | [dermograph-derm7pt](https://www.kaggle.com/datasets/akshat23029/dermograph-derm7pt) |

---

## 🏗️ Project Structure

```
DermoGraph-XAI/
├── notebooks/
│   ├── VGG16_DermoGraph.ipynb
│   ├── ResNet50_DermoGraph.ipynb
│   ├── DenseNet121_DermoGraph.ipynb
│   ├── MaxViT_DermoGraph_v2.ipynb
│   ├── MobileNetV2_DermoGraph.ipynb
│   ├── EfficientNet_B0_DermoGraph.ipynb
│   ├── EfficientNet_B3_DermoGraph.ipynb
│   ├── EfficientNetV2_S_DermoGraph.ipynb
│   ├── ConvNeXt_Small_DermoGraph.ipynb
│   ├── Swin_T_DermoGraph.ipynb            ← 🏆 Current SOTA 92.66%
│   ├── Swin_T_FineTuned_DermoGraph.ipynb  ← 🔄 Phase 2 minority class
│   ├── ViT_B16_DermoGraph.ipynb
│   ├── ResNeXt50_DermoGraph.ipynb
│   ├── DenseNet169_DermoGraph.ipynb
│   ├── RegNetY_8GF_DermoGraph.ipynb
│   └── DermoNet_v2.ipynb                  ← Novel architecture
├── dermograph/
│   ├── backend/                           ← FastAPI + model inference
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   └── weights/                       ← .pth files
│   └── frontend/                          ← React + Tailwind UI
│       └── src/
├── assets/results/                        ← Training visualizations
├── dermograph_output/                     ← JSON benchmark results
├── hair_removal_pipeline.py
├── dataset_loader.py
└── README.md
```

---

## 🚀 Quick Start

### Run Full System Locally

**Backend:**
```bash
cd dermograph/backend
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# API docs: http://localhost:8000/docs
```

**Frontend:**
```bash
cd dermograph/frontend
npm install && npm run dev
# Open: http://localhost:3000
```

### Train on Kaggle
1. Create new Kaggle notebook
2. Add datasets via **+ Add Input**
3. Upload notebook from `notebooks/`
4. Enable GPU T4
5. Run all cells

---

## 🧠 DermoNet — Novel Architecture

| Component | Innovation |
|---|---|
| DualScaleStem | Parallel 3×3 (fine) + 7×7 (coarse) CNN with learned fusion weights |
| LAAG ×2 | Lesion-Aware Attention Gate — channel + spatial + border attention |
| MRTB ×6 | Multi-Resolution Transformer — 3-scale attention (fine + mid + coarse) |

---

## ⚙️ Training Configuration

```python
# Phase 1 — Baseline models
BATCH_SIZE = 32 | N_EPOCHS = 50 | PATIENCE = 15
OPTIMIZER  = AdamW(lr=1e-4, wd=1e-2)
SCHEDULER  = CosineAnnealingLR(T_max=50, eta_min=1e-6)
LOSS       = CrossEntropyLoss(class_weights)
IMAGE_SIZE = 224×224

# Phase 2 — Swin-T minority class fine-tuning
BASE       = swin_t_best.pth (92.66% checkpoint)
DATASETS   = All 6 original + ISIC2019 + BCN20000
LOSS       = FocalLoss(gamma=2, alpha=class_weights)
N_EPOCHS   = 30 | LR = 1e-5
TARGET     = DF + Vascular + BCC (extreme overweighting)
```

---

## 📈 Innovation Modules

| Module | Description | Status |
|---|---|---|
| **Minority Fine-tuning** | Swin-T + ISIC2019 + BCN20000 targeting DF/VASC/BCC | 🔄 In Progress |
| ABCDE Branch | 7-point checklist scoring via Derm7pt | 📋 Planned |
| GAT Pattern Graph | Graph Attention for lesion relationships | 📋 Planned |
| Neural ODE | Continuous-depth lesion evolution | 📋 Planned |
| Fairness MTL | Skin tone fairness via FitzPatrick I–VI | 📋 Planned |

---

## 📚 Dataset Citations

| Dataset | Citation | License |
|---|---|---|
| HAM10000 | Tschandl et al., Scientific Data 2018 | CC BY-NC-SA 4.0 |
| ISIC 2019 | Combalia et al., arXiv 2019 | CC BY-NC-SA 4.0 |
| ISIC 2020 | Rotemberg et al., Scientific Data 2021 | CC BY-NC-SA 4.0 |
| BCN20000 | Combalia et al., arXiv 2019 | CC BY-NC-SA 4.0 |
| PAD-UFES-20 | Pacheco et al., Data in Brief 2020 | CC BY 4.0 |
| MIDAS | Kaggle Community Dataset | See source |
| Melanoma Cancer | SIIM-ISIC Challenge 2020 | CC BY-NC-SA 4.0 |
| FitzPatrick17k | Groh et al., CVPR Workshop 2021 | MIT |
| Derm7pt | Kawahara et al., IEEE JBHI 2019 | See source |

> All datasets used strictly for academic research. We do not own any of these datasets.

---

## 👥 Team

**Team 8 — VIT Bhopal** | B.Tech Final Year Project | Computer Science

---

## 📄 Citation

```bibtex
@misc{dermographxai2026,
  title  = {DermoGraph-XAI: Multi-Dataset Skin Lesion Classification with Explainable AI},
  author = {Team 8, VIT Bhopal},
  year   = {2026},
  url    = {https://github.com/akshat440/DermoGraph-XAI}
}
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

> **Disclaimer:** For research purposes only. Not a substitute for professional medical diagnosis.
