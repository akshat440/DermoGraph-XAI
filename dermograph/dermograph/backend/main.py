"""
DermoGraph-XAI Backend — FastAPI
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io, os, json, time, cv2, base64
from datetime import datetime
import timm

app = FastAPI(
    title="DermoGraph-XAI API",
    description="Skin Lesion Classification with Explainable AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ──────────────────────────────────────────────
CLASS_NAMES = [
    "Melanoma",
    "Nevi",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Benign Keratosis",
    "Dermatofibroma",
    "Vascular Lesion"
]

CLASS_INFO = {
    "Melanoma": {
        "risk": "HIGH", "color": "#ef4444", "icd": "C43",
        "description": "Malignant melanocytic tumor. Immediate dermatologist consultation recommended.",
        "abcde": ["Often asymmetric", "Irregular/notched border", "Multiple colors (brown, black, red, white)", "Diameter >6mm", "Evolving over time"]
    },
    "Nevi": {
        "risk": "LOW", "color": "#22c55e", "icd": "D22",
        "description": "Benign melanocytic nevus (mole). Regular monitoring recommended.",
        "abcde": ["Symmetric", "Regular smooth border", "Uniform brown color", "Diameter <6mm", "Stable — not changing"]
    },
    "Basal Cell Carcinoma": {
        "risk": "HIGH", "color": "#f97316", "icd": "C44.91",
        "description": "Most common skin cancer. Highly treatable when caught early.",
        "abcde": ["Variable asymmetry", "Pearly/rolled border", "Translucent/pink color", "Variable size", "Slow growing"]
    },
    "Actinic Keratosis": {
        "risk": "MEDIUM", "color": "#eab308", "icd": "L57.0",
        "description": "Precancerous lesion caused by UV damage. Treatment advised to prevent progression.",
        "abcde": ["Variable", "Rough/scaly border", "Pink/red/brown color", "Usually small (<1cm)", "May itch or bleed"]
    },
    "Benign Keratosis": {
        "risk": "LOW", "color": "#22c55e", "icd": "L82",
        "description": "Seborrheic keratosis — non-cancerous warty growth. No treatment required.",
        "abcde": ["Symmetric", "Well-defined border", "Tan/brown/black", "Variable size", "Stable — waxy appearance"]
    },
    "Dermatofibroma": {
        "risk": "LOW", "color": "#22c55e", "icd": "D23",
        "description": "Benign fibrous skin nodule. Usually no treatment required.",
        "abcde": ["Symmetric", "Regular border", "Brown/reddish-brown", "Small (3-10mm)", "Stable — firm to touch"]
    },
    "Vascular Lesion": {
        "risk": "LOW", "color": "#3b82f6", "icd": "D18",
        "description": "Vascular skin lesion including hemangiomas and pyogenic granulomas.",
        "abcde": ["Variable", "Well-defined border", "Red/purple/blue color", "Variable size", "May bleed easily"]
    }
}

MODELS_CONFIG = {
    "maxvit_t": {
        "name": "MaxViT-T", "timm_str": "maxvit_tiny_tf_224",
        "accuracy": 91.98, "f1": 0.8325, "auc": 0.9840,
        "params": "31M", "type": "Transformer",
        "description": "Best performing model — Multi-Axis Vision Transformer"
    },
    "efficientnet_b3": {
        "name": "EfficientNet-B3", "timm_str": "efficientnet_b3",
        "accuracy": 90.70, "f1": 0.8234, "auc": 0.9845,
        "params": "12M", "type": "CNN",
        "description": "Best efficiency — EfficientNet family"
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0", "timm_str": "efficientnet_b0",
        "accuracy": 89.37, "f1": 0.7747, "auc": 0.9850,
        "params": "5.3M", "type": "CNN",
        "description": "Lightweight — fastest inference"
    },
    "densenet121": {
        "name": "DenseNet121", "timm_str": "densenet121",
        "accuracy": 87.69, "f1": 0.7663, "auc": 0.9866,
        "params": "8M", "type": "CNN",
        "description": "Dense connections for feature reuse"
    },
    "resnet50": {
        "name": "ResNet50", "timm_str": "resnet50",
        "accuracy": 87.40, "f1": 0.7261, "auc": 0.9823,
        "params": "25M", "type": "CNN",
        "description": "Classic residual network baseline"
    },
}

# ── Model Manager ──────────────────────────────────────────
class ModelManager:
    def __init__(self):
        self.models   = {}
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        print(f"ModelManager initialized on {self.device}")

    def load_model(self, model_key: str, weights_path: str):
        config = MODELS_CONFIG.get(model_key)
        if not config:
            raise ValueError(f"Unknown model key: {model_key}")
        model = timm.create_model(config["timm_str"], pretrained=False, num_classes=7)
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        self.models[model_key] = model
        print(f"✓ Loaded {config['name']} from {weights_path}")
        return True

    def load_from_folder(self, folder: str):
        loaded = []
        mapping = {
            "maxvit_t_best.pth":        "maxvit_t",
            "efficientnet_b3_best.pth": "efficientnet_b3",
            "efficientnet_b0_best.pth": "efficientnet_b0",
            "densenet121_best.pth":     "densenet121",
            "resnet50_best.pth":        "resnet50",
        }
        for fname, key in mapping.items():
            path = os.path.join(folder, fname)
            if os.path.exists(path):
                try:
                    self.load_model(key, path)
                    loaded.append(key)
                except Exception as e:
                    print(f"✗ Failed to load {fname}: {e}")
        return loaded

    def predict(self, image: Image.Image, model_key: str):
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not loaded")

        img_t = self.transform(image).unsqueeze(0).to(self.device)
        t0    = time.time()
        with torch.no_grad():
            logits = self.models[model_key](img_t).float()
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
        ms = (time.time() - t0) * 1000

        pred_idx   = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]

        return {
            "predicted_class": pred_class,
            "predicted_index": pred_idx,
            "confidence":      round(float(probs[pred_idx]) * 100, 2),
            "probabilities":   {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(7)},
            "inference_ms":    round(ms, 1),
            "class_info":      CLASS_INFO[pred_class],
        }

    def gradcam(self, image: Image.Image, model_key: str):
        """GradCAM — works for both CNN and transformer models"""
        if model_key not in self.models:
            return None

        model = self.models[model_key]

        try:
            img_t = self.transform(image).unsqueeze(0).to(self.device)
            img_t.requires_grad_(True)

            gradients  = []
            activations = []

            def save_grad(grad):
                gradients.append(grad.detach())

            def save_act(module, inp, out):
                out_d = out.detach()
                out_d.requires_grad_(True)
                activations.append(out_d)
                out_d.register_hook(save_grad)

            # Find best target layer
            target_layer = None
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module  # keep updating — gets last conv

            if target_layer is None:
                # For pure transformers — use a linear layer
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear) and "head" not in name.lower():
                        target_layer = module

            if target_layer is None:
                return None

            hook = target_layer.register_forward_hook(save_act)

            try:
                model.zero_grad()
                with torch.enable_grad():
                    out      = model(img_t).float()
                    pred_idx = out.argmax(dim=1).item()
                    score    = out[0, pred_idx]
                    score.backward()

                if not gradients or not activations:
                    return None

                grad  = gradients[0]
                activ = activations[0]

                # Handle different tensor shapes
                # Conv: (B, C, H, W) | Transformer: (B, N, C) or (B, C)
                if grad.dim() == 4:
                    # CNN path
                    weights = grad.mean(dim=[2, 3])[0]
                    cam     = (weights[:, None, None] * activ[0]).sum(0)
                elif grad.dim() == 3:
                    # Transformer path (B, N, C)
                    weights = grad.mean(dim=1)[0]
                    activ_2d = activ[0].mean(dim=-1)  # (N,)
                    h = w = int(activ_2d.shape[0] ** 0.5)
                    if h * w == activ_2d.shape[0]:
                        cam = activ_2d.reshape(h, w)
                    else:
                        cam = activ_2d[:h*w].reshape(h, w)
                else:
                    return None

                cam = cam.cpu().numpy().astype(np.float32)
                cam = np.maximum(cam, 0)
                if cam.max() > 0:
                    cam = cam / cam.max()

                # Resize to 224×224
                cam_r = cv2.resize(cam, (224, 224))

                # Overlay on original
                img_arr = np.array(image.resize((224, 224)))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = np.clip(0.6 * img_arr + 0.4 * heatmap, 0, 255).astype(np.uint8)

                _, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                return "data:image/png;base64," + base64.b64encode(buf).decode()

            finally:
                hook.remove()

        except Exception as e:
            print(f"GradCAM error ({model_key}): {e}")
            return None

    def ensemble_predict(self, image: Image.Image, model_keys: list):
        all_probs = []
        results   = {}

        for key in model_keys:
            if key in self.models:
                r     = self.predict(image, key)
                probs = [r["probabilities"][c] for c in CLASS_NAMES]
                all_probs.append(probs)
                results[key] = r

        if not all_probs:
            raise ValueError("No models available for ensemble")

        avg       = np.mean(all_probs, axis=0)
        pred_idx  = int(np.argmax(avg))
        pred_cls  = CLASS_NAMES[pred_idx]
        conf      = float(avg[pred_idx])

        return {
            "predicted_class":    pred_cls,
            "predicted_index":    pred_idx,
            "confidence":         round(conf, 2),
            "probabilities":      {CLASS_NAMES[i]: round(float(avg[i]), 2) for i in range(7)},
            "individual_results": results,
            "models_used":        list(results.keys()),
            "class_info":         CLASS_INFO[pred_cls],
        }


# ── Initialize ─────────────────────────────────────────────
manager      = ModelManager()
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "./weights")

if os.path.exists(WEIGHTS_PATH):
    loaded = manager.load_from_folder(WEIGHTS_PATH)
    print(f"✓ Auto-loaded: {loaded}")
else:
    print(f"⚠ Weights folder not found: {WEIGHTS_PATH}")


# ── Routes ─────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "DermoGraph-XAI API",
        "version": "1.0.0",
        "status": "running",
        "loaded_models": list(manager.models.keys()),
        "device": str(manager.device),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(manager.device),
        "models_loaded": len(manager.models),
        "loaded_models": list(manager.models.keys()),
    }


@app.get("/models")
def get_models():
    return {
        "models": [
            {"key": key, "loaded": key in manager.models, **config}
            for key, config in MODELS_CONFIG.items()
        ]
    }


@app.get("/classes")
def get_classes():
    return {
        "classes": [
            {"index": i, "name": name, **CLASS_INFO[name]}
            for i, name in enumerate(CLASS_NAMES)
        ]
    }


@app.get("/benchmark")
def get_benchmark():
    return {
        "benchmark": [
            {"model": "VGG16",           "accuracy": 80.48, "f1": 0.7102, "auc": 0.9601, "params": "138M", "type": "CNN"},
            {"model": "MobileNetV2",     "accuracy": 83.74, "f1": 0.7334, "auc": 0.9805, "params": "3.4M", "type": "CNN"},
            {"model": "ResNet50",        "accuracy": 87.40, "f1": 0.7261, "auc": 0.9823, "params": "25M",  "type": "CNN"},
            {"model": "DenseNet121",     "accuracy": 87.69, "f1": 0.7663, "auc": 0.9866, "params": "8M",   "type": "CNN"},
            {"model": "EfficientNet-B0", "accuracy": 89.37, "f1": 0.7747, "auc": 0.9850, "params": "5.3M", "type": "CNN"},
            {"model": "EfficientNet-B3", "accuracy": 90.70, "f1": 0.8234, "auc": 0.9845, "params": "12M",  "type": "CNN"},
            {"model": "MaxViT-T",        "accuracy": 91.98, "f1": 0.8325, "auc": 0.9840, "params": "31M",  "type": "Transformer"},
        ]
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_key: str   = "maxvit_t",
    gradcam: bool    = False
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if model_key not in manager.models:
        available = list(manager.models.keys())
        if not available:
            raise HTTPException(status_code=503, detail="No models loaded. Place .pth files in ./weights/")
        model_key = available[0]

    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents)).convert("RGB")
        result   = manager.predict(image, model_key)
        result.update({
            "model_used":  model_key,
            "model_name":  MODELS_CONFIG[model_key]["name"],
            "timestamp":   datetime.now().isoformat(),
            "image_size":  f"{image.width}×{image.height}",
        })
        if gradcam:
            result["gradcam_image"] = manager.gradcam(image, model_key)

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/ensemble")
async def predict_ensemble(
    file: UploadFile = File(...),
    models: str      = "maxvit_t,efficientnet_b3"
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        contents   = await file.read()
        image      = Image.open(io.BytesIO(contents)).convert("RGB")
        model_keys = [m.strip() for m in models.split(",")]
        result     = manager.ensemble_predict(image, model_keys)
        result["timestamp"]  = datetime.now().isoformat()
        result["image_size"] = f"{image.width}×{image.height}"
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
