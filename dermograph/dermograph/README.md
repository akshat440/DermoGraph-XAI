# DermoGraph-XAI — Full Stack System

## Quick Start

### 1. Backend (FastAPI)

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your trained model weights
mkdir -p weights
# Copy your .pth files:
# weights/maxvit_t_best.pth
# weights/efficientnet_b3_best.pth
# weights/efficientnet_b0_best.pth
# weights/densenet121_best.pth
# weights/resnet50_best.pth

# Start the API server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API will be live at: http://localhost:8000
API docs at: http://localhost:8000/docs

### 2. Frontend (React + Vite)

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Frontend will be live at: http://localhost:3000

---

## Project Structure

```
dermograph/
├── backend/
│   ├── main.py              ← FastAPI app + all endpoints
│   ├── requirements.txt
│   ├── start.sh             ← Quick start script
│   └── weights/             ← Place .pth files here
│       ├── maxvit_t_best.pth
│       ├── efficientnet_b3_best.pth
│       ├── efficientnet_b0_best.pth
│       ├── densenet121_best.pth
│       └── resnet50_best.pth
│
└── frontend/
    ├── src/
    │   ├── App.jsx              ← Router + Navbar
    │   ├── main.jsx
    │   ├── index.css
    │   ├── pages/
    │   │   ├── AnalyzePage.jsx  ← Main upload + predict page
    │   │   ├── DashboardPage.jsx← Benchmark results
    │   │   ├── ModelsPage.jsx   ← All models + innovation modules
    │   │   └── ResearchPage.jsx ← Citations + team
    │   └── components/
    │       ├── ResultsPanel.jsx ← Prediction results display
    │       └── ModelSelector.jsx← Model selection UI
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── index.html
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info + loaded models |
| GET | `/health` | Health check |
| GET | `/models` | All models + status |
| GET | `/benchmark` | Benchmark results |
| GET | `/classes` | Class names + info |
| POST | `/predict` | Single model prediction |
| POST | `/predict/ensemble` | Ensemble prediction |

### Example — Predict
```bash
curl -X POST "http://localhost:8000/predict?model_key=maxvit_t&gradcam=true" \
  -F "file=@your_image.jpg"
```

---

## Weights Setup

Download your trained .pth files from Kaggle and place them in `backend/weights/`:

| File | Model | Accuracy |
|---|---|---|
| `maxvit_t_best.pth` | MaxViT-T | 91.98% |
| `efficientnet_b3_best.pth` | EfficientNet-B3 | 90.70% |
| `efficientnet_b0_best.pth` | EfficientNet-B0 | 89.37% |
| `densenet121_best.pth` | DenseNet121 | 87.69% |
| `resnet50_best.pth` | ResNet50 | 87.40% |

The API auto-loads all .pth files on startup.

---

## Deploy

### Frontend → Vercel
```bash
cd frontend
npm run build
# Push to GitHub → connect to Vercel → auto-deploy
```

### Backend → Railway / Render
```bash
# Set environment variable:
WEIGHTS_PATH=/app/weights

# Railway: connect GitHub repo, set start command:
uvicorn main:app --host 0.0.0.0 --port $PORT
```
