#!/bin/bash
# DermoGraph-XAI Backend Startup Script

echo "🔬 DermoGraph-XAI Backend"
echo "========================="

# Check if weights folder exists
if [ ! -d "./weights" ]; then
    echo "⚠️  Creating weights folder..."
    mkdir -p weights
    echo "   Place your .pth files here:"
    echo "   weights/maxvit_t_best.pth"
    echo "   weights/efficientnet_b3_best.pth"
    echo "   weights/efficientnet_b0_best.pth"
    echo "   weights/densenet121_best.pth"
    echo "   weights/resnet50_best.pth"
fi

# Install deps if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Start server
echo ""
echo "🚀 Starting FastAPI server..."
echo "   API docs: http://localhost:8000/docs"
echo "   Health:   http://localhost:8000/health"
echo ""

export WEIGHTS_PATH="./weights"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
