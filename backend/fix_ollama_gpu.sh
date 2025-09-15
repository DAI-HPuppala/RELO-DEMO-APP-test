#!/bin/bash
# Fix Ollama GPU loading issue

echo "========================================================================"
echo "FIXING OLLAMA GPU LOADING ISSUE"
echo "========================================================================"

# Step 1: Stop current Ollama service
echo "📍 Step 1: Stopping current Ollama service..."
pkill -f "ollama serve" 2>/dev/null
sleep 2

# Step 2: Set environment variables
echo "📍 Step 2: Setting CUDA environment variables..."
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_CUDA_COMPUTE_CAP=8.6
export OLLAMA_GPU_MEMORY_FRACTION=0.85
export CUDA_CACHE_DISABLE=0
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Step 3: Check GPU availability
echo "📍 Step 3: Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

# Step 4: Start Ollama with environment variables
echo "📍 Step 4: Starting Ollama with GPU support..."
nohup ollama serve > /tmp/ollama_gpu.log 2>&1 &
OLLAMA_PID=$!
echo "   Ollama started with PID: $OLLAMA_PID"

# Step 5: Wait for Ollama to be ready
echo "📍 Step 5: Waiting for Ollama to be ready..."
for i in {1..10}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "   Ollama is ready!"
        break
    fi
    sleep 1
done

# Step 6: Test GPU inference
echo "📍 Step 6: Testing GPU inference..."
curl -X POST http://localhost:11434/api/generate \
    -d '{
        "model": "qwen2.5vl:3b",
        "prompt": "GPU test",
        "stream": false,
        "options": {
            "num_gpu": -1,
            "gpu_layers": 37,
            "f16_kv": true,
            "use_mmap": true,
            "num_predict": 5
        }
    }' 2>/dev/null | jq -r '.response' | head -2

# Step 7: Check GPU usage
echo "📍 Step 7: Checking GPU memory usage..."
sleep 1
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Step 8: Check Ollama logs for GPU loading
echo "📍 Step 8: Checking Ollama logs for GPU status..."
tail -20 /tmp/ollama_gpu.log | grep -E "GPU|gpu|CUDA|cuda|layers|offload" | tail -5

echo ""
echo "========================================================================"
echo "If GPU layers are still 0, try:"
echo "1. Check CUDA installation: nvidia-smi"
echo "2. Reinstall Ollama: curl -fsSL https://ollama.com/install.sh | sh"
echo "3. Check Ollama version: ollama --version"
echo "========================================================================"