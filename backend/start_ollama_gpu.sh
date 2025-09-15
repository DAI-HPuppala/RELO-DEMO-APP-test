#!/bin/bash
# Start Ollama with GPU support - Manual override for CUDA issues

echo "========================================================================"
echo "STARTING OLLAMA WITH GPU SUPPORT"
echo "========================================================================"

# Kill any existing Ollama processes
echo "📍 Stopping any existing Ollama processes..."
pkill -f "ollama serve" 2>/dev/null
sleep 2

# Set comprehensive CUDA environment variables
echo "📍 Setting CUDA environment variables..."
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_DEBUG=1  # Enable debug logging
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_MODELS=/usr/share/ollama/.ollama/models
export OLLAMA_KEEP_ALIVE=600s
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=1

# RTX A1000 specific settings
export OLLAMA_CUDA_COMPUTE_CAP=8.6
export OLLAMA_GPU_MEMORY_FRACTION=0.90  # Use 90% of GPU memory
export OLLAMA_NUM_GPU=999  # Force maximum GPU layers

# CUDA runtime settings
export CUDA_CACHE_DISABLE=0
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_MODULE_LOADING=EAGER  # Force eager loading of CUDA modules

# LD_LIBRARY_PATH for CUDA libraries
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

echo "📍 Environment variables set:"
echo "   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "   OLLAMA_CUDA_COMPUTE_CAP=$OLLAMA_CUDA_COMPUTE_CAP"
echo "   OLLAMA_NUM_GPU=$OLLAMA_NUM_GPU"
echo "   OLLAMA_GPU_MEMORY_FRACTION=$OLLAMA_GPU_MEMORY_FRACTION"

# Check GPU status
echo ""
echo "📍 GPU Status:"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader

# Start Ollama with debug logging
echo ""
echo "📍 Starting Ollama service with GPU support..."
echo "   Logs will be written to: /tmp/ollama_gpu_debug.log"
echo ""

# Start Ollama in foreground with debug output
ollama serve 2>&1 | tee /tmp/ollama_gpu_debug.log &
OLLAMA_PID=$!

echo "   Ollama started with PID: $OLLAMA_PID"
echo ""

# Wait for Ollama to be ready
echo "📍 Waiting for Ollama to initialize..."
sleep 5

# Test if Ollama is responding
for i in {1..10}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "   ✅ Ollama is ready!"
        break
    fi
    echo "   Waiting... ($i/10)"
    sleep 2
done

# Show GPU-related log entries
echo ""
echo "📍 GPU initialization status from logs:"
grep -i -E "cuda|gpu|offload|layers|ggml" /tmp/ollama_gpu_debug.log | tail -10

# Test GPU inference
echo ""
echo "📍 Testing GPU inference with qwen2.5vl:3b..."
time curl -X POST http://localhost:11434/api/generate \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2.5vl:3b",
        "prompt": "Test GPU inference",
        "stream": false,
        "options": {
            "num_gpu": 999,
            "num_thread": 4,
            "num_predict": 10,
            "temperature": 0.1
        }
    }' 2>/dev/null | jq -r '.response' | head -3

# Check if GPU is being used
echo ""
echo "📍 Checking GPU memory usage during inference..."
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

echo ""
echo "========================================================================"
echo "Monitor GPU usage with: watch -n 1 nvidia-smi"
echo "Check logs with: tail -f /tmp/ollama_gpu_debug.log | grep -i gpu"
echo "Stop Ollama with: pkill -f 'ollama serve'"
echo "========================================================================"