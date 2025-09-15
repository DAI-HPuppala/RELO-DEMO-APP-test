#!/bin/bash
# Fix CUDA device nodes and permissions for Ollama

echo "========================================================================"
echo "FIXING CUDA DEVICE NODES AND PERMISSIONS"
echo "========================================================================"

# Check current driver version
echo "📍 Current NVIDIA Driver Version:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# Load NVIDIA kernel modules
echo ""
echo "📍 Loading NVIDIA kernel modules..."
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_modeset
sudo modprobe nvidia_drm

# Create NVIDIA device nodes if they don't exist
echo ""
echo "📍 Creating NVIDIA device nodes..."

# Check if /dev/nvidia* exists
if [ ! -e /dev/nvidia0 ]; then
    echo "   Creating /dev/nvidia0..."
    sudo mknod -m 666 /dev/nvidia0 c 195 0
fi

if [ ! -e /dev/nvidiactl ]; then
    echo "   Creating /dev/nvidiactl..."
    sudo mknod -m 666 /dev/nvidiactl c 195 255
fi

if [ ! -e /dev/nvidia-modeset ]; then
    echo "   Creating /dev/nvidia-modeset..."
    sudo mknod -m 666 /dev/nvidia-modeset c 195 254
fi

# Create nvidia-uvm device
if [ ! -e /dev/nvidia-uvm ]; then
    echo "   Creating /dev/nvidia-uvm..."
    # Get the major number for nvidia-uvm
    D=`grep nvidia-uvm /proc/devices | awk '{print $1}'`
    if [ -n "$D" ]; then
        sudo mknod -m 666 /dev/nvidia-uvm c $D 0
    fi
fi

# Set permissions
echo ""
echo "📍 Setting device permissions..."
sudo chmod 666 /dev/nvidia* 2>/dev/null

# Add ollama user to video group
echo ""
echo "📍 Adding ollama user to video group..."
sudo usermod -a -G video ollama 2>/dev/null || echo "   Ollama user may not exist or already in group"

# List devices
echo ""
echo "📍 NVIDIA devices:"
ls -la /dev/nvidia* 2>/dev/null || echo "   No NVIDIA devices found"

# Check if CUDA runtime is accessible
echo ""
echo "📍 Checking CUDA runtime..."
if [ -f /usr/local/lib/ollama/libcudart.so.12.8.90 ]; then
    echo "   CUDA runtime found at /usr/local/lib/ollama/"
    ldd /usr/local/lib/ollama/libcudart.so.12.8.90 2>/dev/null | head -5
else
    echo "   CUDA runtime not found in Ollama directory"
fi

# Restart Ollama with fixed permissions
echo ""
echo "📍 Restarting Ollama service..."
sudo systemctl restart ollama

# Wait for service to start
sleep 5

# Check Ollama logs
echo ""
echo "📍 Checking Ollama GPU initialization..."
sudo journalctl -u ollama -n 20 --no-pager | grep -E "GPU|gpu|CUDA|cuda|offload" | tail -10

echo ""
echo "========================================================================"
echo "Device nodes created and permissions set."
echo "If still having issues, you may need to:"
echo "1. Reboot the system"
echo "2. Install CUDA toolkit: sudo apt install nvidia-cuda-toolkit"
echo "3. Downgrade NVIDIA driver from 550.x to 535.x"
echo "========================================================================"