# Dockerfile for GPU-enabled defect detection
# syntax=docker/dockerfile:1.4

# Stage 1: pre-download model weights into cache
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && rm -rf /var/lib/apt/lists/*
# Install Python deps
COPY requirements.txt /app/
RUN python -m pip install --no-cache-dir -r requirements.txt
# Pre-download Moondream2 revision 2025-04-14 into builder image
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('vikhyatk/moondream2', revision='2025-04-14', cache_dir='/root/.cache/huggingface')"

## Stage 2: final image with baked weights
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
WORKDIR /app
# System deps
RUN apt-get update && apt-get install -y python3 python3-pip python3-numpy python3-dev build-essential swig && \
    ln -s /usr/bin/python3 /usr/bin/python && rm -rf /var/lib/apt/lists/*
# Install Python deps
COPY requirements.txt /app/
RUN python -m pip install --no-cache-dir -r requirements.txt
# Copy cached HF weights from builder
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
# Copy all .deb packages (runtime SDK + Python wrapper + Docker driver) into the image
COPY *.deb /app/
## Install the .deb packages (apt-get install), then cleanup
RUN apt-get update && apt-get install -y python3-numpy && \
    apt-get install -y /app/eBUS_SDK_JAI_Ubuntu-22.04-x86_64-*.deb /app/eBUS_Python_JAI_Ubuntu-22.04-x86_64-*.deb && \
    rm -f /app/*.deb
RUN ln -sf /opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib/_ebus_python.so /usr/lib/python3/dist-packages/eBUS.so
ENV PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH

## Set up eBUS SDK environment
ENV EBUS_VSDK_ROOT=/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64
ENV LD_LIBRARY_PATH=$EBUS_VSDK_ROOT/lib:$LD_LIBRARY_PATH
ENV GENICAM_ROOT_V3_4=$EBUS_VSDK_ROOT/lib/genicam
ENV PYTHONPATH=$EBUS_VSDK_ROOT/lib:$PYTHONPATH

## Install eBUS SDK libraries & Python bindings per Quick Start Guide
RUN chmod +x $EBUS_VSDK_ROOT/bin/install_libraries.sh && \
    bash $EBUS_VSDK_ROOT/bin/install_libraries.sh

## Debug eBUS SDK installation
RUN echo "=== eBUS SDK Installation ===" && \
    ls -R $EBUS_VSDK_ROOT && \
    echo "=== Python sys.path ===" && python -c "import sys; print(sys.path)" && \
    echo "=== /usr/local/lib/python3.10/dist-packages ===" && ls -R /usr/local/lib/python3.10/dist-packages && \
    echo "=== /usr/lib/python3.10/dist-packages ===" && ls -R /usr/lib/python3.10/dist-packages && \
    echo "=== Attempt import eBUS ===" && python -c "import eBUS; print('eBUS imported successfully')" || true

# Add Python path for eBUS SDK
ENV PYTHONPATH=$EBUS_VSDK_ROOT/lib:$PYTHONPATH

# Copy application code and capture utility
COPY main.py simple_capture.py /app/

# Set up environment for eBUS Docker Driver
ENV EBUS_DOCKER=1

# Expose port and start server
EXPOSE 80
CMD ["python","-m","uvicorn","main:app","--host","0.0.0.0","--port","80"]
