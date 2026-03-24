FROM python:3.11-slim

WORKDIR /app

# Systemlibs für OpenCV (grad-cam dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (kein conda, kein Konflikt)
RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Übrige Abhängigkeiten
RUN pip install --no-cache-dir \
    fastapi>=0.111.0 \
    uvicorn[standard]>=0.29.0 \
    python-multipart>=0.0.9 \
    Pillow>=10.0.0 \
    "numpy>=1.24,<2" \
    timm>=0.9.0 \
    opencv-python-headless>=4.8.0 \
    grad-cam>=1.5.0 \
    huggingface_hub>=0.20.0

COPY . .

ENV HF_REPO_ID=VenthanVi/fracture-detection
ENV MODEL_NAME=tf_efficientnetv2_m
ENV IMG_SIZE=480

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
