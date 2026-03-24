FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch (kein CUDA nötig, spart ~2GB)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet --no-cache-dir

# API-Abhängigkeiten
COPY api/requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Gesamten Code kopieren
COPY . .

# Umgebungsvariablen
ENV HF_REPO_ID=VenthanVi/fracture-detection
ENV MODEL_NAME=tf_efficientnetv2_m
ENV IMG_SIZE=480

# HF Spaces erwartet Port 7860
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
