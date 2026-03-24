FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /app

# Nur die leichten Abhängigkeiten installieren (torch ist bereits im Base-Image)
RUN pip install --no-cache-dir \
    fastapi>=0.111.0 \
    uvicorn[standard]>=0.29.0 \
    python-multipart>=0.0.9 \
    Pillow>=10.0.0 \
    numpy>=1.24.0 \
    timm>=0.9.0 \
    torchvision>=0.17.0 \
    grad-cam>=1.5.0 \
    huggingface_hub>=0.20.0

COPY . .

ENV HF_REPO_ID=VenthanVi/fracture-detection
ENV MODEL_NAME=tf_efficientnetv2_m
ENV IMG_SIZE=480

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
