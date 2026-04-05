---
title: Fracture Detection API
emoji: 🦴
colorFrom: blue
colorTo: red
sdk: docker
port: 7860
pinned: false
---

# Forearm Fracture Detection

AI-gestützte Frakturerkennung aus Unterarm-Röntgenbildern — konzipiert für den radiologischen Einsatz.

**Modell:** EfficientNetV2-M · **Test-AUC:** 0.884 · **Sensitivität:** 74.6% · **Spezifität:** 85.1%

---

## Inhalt

- [Projektübersicht](#projektübersicht)
- [Ergebnisse](#ergebnisse)
- [Projektstruktur](#projektstruktur)
- [Schnellstart](#schnellstart)
- [Training](#training)
- [API-Referenz](#api-referenz)
- [Konfiguration](#konfiguration)

---

## Projektübersicht

Dieses Projekt implementiert einen binären Klassifikator (Fraktur / Normal) für Unterarm-Röntgenbilder. Die Pipeline umfasst:

- **Zwei-Phasen-Training** — Phase 1: Backbone eingefroren, nur Head trainieren · Phase 2: Letzte Backbone-Blöcke fine-tunen
- **Grad-CAM++ Heatmaps** — visuelle Erklärung der Modellentscheidung für Radiologen
- **FastAPI Backend** — REST-API mit Einzel- und Batch-Inferenz
- **React Frontend** — Side-by-side Bildvergleich mit Opacity-Slider und Konfidenz-Warnung

---

## Ergebnisse

Evaluiert auf dem gehaltenen Test-Set (495 Bilder, nie während des Trainings gesehen):

| Metrik | Wert |
|---|---|
| AUC-ROC | **0.884** |
| Sensitivität (Fraktur-Recall) | 0.746 |
| Spezifität | 0.851 |
| Accuracy | 0.800 |
| PPV (Precision) | 0.825 |
| NPV | 0.781 |
| Optimaler Threshold (Youden's J) | 0.613 |

**Datensplit:** 3 831 Train · 480 Val · 495 Test (80/10/10, patienten-stratifiziert)

### Modell-Vergleich (Val-AUC)

| Modell | Params | Val AUC | Trainingszeit |
|---|---|---|---|
| tf_efficientnetv2_m *(gewählt)* | — | — | Kaggle GPU |
| convnext_tiny | 28M | 0.788 | 799s |
| densenet121 | 7.2M | 0.775 | 869s |
| efficientnet_b2 | 8.1M | 0.762 | 754s |
| efficientnet_b0 | 4.3M | 0.740 | 690s |

---

## Projektstruktur

```
FractureDataset/
├── ml/                        # Python-Package: alle ML-Komponenten
│   ├── config.py              # Hyperparameter & Pfade (env-var-overridable)
│   ├── model.py               # EfficientNet-Wrapper (timm) + Freeze-Utilities
│   ├── dataloader.py          # Dataset + Augmentation-Pipelines
│   ├── dataset.py             # splits.csv generieren (einmalig ausführen)
│   ├── train.py               # Zwei-Phasen-Training-Loop
│   ├── evaluate.py            # Test-Set-Evaluation + Plots
│   ├── tune.py                # Optuna Hyperparameter-Suche
│   └── compare_models.py      # Architektur-Vergleich
│
├── api/                       # FastAPI Backend
│   ├── main.py                # Endpoints: /health /predict /predict/batch /model/stats
│   ├── predictor.py           # Modell-Singleton + Grad-CAM++ (COLORMAP_INFERNO)
│   ├── schemas.py             # Pydantic Response-Modelle
│   └── requirements.txt       # API-spezifische Dependencies
│
├── frontend/                  # React + Vite Frontend
│   └── src/
│       ├── pages/             # SingleUpload, BatchUpload, ModelStats
│       ├── components/        # HeatmapViewer, PredictionCard, ThresholdStatusBar, ...
│       ├── hooks/             # useOptimalThreshold
│       └── utils/             # isUncertain(), getErrorMessage(), DEFAULT_THRESHOLD
│
├── notebooks/                 # Jupyter Notebooks
│   ├── kaggle_train.ipynb     # Kompletter Kaggle-Trainings-Workflow
│   ├── thesis_analysis.ipynb
│   └── eda.ipynb
│
├── checkpoints/               # Modell-Gewichte (nicht in git)
│   ├── best_model.pth
│   └── best_model_config.json # Architektur-Metadaten zum Checkpoint
│
├── data/
│   └── splits.csv             # Train/Val/Test-Splits (in git)
│
├── logs/                      # Evaluations-Ergebnisse
│   ├── eval_results.json      # Test-Metriken (in git, Frontend liest sie)
│   ├── model_comparison.csv
│   └── *.png                  # ROC-Kurve, Konfusionsmatrix (nicht in git)
│
├── Dockerfile
├── render.yaml
└── requirements.txt
```

---

## Schnellstart

### Voraussetzungen

- Python 3.11+
- Node.js 18+
- Checkpoint-Datei: `checkpoints/best_model.pth` (von Kaggle oder Hugging Face laden)

### Backend starten

```bash
cd api
uvicorn main:app --reload --port 8000
```

→ API erreichbar unter `http://localhost:8000`  
→ Swagger-Doku unter `http://localhost:8000/docs`

### Frontend starten

```bash
cd frontend
npm install
npm run dev
```

→ App erreichbar unter `http://localhost:5173`

### Checkpoint laden (falls nicht vorhanden)

```bash
# Von Hugging Face (automatisch beim ersten API-Start falls HF_REPO_ID gesetzt):
export HF_REPO_ID=VenthanVi/fracture-detection
cd api && uvicorn main:app --port 8000
```

---

## Training

### Lokal (MacBook M4, Debug)

```bash
python -m ml.train
```

Verwendet `efficientnet_b0` mit `IMG_SIZE=224`, ~50–80s pro Epoch.

### Kaggle (GPU T4 x2, Produktionstraining)

1. Notebook öffnen: `notebooks/kaggle_train.ipynb`
2. Accelerator: **GPU T4 x2** · Internet: **On**
3. `HF_TOKEN` und `IMAGE_DATASET_SLUG` in Cell 2 eintragen
4. Alle Cells ausführen (~2–5h)
5. `best_model.pth` aus dem Output-Tab herunterladen → `checkpoints/`

#### Umgebungsvariablen für Cloud-Training

| Variable | Lokal | Kaggle |
|---|---|---|
| `MODEL_NAME` | `efficientnet_b0` | `tf_efficientnetv2_m` |
| `IMG_SIZE` | `224` | `480` |
| `BATCH_SIZE` | `32` | `16` |
| `NUM_WORKERS` | `4` | `8` |
| `DATA_ROOT` | lokaler Pfad | `/kaggle/input/.../alle Bilder` |

### Evaluation

```bash
python -m ml.evaluate --save-json
# → schreibt logs/eval_results.json (wird vom Frontend angezeigt)
```

### Hyperparameter-Suche

```bash
python -m ml.tune --trials 20
# → Ergebnisse in logs/best_params.json
```

---

## API-Referenz

### `GET /health`

```json
{ "status": "ok", "model": "tf_efficientnetv2_m" }
```

### `POST /predict`

Einzelbild-Inferenz mit Grad-CAM.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@bild.png" \
  -F "threshold=0.613"
```

```json
{
  "filename": "bild.png",
  "label": "fracture",
  "probability": 0.8734,
  "threshold_used": 0.613,
  "gradcam_image": "data:image/png;base64,..."
}
```

### `POST /predict/batch`

Mehrere Bilder in einem Request. Führt einen einzigen Batch-Forward-Pass durch.

```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "files=@bild1.png" \
  -F "files=@bild2.png" \
  -F "threshold=0.613"
```

### `GET /model/stats`

Gibt den Inhalt von `logs/eval_results.json` zurück (gecacht in-memory).

---

## Konfiguration

Alle Hyperparameter sind in `ml/config.py` definiert und per Umgebungsvariable überschreibbar:

| Konstante | Standard | Env-Var |
|---|---|---|
| `MODEL_NAME` | `efficientnet_b0` | `MODEL_NAME` |
| `IMG_SIZE` | `224` | `IMG_SIZE` |
| `BATCH_SIZE` | `32` | `BATCH_SIZE` |
| `HEAD_DROPOUT` | `0.4` | `HEAD_DROPOUT` |
| `DEFAULT_THRESHOLD` | `0.5` | — |
| `UNCERTAINTY_MARGIN` | `0.05` | — |
| `GRADCAM_IMAGE_WEIGHT` | `0.4` | — |
| `PHASE1_LR` | `1e-3` | — |
| `PHASE2_LR` | `1e-5` | — |

### Checkpoint-Format

Ab dieser Version speichert `train.py` Checkpoints mit Architektur-Metadaten:

```python
{
  "state_dict": { ... },
  "model_name": "tf_efficientnetv2_m",
  "img_size": 480
}
```

Ältere plain-state-dict Checkpoints werden weiterhin geladen (Sidecar `best_model_config.json`).
