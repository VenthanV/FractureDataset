# Fracture Detection — Projektstatus

> Zuletzt aktualisiert: 2026-02-25
> Lies diese Datei am Anfang einer neuen Session, um direkt weitermachen zu können.

---

## Modell-Status

| | Lokal (M4 MacBook) | Kaggle (Ziel) |
|---|---|---|
| Backbone | `efficientnet_b0` | `tf_efficientnetv2_m` |
| IMG_SIZE | 224 | 480 |
| Trainiert? | Ja — 11 Epochen Phase 2 | Nein — noch nicht gestartet |
| Bestes val AUC | **0.8187** | — |
| Checkpoint | `checkpoints/best_model.pth` | `/kaggle/working/best_model.pth` |
| `eval_results.json` | **fehlt** (muss erst generiert werden) | — |

Das lokale Modell (`efficientnet_b0`) läuft, ist aber nur das kleine Debug-Modell.
Das echte Modell für die Web-App ist `tf_efficientnetv2_m` — muss auf Kaggle trainiert werden.

---

## Was bereits fertig ist

### Backend (`api/`)
- `api/main.py` — FastAPI: `GET /health`, `POST /predict`, `POST /predict/batch`, `GET /model/stats`
- `api/predictor.py` — Model-Singleton, Grad-CAM++ Heatmap (base64 PNG)
- `api/schemas.py` — Pydantic-Modelle
- `api/requirements.txt` — `fastapi`, `uvicorn`, `python-multipart`

### Frontend (`frontend/`)
- React + Vite, 3 Seiten: Einzelbild (`/`), Batch (`/batch`), Metriken (`/stats`)
- Optimaler Schwellenwert wird automatisch aus `/model/stats` geladen (Standard: Youden's J)
- Schwellenwert-Slider in Upload-Pages: **eingeklappt** hinter `⚙ Debug`
- `ModelStats`-Page: interaktiver Schwellenwert-Explorer mit Live-Metriken + SVG-Kurve
  (Sensitivität/Spezifität live berechnet aus `test_probs`/`test_labels` im JSON)

### `evaluate.py`
- `--save-json` Flag: schreibt `logs/eval_results.json` mit allen Metriken
- Speichert auch `test_probs` und `test_labels` → Frontend kann Metriken live berechnen

### Kaggle-Fix (`config.py` + `train.py`)
- Erkennt Kaggle automatisch (`/kaggle/working` existiert)
- Schreibt Checkpoints nach `/kaggle/working/checkpoints/` (nicht read-only Input-Dir)
- Kopiert `best_model.pth` nach `/kaggle/working/` → erscheint im Output-Tab

---

## Nächste Aufgabe: Kaggle Notebook für `tf_efficientnetv2_m`

### Ziel
`tf_efficientnetv2_m` mit IMG_SIZE=480 auf GPU (T4/P100) trainieren,
`best_model.pth` runterladen, lokal in `checkpoints/` ablegen, Web-App starten.

### Kaggle Notebook Setup

**1. Neues Notebook erstellen**
- kaggle.com → Code → New Notebook
- Accelerator: **GPU T4 x2** (oder P100)
- Internet: **On** (für timm-Download)

**2. Dataset hinzufügen**
- Add Data → dein Röntgenbild-Dataset (forearm fracture xrays)
- Add Data → dieses Repo als Dataset (oder Code via Git)

**3. Notebook-Zellen**

```python
# Zelle 1 — Abhängigkeiten
!pip install timm grad-cam --quiet
```

```python
# Zelle 2 — Repo klonen (oder Dataset-Pfad setzen)
import subprocess
# Option A: aus Kaggle-Dataset (wenn du den Code hochgeladen hast)
import sys
sys.path.insert(0, '/kaggle/input/DEIN-CODE-DATASET/')

# Option B: direkt aus GitHub
# !git clone https://github.com/DEIN-REPO /kaggle/working/fracture
# sys.path.insert(0, '/kaggle/working/fracture')
```

```python
# Zelle 3 — Umgebungsvariablen setzen
import os
os.environ['MODEL_NAME']    = 'tf_efficientnetv2_m'
os.environ['IMG_SIZE']      = '480'
os.environ['BATCH_SIZE']    = '16'
os.environ['NUM_WORKERS']   = '4'
os.environ['DATA_ROOT']     = '/kaggle/input/DEIN-DATASET/alle Bilder'
# CHECKPOINT_DIR und LOG_DIR werden automatisch nach /kaggle/working/ gesetzt
```

```python
# Zelle 4 — Splits generieren (nur beim ersten Mal)
# import dataset  # falls du ein dataset.py hast
# dataset.main()
```

```python
# Zelle 5 — Training starten
from train import train_model
best_auc = train_model()
print(f"Bestes val AUC: {best_auc:.4f}")
```

```python
# Zelle 6 — Checkpoint downloadbar machen
from IPython.display import FileLink
FileLink('best_model.pth')   # erscheint als Klick-Link
```

```python
# Zelle 7 — Evaluation + JSON für Web-App
import subprocess
result = subprocess.run(
    ['python', 'evaluate.py', '--save-json'],
    capture_output=True, text=True
)
print(result.stdout)
# Dann auch eval_results.json runterladen:
FileLink('logs/eval_results.json')
```

### Nach dem Training — Web-App starten

```bash
# 1. Heruntergeladene Dateien ablegen
cp ~/Downloads/best_model.pth FractureDataset/checkpoints/best_model.pth
cp ~/Downloads/eval_results.json FractureDataset/logs/eval_results.json

# 2. Backend starten (aus FractureDataset/)
cd api && uvicorn main:app --reload --port 8000

# 3. Frontend starten (neues Terminal)
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

---

## Offene Punkte / Todo

- [ ] `tf_efficientnetv2_m` auf Kaggle trainieren
- [ ] `best_model.pth` + `eval_results.json` lokal ablegen
- [ ] Web-App end-to-end testen (Bild hochladen → Prediction + Heatmap)
- [ ] Grad-CAM testen — target layer: `model.backbone.blocks[-1]`
- [ ] (Optional) `tune.py` auf Kaggle für Hyperparameter-Optimierung

---

## Start-Befehle Übersicht

```bash
# Lokales Training (debug, efficientnet_b0)
python train.py

# Evaluation + JSON generieren (nach Training)
python evaluate.py --save-json

# Backend
cd api && uvicorn main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

---

## Wichtige Pfade

| Was | Pfad |
|---|---|
| Checkpoint (lokal) | `checkpoints/best_model.pth` |
| Metriken-JSON | `logs/eval_results.json` |
| Training-Log | `logs/train_log.csv` |
| API | `api/main.py` |
| Predictor + Grad-CAM | `api/predictor.py` |
| Frontend Entry | `frontend/src/App.jsx` |
| Config (alle Pfade/HPs) | `config.py` |
