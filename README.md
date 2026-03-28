# AI-Powered Multi-Modal Fake Content Detector

> Detect deepfakes in **images**, **videos**, and **audio** with explainable AI — complete with a React dashboard, FastAPI backend, and a Chrome extension.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖼️ **Image Detection** | EfficientNet-B0 CNN classifies images as real or fake |
| 🎬 **Video Detection** | ResNet-18 frame encoder + LSTM for temporal analysis |
| 🎵 **Audio Detection** | Mel-spectrogram CNN distinguishes authentic vs synthesised speech |
| 🔍 **Explainable AI** | Grad-CAM heatmaps highlight *why* content is flagged as fake |
| 🌐 **Web Dashboard** | React + Tailwind dark-mode UI for upload and result history |
| 🧩 **Chrome Extension** | Scans any webpage, marks fake images with an overlay badge |
| 📊 **Result History** | MongoDB stores every prediction for later review |
| ⚡ **Async FastAPI** | High-performance async backend with auto-generated OpenAPI docs |

---

## 🏗️ Project Structure

```
Deepdetect-AI/
├── backend/
│   ├── main.py                 # FastAPI app (startup, CORS, MongoDB)
│   ├── routes.py               # POST /predict/image|video|audio, GET /results
│   ├── models/
│   │   ├── image_model.py      # EfficientNet-B0 image deepfake detector
│   │   ├── video_model.py      # ResNet-18 + LSTM video deepfake detector
│   │   └── audio_model.py      # Mel-spectrogram CNN audio deepfake detector
│   ├── utils/
│   │   ├── preprocessing.py    # Validation, resizing, frame extraction
│   │   └── explainability.py   # Grad-CAM for all three modalities
│   └── requirements.txt
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.jsx             # Root component + navbar
│   │   ├── api.js              # Axios helper for all API calls
│   │   ├── index.js
│   │   ├── index.css           # Tailwind base
│   │   ├── components/
│   │   │   ├── FileUploader.jsx   # Drag-and-drop upload
│   │   │   ├── ResultCard.jsx     # Prediction result display
│   │   │   ├── HeatmapViewer.jsx  # Grad-CAM overlay with lightbox
│   │   │   └── ProgressBar.jsx    # Animated progress indicator
│   │   └── pages/
│   │       ├── Home.jsx           # Main detector page
│   │       └── Dashboard.jsx      # Result history dashboard
│   ├── package.json
│   ├── tailwind.config.js
│   └── postcss.config.js
├── extension/
│   ├── manifest.json           # Chrome Manifest v3
│   ├── content.js              # Page scanner + fake-image badge overlay
│   ├── background.js           # Service worker for API calls
│   ├── popup.html              # Extension popup UI
│   └── popup.js                # Popup logic
├── training/
│   ├── data_pipeline.py        # Dataset classes + dataloader utilities
│   ├── train_image.py          # Image model training script
│   ├── train_video.py          # Video model training script
│   └── train_audio.py          # Audio model training script
├── .env.example
└── README.md
```

---

## 🔧 Tech Stack

**Backend:** Python 3.11, FastAPI, PyTorch, TorchVision, OpenCV, librosa, Motor (async MongoDB)

**Frontend:** React 18, Tailwind CSS 3, Axios, React Router v6, React Dropzone

**Database:** MongoDB (optional — results stored if `MONGO_URI` is set)

**Extension:** Chrome Manifest v3, Vanilla JS

---

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.10
- Node.js ≥ 18
- (Optional) MongoDB instance

---

### 1 — Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) configure environment
cp ../.env.example .env
# Edit .env: set MONGO_URI, model paths, etc.

# Start the API server (run from repo root)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at **http://localhost:8000/docs**

---

### 2 — Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

Opens **http://localhost:3000**

---

### 3 — Chrome Extension

1. Open **chrome://extensions**
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select the `extension/` folder

The Deepdetect AI icon will appear in your toolbar.

---

### 4 — Training (optional)

Prepare your data in the expected layout:

```
data/
  images/
    real/   *.jpg
    fake/   *.jpg
  videos/
    real/   *.mp4
    fake/   *.mp4
  audio/
    real/   *.wav
    fake/   *.wav
```

Then train each modality:

```bash
# Image detector
python -m training.train_image --data_dir data/images --epochs 30 --output_path models/image_model.pt

# Video detector
python -m training.train_video --data_dir data/videos --epochs 20 --output_path models/video_model.pt

# Audio detector
python -m training.train_audio --data_dir data/audio  --epochs 30 --output_path models/audio_model.pt
```

Point the backend to your trained weights via environment variables:

```bash
export IMAGE_MODEL_PATH=models/image_model.pt
export VIDEO_MODEL_PATH=models/video_model.pt
export AUDIO_MODEL_PATH=models/audio_model.pt
```

---

## ⚡ API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/predict/image` | Analyse uploaded image |
| `POST` | `/predict/video` | Analyse uploaded video |
| `POST` | `/predict/audio` | Analyse uploaded audio |
| `GET` | `/results` | List recent predictions |

### Example response

```json
{
  "prediction": "fake",
  "confidence": 0.92,
  "raw_score": 0.92,
  "heatmap_url": "data:image/png;base64,...",
  "filename": "sample.jpg",
  "timestamp": "2024-07-01T12:00:00+00:00"
}
```

---

## 🔍 Explainable AI

Grad-CAM is computed automatically for every prediction:

- **Images** — heatmap overlay on the original image highlights manipulated regions
- **Videos** — Grad-CAM applied to the most representative frame
- **Audio** — saliency map over the mel-spectrogram shows which frequencies were manipulated

The `heatmap_url` field in every response is a base64-encoded PNG (data URI) that can be displayed directly in an `<img>` tag.

---

## 🧩 Chrome Extension Usage

1. Navigate to any webpage containing images.
2. Click the **Deepdetect AI** toolbar icon.
3. The popup lists detected images — click **Analyse** next to any image.
4. Fake images are flagged with a red **⚠ FAKE xx%** badge overlaid on the image.
5. Click **Clear Marks** to remove all overlays.

---

## 🌍 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_ORIGINS` | `*` | Comma-separated list of allowed CORS origins |
| `MONGO_URI` | *(unset)* | MongoDB connection string |
| `MONGO_DB` | `deepdetect` | MongoDB database name |
| `IMAGE_MODEL_PATH` | *(unset)* | Path to trained image model `.pt` file |
| `VIDEO_MODEL_PATH` | *(unset)* | Path to trained video model `.pt` file |
| `AUDIO_MODEL_PATH` | *(unset)* | Path to trained audio model `.pt` file |

---

## 📈 Model Architecture Summary

### Image Model (EfficientNet-B0)
- Input: 224×224 RGB image
- Backbone: EfficientNet-B0 (ImageNet pre-trained)
- Head: Dropout → Linear(1280→256) → ReLU → Dropout → Linear(256→1)
- Output: Binary logit (sigmoid → fake probability)

### Video Model (ResNet-18 + LSTM)
- Input: 16 evenly-sampled frames (224×224)
- Frame encoder: ResNet-18 (without FC) → Linear(512→512)
- Temporal: LSTM (512→256, 2 layers)
- Head: Linear(256→128) → ReLU → Dropout → Linear(128→1)

### Audio Model (Mel-spectrogram CNN)
- Input: 5-second audio clip → 128×128 mel-spectrogram
- Architecture: 4 × (Conv2d → BatchNorm → ReLU → MaxPool/AvgPool)
- Head: Flatten → Linear(4096→512) → ReLU → Dropout → Linear(128→1)

---

## 🛣️ Future Improvements

- [ ] Fine-tune models on public deepfake datasets (FaceForensics++, ASVspoof)
- [ ] Add face-detection pre-processing (MTCNN) for more targeted analysis
- [ ] Real-time webcam detection with MediaPipe
- [ ] Batch processing endpoint (`POST /predict/batch`)
- [ ] User authentication and per-user result history
- [ ] Model versioning and A/B testing
- [ ] Docker Compose setup for one-command deployment
- [ ] Firefox extension support

---

## 📄 License

MIT © 2024 Deepdetect AI Contributors