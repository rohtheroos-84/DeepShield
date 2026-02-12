# DeepShield Repository Status

## Project Overview

**DeepShield** is a web app for deepfake video detection that uses a Vision Transformer (ViT) model to classify frames as real or manipulated. Users upload videos or images, and the system streams prediction results in real time.

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | React 18, Create React App, Framer Motion, React Spring, OGL, FontAwesome |
| **Backend** | Flask, Flask-CORS |
| **ML / Inference** | PyTorch, timm (ViT base patch16 224), OpenCV, NumPy |
| **Training** | Jupyter (Kaggle / Google Colab) |

## Directory Overview

- `backend/` – Flask API providing `/predict` (SSE) and `/health`; loads ViT model and processes video frames.
- `src/` – React app: landing page, upload UI, live results, workflow diagram, team section.
- `src/components/` – Reusable UI components (Particles, DecryptedText, RotatingText, LiveBar, ConstellationBackground, etc.).
- `notebooks/` – Training notebooks: `DD_ViT_Final.ipynb` (Kaggle), `deepfake_detection_vit_best.ipynb` (Colab).
- `docs/` – Screenshots and a README-like doc (HTML snippets).
- `public/` – Static assets such as `arch.png`, `manifest.json`, and `index.html`.

## Main Features & Flow

1. **Frontend**
   - Upload area (drag-and-drop), media preview, “Detect Deepfake” button.
   - Live bar chart with real vs deepfake probabilities.
   - Final verdict (Real / Deepfake) and explanatory text.
2. **Backend**
   - Accepts video uploads.
   - Samples approximately two frames per second and runs ViT inference.
   - Streams real/fake counts via Server-Sent Events from `/predict`.
3. **Model**
   - ViT base patch16 224 (ImageNet pretrained), 2 classes (Real, Manipulated).
   - Weights are expected at `backend/models/best_vit_model.pth` (not committed; `.gitignore` excludes `.pth` files).

## Current Status

| Area | Status |
|------|--------|
| **Frontend** | Feature-complete UI; hardcoded backend URL `http://127.0.0.1:5000`. |
| **Backend** | Implemented; requires `backend/models/best_vit_model.pth` to run inference. |
| **Tests** | `App.test.js` exists but checks the default CRA “learn react” text, not the current UI. |
| **CI** | No GitHub Actions or other CI configuration found. |
| **Linting / Formatting** | No ESLint/Prettier configuration present in the repo root. |
| **Docs** | `README.md` is HTML-like; `docs/Readme.txt` is empty. |
| **Notebooks** | Training notebooks are present and configured for Kaggle/Colab with external datasets. |

**Important**: The backend expects the model weights at `backend/models/best_vit_model.pth`. You must train the model (via the notebooks) or otherwise obtain these weights and place them there before running inference.

## Recommended Next Steps for Contributors

1. **Run the app end-to-end**
   - Train or obtain `best_vit_model.pth`, place it in `backend/models/`.
   - Start the backend (`python backend/app.py`) and frontend (`npm start` from the React app directory).
   - Verify upload, inference, and streaming predictions work as expected.
2. **Stabilize tests and infrastructure**
   - Update `App.test.js` to reflect the current UI and flows.
   - Add basic backend tests for `/health` and `/predict`.
   - Optionally introduce ESLint/Prettier and simple CI (e.g., GitHub Actions) to run lint and tests on pushes/PRs.

