# DeepShield — Code Reference

> Deepfake Video Detection with Vision Transformer (ViT)

This document describes every file in the repository, organized by directory.

---

## Tech Stack

| Layer        | Technologies                                                                 |
|--------------|-----------------------------------------------------------------------------|
| **Frontend** | React 18, Create React App, Framer Motion, React Spring, OGL, FontAwesome  |
| **Backend**  | Flask, Flask-CORS                                                           |
| **ML**       | PyTorch, timm (`vit_base_patch16_224`), OpenCV, NumPy                       |
| **Training** | Jupyter Notebooks (Kaggle / Google Colab)                                   |

---

## Root Files

| File | Description |
|------|-------------|
| `package.json` | React project manifest — defines dependencies (React 18, Framer Motion, React Spring, OGL, FontAwesome), scripts (`start`, `build`, `test`, `eject`), and browserslist. |
| `package-lock.json` | Auto-generated lockfile pinning exact dependency versions. |
| `.gitignore` | Ignores `node_modules/`, build artifacts, Python bytecode, trained model weights (`.pth`, `.pt`, `.h5`, `.onnx`), Jupyter checkpoints, and IDE configs. |
| `README.md` | HTML-formatted project overview covering dataset preparation, ViT model architecture, training process, validation metrics, video prediction, installation, results (~89.71% train / ~87.77% val accuracy), and contributor info. |
| `STATUS.md` | Current project status — feature-complete frontend, backend requiring model weights, notes on tests, CI, linting, and recommended next steps. |

---

## `backend/` — Flask API Server

| File | Description |
|------|-------------|
| `app.py` | Main Flask application (147 lines). Loads a ViT model (`vit_base_patch16_224`, 2 classes) from `models/best_vit_model.pth`, exposes two endpoints: **`POST /predict`** — accepts a video upload, samples ~2 frames/sec, runs ViT inference on each frame, and streams real/fake counts + percentages via Server-Sent Events (SSE). **`GET /health`** — returns device info and model-loaded status. Handles `module.` prefix stripping from DataParallel-trained state dicts, ImageNet normalization, and temp file cleanup. |
| `requirements.txt` | Python dependencies: `flask`, `flask-cors`, `torch`, `torchvision`, `timm`, `opencv-python`, `numpy`, `Pillow`, `scikit-learn`, `seaborn`, `matplotlib`. |

### `backend/models/`

| File | Description |
|------|-------------|
| `best_vit_model.pth` | Trained ViT model weights (~327 MB). Listed in `.gitignore` — must be obtained by training via the notebooks or separately. |

---

## `src/` — React Frontend Source

### Core Files

| File | Description |
|------|-------------|
| `index.js` | React entry point — renders `<App />` inside `React.StrictMode` using `createRoot`. Imports `index.css` and calls `reportWebVitals()`. |
| `index.css` | Global CSS — base font, body/code styles. |
| `App.js` | Main application component (378 lines). Contains: **Landing section** with `Particles` background, `DecryptedText` title, `RotatingText` subtitle. **Upload section** with drag-and-drop area, file type validation (video/image), media preview. **Detection section** — POSTs to `http://127.0.0.1:5000/predict`, consumes SSE stream, displays live `LiveBar` chart, and shows final Real/Deepfake verdict. **Workflow diagram** — step-by-step visual of the detection pipeline. **Team section** — contributor `Card` components. Uses scroll-based section visibility, Framer Motion animations, and `useRef`/`useCallback` hooks. |
| `App.css` | Styles for the entire App — layout, upload area, results panel, workflow diagram, team section, responsive design, animations, and dark theme. |
| `App.test.js` | Default CRA test — checks for "learn react" text (not updated for current UI). |
| `Card.js` | Reusable team member card component — displays name, email, phone, GitHub, and LinkedIn links. Uses `SmallIcons` for FontAwesome icons. |
| `Card.css` | Styles for the `Card` component — card layout, hover effects. |
| `logo.svg` | Default React logo SVG (unused in the current UI). |
| `reportWebVitals.js` | CRA performance measurement utility — dynamically imports `web-vitals` and reports CLS, FID, FCP, LCP, TTFB. |
| `setupTests.js` | Jest setup — imports `@testing-library/jest-dom` matchers. |

### `src/fonts/`

| File | Description |
|------|-------------|
| `ASM-Bold.woff` | Custom "ASM Bold" web font (WOFF format). |
| `ASM-Bold.woff2` | Custom "ASM Bold" web font (WOFF2 format, smaller). |

### `src/components/` — Reusable UI Components

| File | Description |
|------|-------------|
| `Particles.js` | WebGL particle system using **OGL** — renders 1000 floating 3D particles with depth-based sizing, cursor-following, bounce physics, and custom vertex/fragment shaders. |
| `Particles.css` | Styles for the particles container (full-size, pointer-events-none). |
| `DecryptedText.js` | Text scramble/decrypt animation component (244 lines) using **Framer Motion**. Reveals text character-by-character with configurable speed, direction (`start`/`end`/`center`), and trigger (`hover`/`view` via IntersectionObserver). Supports custom character sets and original-chars-only mode. |
| `DecryptedText.css` | Optional styling for the decrypted text effect. |
| `RotatingText.js` | Rotating text carousel component (196 lines) using **Framer Motion** `AnimatePresence`. Cycles through an array of strings with spring-based enter/exit transitions, staggered character animation, and configurable split mode (`characters`/`words`/`lines`). Exposes `next`/`previous`/`jumpTo`/`reset` via `useImperativeHandle`. |
| `RotatingText.css` | Styles for the rotating text — layout, SR-only spans, word/element spacing. |
| `GradientText.js` | Animated gradient text component — applies a scrolling `linear-gradient` to text with configurable colors and animation speed. Optional gradient border overlay. |
| `GradientText.css` | Keyframe animation for the gradient scroll effect and text content styling. |
| `SplitText.js` | Letter-by-letter text reveal component (97 lines) using **React Spring**. Splits text into individual characters, animates each with staggered spring transitions triggered on viewport intersection via IntersectionObserver. Fires `onLetterAnimationComplete` callback. |
| `SplitText.css` | Styles for split text parent container. |
| `LiveBar.js` | Horizontal stacked bar chart component — shows real vs. fake percentages as colored bar segments with percentage labels. Used during live inference streaming. |
| `LiveBar.css` | Styles for the live bar — green (real) and red (fake) segments, labels, rounded corners. |
| `ConstellationBackground.js` | Canvas-based constellation animation (91 lines) — renders 150 moving stars connected by proximity lines (< 100px). Runs a continuous `requestAnimationFrame` loop, handles window resize. |
| `SpotLightCard.js` | Interactive card with mouse-following spotlight effect — tracks cursor position via CSS custom properties (`--mouse-x`, `--mouse-y`) and renders a radial gradient highlight. |
| `SpotLightCard.css` | Styles for the spotlight card — background, border, radial gradient based on mouse position. |
| `SmallIcons.js` | Renders a row of FontAwesome icons (email, phone, GitHub, LinkedIn) — used inside team `Card` components. |

---

## `notebooks/` — Model Training

| File | Description |
|------|-------------|
| `DD_ViT_Final.ipynb` | Training notebook (Kaggle environment) — end-to-end ViT deepfake detection pipeline: dataset loading from `/kaggle/input/`, frame extraction, data augmentation, training loop with `CrossEntropyLoss` and `AdamW`, validation, and model export. |
| `deepfake_detection_vit_best.ipynb` | Extended training notebook (Google Colab) — larger dataset, more comprehensive experimentation, confusion matrix visualization, classification reports, and model checkpointing. |

---

## `public/` — Static Assets

| File | Description |
|------|-------------|
| `index.html` | HTML shell for the React SPA — contains the `<div id="root">` mount point, meta tags, and `<title>`. |
| `manifest.json` | PWA manifest — app name "deepshield", theme color, icons configuration. |
| `arch.png` | Model architecture diagram image used in documentation. |
| `dp.jpg` | Default profile picture used for team member cards. |

---

## `docs/` — Documentation Assets

| File | Description |
|------|-------------|
| `Img1.png` | Screenshot — website landing page. |
| `Img2.png` | Screenshot — upload interface. |
| `Img3.png` | Screenshot — processing results page. |
| `Img4.png` | Model architecture diagram (same as `public/arch.png`). |
| `Readme.txt` | Empty placeholder file. |
