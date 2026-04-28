
# DeepShield: Deepfake Video Detection (ViT Baseline + CNN/ViT/FFT/BiLSTM Hybrid)

This project includes two model tracks: a deployment-ready ViT frame classifier used by the Flask backend, and a newer video-level hybrid notebook that adds CNN + ViT spatial features, FFT frequency cues, and temporal LSTM modeling.

**Live demo:** https://deepshieldetector.netlify.app

## Table of Contents
- Dataset Preparation
- Model Architecture
- Latest Model Updates
- Training Process
- Evaluation Metrics
- Video Prediction
- Installation and Setup
- Results
- Website Usage
- Contributors

## Dataset Preparation
### Current Working Dataset Layout
- **Frame dataset used by training:** `data/FF_frames/fake` and `data/FF_frames/real`
- **Source video collections (optional/reference):** `data/FF++` style real/manipulated folders

### Frame Extraction
Extract frames at approximately 1 FPS. Frame filenames should preserve video identity and frame order for sequence modeling, e.g. `video123_00045.jpg`.

## Model Architecture
![Model Architecture Diagram](docs/arch.png)

### Model Variants
- **Backend inference model (current production path):** ViT (`vit_base_patch16_224`), frame-level inference
- **Latest notebook model:** ResNet50 + ViT + FFT branch + BiLSTM (video-level sequence classification)
- **Input:** 224x224 frame tensors grouped into temporal windows (default `SEQ_LEN = 8`)
- **Classes:** 2 (folder-sorted class order, typically `fake`, `real`)
- **Pretrained weights:** ImageNet for spatial backbones

### Baseline ViT Initialization (backend-compatible)
```python
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)
model.to(device)
model = nn.DataParallel(model)
```

### Latest Hybrid Initialization (notebook)
```python
model = TemporalHybridModel(
    num_classes=2,
    fft_dim=128,
    lstm_hidden=192,
    freeze_backbones=True,
    temporal_pool="mean",
).to(device)
```

## Latest Model Updates
- **Video-level split:** Train/validation split uses video identity to reduce leakage.
- **Sequence sampling:** Sliding windows over ordered frames (`SEQ_LEN`, `SEQ_STRIDE`).
- **Frequency branch:** FFT magnitude features fused with spatial features.
- **Temporal modeling:** BiLSTM over per-frame fused embeddings with configurable pooling.
- **Stability controls:** Class-weighted loss, label smoothing, gradient clipping, LR plateau scheduler, and early stopping.
- **Checkpoint policy:** Final metrics are computed from the best validation checkpoint, not just last epoch.

## Training Process
### Baseline ViT Transformations
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Latest Hybrid Training Pipeline (notebook)
```python
# 1) build video records -> split by video -> create sequence samples
video_records, class_names, class_to_idx = build_video_records(DATA_PATH, min_frames=SEQ_LEN)
train_videos, val_videos = split_video_records(video_records, val_ratio=VAL_RATIO, seed=SEED)
train_samples = make_sequence_samples(train_videos, seq_len=SEQ_LEN, seq_stride=SEQ_STRIDE)
val_samples = make_sequence_samples(val_videos, seq_len=SEQ_LEN, seq_stride=SEQ_STRIDE)

# 2) model + optimizer + scheduler
model = TemporalHybridModel(...).to(device)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

# 3) train and save best checkpoint
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc, _, _, _ = evaluate(...)
    scheduler.step(val_acc)
    if val_acc improved:
        save best checkpoint
```

## Evaluation Metrics
The evaluation pipeline reports accuracy, precision, recall, F1, confusion matrix, and ROC-AUC (binary mode). It also prints label and prediction counts for sanity checks.

```python
precision = precision_score(labels_all, preds_all, average="weighted", zero_division=0)
recall = recall_score(labels_all, preds_all, average="weighted", zero_division=0)
f1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0)
print(classification_report(labels_all, preds_all, target_names=class_names, zero_division=0))

cm = confusion_matrix(labels_all, preds_all)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
```

## Video Prediction
The Flask backend serves frame-level inference using the ViT checkpoint in `backend/models/best_vit_model.pth` and streams running real/fake percentages to the frontend.

```python
def predict_video(video_path, model, transform, device):
    cap = cv2.VideoCapture(video_path)
    real_count, manipulated_count = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        real_count += (predicted.item() == 0)
        manipulated_count += (predicted.item() == 1)
    cap.release()
```

## Installation and Setup
### Frontend (React)
```bash
cd DeepShield
npm install
npm start
```

### Backend (Flask)
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Model Files
- Place your trained model file at `backend/models/best_vit_model.pth`.
- Latest hybrid experiment notebook: `notebooks/frac_df_cnnvit_fft_temporal.ipynb`.
- Hybrid training checkpoint (default filename): `small_cnn_vit_fft_lstm_model.pth`.

### CUDA Verification
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

## Results
### Baseline ViT (frame-level)
- Training Accuracy: ~89.71%
- Validation Accuracy: ~87.77%

### Latest Hybrid (video-level, experimental)
- Architecture and training flow are updated in the notebook with temporal + FFT fusion.
- Current runs are under active tuning; use notebook metric cells for the latest measured accuracy/F1.

## Website Usage
![Website Landing Page](docs/Img1.png)
![Upload Interface](docs/Img2.png)
![Processing Results](docs/Img3.png)

## Contributors
- **Rohit N** — rohit84.official@gmail.com — [LinkedIn](https://www.linkedin.com/in/rohit-n-1b0984280)
- **Rahul B** — rahulbalachandar24@gmail.com — [LinkedIn](https://www.linkedin.com/in/rahul-balachandar-a9436a293)
- **Yadeesh T** — yadeesh005@gmail.com — [LinkedIn](https://www.linkedin.com/in/yadeesh-t-259640288)
- **Gokul Ram K** — gokul.ram.kannan210905@gmail.com — [LinkedIn](https://www.linkedin.com/in/gokul-ram-k-277a6a308)
