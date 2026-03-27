# Model Card: Naso-Net

## Model Details

| Field | Value |
|---|---|
| **Model name** | Naso-Net |
| **Version** | 1.0 |
| **Type** | Temporal-aware image classification (binary) |
| **Architecture** | Time-Distributed ResNet50 + Weighted Mean Voting (WMV) |
| **Framework** | TensorFlow 2.10 / Keras |
| **Task** | Velopharyngeal port (VPP) closure prediction from nasopharyngoscopy video |
| **License** | MIT |
| **Paper** | *Automated detection of velopharyngeal port dynamics from nasopharyngoscopy videos using deep learning*, PLOS ONE (2026) |

---

## Intended Use

- **Primary use**: Research tool for automated analysis of velopharyngeal port dynamics in nasopharyngoscopy videos.
- **Intended users**: Researchers in medical image analysis, speech-language pathology, and craniofacial surgery.
- **Out-of-scope**: This model is **not** intended for standalone clinical diagnosis. It has been validated only on a single-center dataset and requires further multicenter validation before any clinical deployment.

---

## Architecture

```
Input: (batch, 45, H, W, 3) — 45-frame temporal window
  │
  ├── TimeDistributed(ResNet50, ImageNet pretrained)
  │     └── Last 20 layers fine-tuned
  ├── TimeDistributed(GlobalAveragePooling2D)
  ├── TimeDistributed(Dense(256, gelu))
  ├── TimeDistributed(BatchNormalization)
  ├── TimeDistributed(Dropout(0.5))
  │
  ├── Frame probability head: Dense(1, sigmoid) → p_i
  ├── Frame weight head: Dense(1, softplus) → w_i
  │
  └── WMV: output = Σ(p_i × w_i) / Σ(w_i)
```

**WMV (Weighted Mean Voting)**: Each frame produces a probability $p_i$ and an importance weight $w_i$. The sequence-level prediction is the weighted average, allowing the model to learn which frames are most informative for the classification decision.

---

## Training Details

### Hyperparameters

| Parameter | Value |
|---|---|
| Backbone | ResNet50 (ImageNet pretrained) |
| Trainable layers | Last 20 of ResNet50 + all dense heads |
| Input resolution | 90×90 (main), 128×128 / 160×160 (ablation) |
| Temporal window | 45 frames (~1.5 seconds at 30 fps) |
| Batch size | 8 |
| Optimizer | Adam |
| Learning rate | 1e-3 (max), 1e-4 (base) |
| LR schedule | OneCycleLR (pct_start=0.3) |
| Max epochs | 75 |
| Early stopping | patience=15, monitor=val_loss |
| Checkpoint | Best val_auc per fold |
| Loss | Binary cross-entropy |
| Activation | GELU |
| Dropout | 0.5 (backbone head), varies by layer |

### Cross-Validation

- **Method**: GroupKFold (scikit-learn), 10 folds
- **Grouping variable**: Patient ID (ensures no patient appears in both train and test)
- **Random seed**: 42

### Data Augmentation (when applied)

| Level | Rotation | Width/Height shift | Shear | Zoom | Flip |
|---|---|---|---|---|---|
| None | — | — | — | — | — |
| Conservative | ±10° | ±5% | ±5° | ±5% | Horizontal |
| Moderate | ±15° | ±10% | ±10° | ±10% | Horizontal |

---

## Dataset

| Statistic | Value |
|---|---|
| Patients | 24 (25 video clips) |
| Total frames | 78,849 |
| Temporal sequences | 629 (322 closed, 307 open) |
| Frame rate | 30 fps |
| Center | Single tertiary academic children's hospital |
| Video type | Nasopharyngoscopy (NP) |

---

## Performance

### Main Results (90×90, conservative augmentation, window=45)

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 80.22% | [77.5%, 82.8%] |
| AUC | 82.36% | [78.4%, 84.5%] |

### Baseline Comparisons (128×128, no augmentation)

| Model | Accuracy [95% CI] | AUC [95% CI] | F1 | Fold AUC σ |
|---|---|---|---|---|
| ResNet50+MeanPool | 0.740 [0.665, 0.795] | 0.777 [0.684, 0.847] | 0.734 | 0.154 |
| **Naso-Net (WMV)** | **0.788 [0.696, 0.834]** | **0.792 [0.707, 0.852]** | **0.748** | **0.094** |
| ResNet50+LSTM | 0.687 [0.619, 0.752] | 0.717 [0.646, 0.795] | 0.683 | 0.142 |

### Ablation: Resolution × Augmentation (best result per config)

| Resolution | Augmentation | Accuracy [95% CI] | AUC [95% CI] |
|---|---|---|---|
| 128×128 | none | 0.696 [0.646, 0.744] | 0.732 [0.667, 0.802] |
| 128×128 | conservative | 0.680 [0.625, 0.736] | 0.721 [0.660, 0.789] |
| 128×128 | moderate | 0.688 [0.638, 0.738] | 0.725 [0.665, 0.790] |
| 160×160 | none | 0.678 [0.625, 0.731] | 0.717 [0.659, 0.779] |
| 160×160 | conservative | 0.680 [0.627, 0.733] | 0.723 [0.667, 0.774] |
| 160×160 | moderate | 0.669 [0.629, 0.708] | 0.721 [0.675, 0.766] |

---

## Limitations

- **Single-center data**: Model was trained and validated on data from one pediatric hospital. Generalization to other centers, patient populations, or endoscope hardware has not been tested.
- **Small sample size**: 24 patients / 629 sequences. Performance metrics have wide confidence intervals.
- **Binary classification only**: The model predicts open vs. closed; it does not characterize closure patterns (coronal, sagittal, circular) or degree of closure.
- **No audio integration**: Only visual frames are analyzed; speech audio is not used.

---

## Ethical Considerations

- The dataset was collected under IRB approval. Raw videos are not publicly shared due to patient privacy (HIPAA).
- The model is intended for research purposes only and should not be used for clinical decision-making without further validation.
- Annotation was performed by trained experts; inter-rater variability was not formally quantified.

---

## Contact

For questions, data access requests, or collaboration inquiries, please contact the corresponding author:
- **Miles J. Pfaff, MD** — mpfaff@hs.uci.edu
