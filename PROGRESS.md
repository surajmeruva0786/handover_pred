# Handover Prediction — Progress Log

## Paper Architecture Applied
**Source:** `AS_JNL_IEEE_iDecide_Backup_1.pdf`  
**Title:** *Deep Learning-Based Handover Prediction for 5G and Beyond Networks*  
**Authors:** Lima et al., IEEE ICC 2023 / IEEE Journal  

---

## Architecture Overview

### Stage 1 — LSTM RSRP Regression
Predicts future Reference Signal Received Power (RSRP) values using a stacked LSTM.

| Parameter     | Value                                           |
|---------------|-------------------------------------------------|
| Lookback      | 100 samples                                     |
| Layer 1       | LSTM(120, return_sequences=True) + Dropout(0.3) |
| Layer 2       | LSTM(50,  return_sequences=True) + Dropout(0.3) |
| Layer 3       | LSTM(50,  return_sequences=True) + Dropout(0.3) |
| Layer 4       | LSTM(50)                         + Dropout(0.3) |
| Output        | Dense(1, activation='linear')                   |
| Optimizer     | RMSprop                                         |
| Loss          | Mean Squared Error (MSE)                        |
| Max Epochs    | 100 (EarlyStopping patience=10)                 |
| Batch size    | 128                                             |
| Train/Test    | 80% / 20%                                       |
| Total params  | 133,211 (520 KB)                                |

### Stage 2 — Binary Classification for HO Prediction
Takes 50 consecutive LSTM-predicted RSRP values as a feature vector.

| Parameter        | Value                                           |
|------------------|-------------------------------------------------|
| Feature window   | 50 LSTM-predicted RSRP samples                  |
| Class balancing  | Tomek Links (under) → SMOTE (over)              |
| Feature scaling  | StandardScaler (z-score)                        |
| Classifiers      | Random Forest, SVM (RBF), MLP, KNN              |
| Validation       | 10 × StratifiedKFold(5 splits) = 50 folds each  |
| Metrics          | Accuracy, F1, Precision, Recall, Confusion Matrix |

---

## Dataset: `network_logs_1.csv`

| Property            | Value                                            |
|---------------------|--------------------------------------------------|
| Raw rows            | 10,570                                           |
| Network breakdown   | LTE (4G): 10,301 · WCDMA (3G): 99 · GSM (2G): 98 · 5G NR: 60 · Not Registered: 12 |
| LTE rows used       | **10,301** (after filtering + cleaning)          |
| Device              | Samsung SM-S901E (single device)                 |
| Network provider    | Airtel                                           |
| Data collection     | Drive test                                       |
| Timespan            | 2025-08-28 14:36 → 2025-08-31 22:12             |
| RSRP range          | −129 dBm to −60 dBm                             |
| Unique PCI values   | 169                                              |
| Handover detection  | PCI change between consecutive samples           |

---

## Files Created

| File | Description |
|------|-------------|
| `network_logs_pipeline.py` | Main end-to-end pipeline script |
| `CLAUDE.md` | Project context, architecture, hardware notes |
| `results/preprocessed_data.csv` | Cleaned LTE-4G dataset with ho_trig labels |
| `results/classification_base.csv` | Raw 50-window classification features |
| `results/balanced_classification_base.csv` | Post-Tomek/SMOTE balanced features |
| `results/lstm_model.keras` | Saved LSTM model (best weights) |
| `results/lstm_training_loss.png` | Train vs validation loss curve |
| `results/lstm_rsrp_prediction.png` | Real vs predicted RSRP on test set |
| `results/results_summary.json` | All metrics in JSON format |
| `results/run_log.txt` | Full console output of run |

---

## Run 1 Results — 2026-06-16

### Hardware
- CPU: Training ran on CPU (Intel/AMD)
- GPU: NVIDIA RTX 4050 Laptop (6 GB VRAM, CUDA 13.0) — GPU support being configured
- TF version: 2.20.0, Keras 3.14.1

### Step 1 — Data Preprocessing

| Metric | Value |
|--------|-------|
| Raw rows | 10,570 |
| LTE-4G rows (used) | **10,301** |
| Handover events (ho_trig=1) | **323** |
| No-HO samples (ho_trig=0) | **9,978** |
| HO rate | **3.14%** |
| RSRP range | −129 to −60 dBm |
| Unique PCI values | 169 |
| Preprocessing time | 0.4 s |

### Step 2 — LSTM RSRP Prediction (Stage 1)

| Metric | Value |
|--------|-------|
| Train samples | 8,140 sequences (lookback=100) |
| Test samples | 2,061 sequences |
| Training epochs run | **20** (early stopped; best at epoch 10) |
| Best val_loss (MSE) | **0.001393** |
| **Test MAE** | **1.4427 dBm** |
| **Test RMSE** | **2.7585 dBm** |
| Training time | 545.5 s (~9 min) |

> **Note:** The paper reports MAE ≈ 0.6 dB on simulation data with highly regular RSRP patterns. Our dataset is real-world drive-test data with 169 distinct cells, irregular movement, and abrupt PCI transitions — making RSRP prediction harder. An MAE of 1.44 dBm on this data is consistent with real-network complexity.

### Step 3 — Classification Feature Construction

| Metric | Value |
|--------|-------|
| Full-set LSTM prediction shape | (10,201, 1) |
| Classification windows | **10,151 samples** (50-step sliding window) |
| Class-0 (no HO) before balance | 9,841 |
| Class-1 (HO) before balance | 310 |
| After Tomek Links | 9,802 (no-HO), 310 (HO) |
| After SMOTE | **9,802 (no-HO), 9,802 (HO)** |
| Total balanced samples | **19,604** |
| Feature construction time | 8.2 s |
| Balancing time | 2.2 s |

### Step 4 — Classification Results (Stage 2)

**10 × StratifiedKFold(5 splits) cross-validation on balanced dataset of 19,604 samples.**

| Classifier | Accuracy | Std | F1-Score | Std | Precision | Recall | Time |
|------------|----------|-----|----------|-----|-----------|--------|------|
| **Random Forest** | **97.41%** | ±0.26% | **97.46%** | ±0.25% | 95.86% | **99.11%** | 92.4 s |
| SVM (RBF) | 88.24% | ±0.53% | 88.03% | ±0.54% | 89.64% | 86.49% | 1466.7 s |
| MLP | 95.09% | ±0.39% | 95.12% | ±0.38% | 94.72% | 95.52% | 2606.4 s |
| KNN | 96.90% | ±0.24% | 96.91% | ±0.24% | 96.49% | 97.34% | 41.8 s |

#### Confusion Matrices (mean over 50 folds, class 0 = no-HO, class 1 = HO)

**Random Forest:**
```
              Predicted 0   Predicted 1
Actual 0      1876.52       83.88
Actual 1        17.54     1942.86
```

**SVM (RBF):**
```
              Predicted 0   Predicted 1
Actual 0      1764.24      196.16
Actual 1       264.84     1695.56
```

**MLP:**
```
              Predicted 0   Predicted 1
Actual 0      1855.90      104.50
Actual 1        87.82     1872.58
```

**KNN:**
```
              Predicted 0   Predicted 1
Actual 0      1891.00       69.40
Actual 1        52.18     1908.22
```

### Summary vs Paper

| Metric | Paper (simulation) | Ours (real-world drive test) |
|--------|-------------------|------------------------------|
| LSTM MAE | ~0.6 dB | 1.44 dBm |
| Best classifier | Random Forest | Random Forest |
| Best accuracy | >97% | **97.41%** |
| Best F1-score | >97% | **97.46%** |
| Best recall (HO detection) | — | **99.11%** |

The paper's target of >97% accuracy is **achieved** with Random Forest. The higher MAE reflects real-world signal variability (169 cells, drive-test mobility) vs controlled simulation.

---

## GPU Setup Status

| Step | Status |
|------|--------|
| Hardware detected | NVIDIA RTX 4050 Laptop, 6 GB VRAM |
| CUDA toolkit installed | CUDA 13.0 (nvcc v13.0) |
| TF GPU detection | Not working — TF 2.20 needs CUDA 12.x, system has CUDA 13.0 |
| PyTorch GPU | CPU-only build installed |
| CUDA 12 pip packages | Download timed out |
| Next step | Install CUDA 12 runtime libs (retry with higher timeout) |

---

## Commit History

| Commit | Description |
|--------|-------------|
| `5c43c66` | Add two-stage handover prediction pipeline and PROGRESS.md |
| `7a0fd49` | Add CLAUDE.md project context and update PROGRESS.md |
| `e4ab5c6` | Enable GPU memory growth for RTX 4050 in pipeline |
| _(next)_ | Update PROGRESS.md with Run 1 results + GPU retry |
