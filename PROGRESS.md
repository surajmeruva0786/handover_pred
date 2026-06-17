# Handover Prediction — Progress Log

## Paper Architecture Applied
**Source:** `AS_JNL_IEEE_iDecide_Backup_1.pdf`
**Title:** *Deep Learning-Based Handover Prediction for 5G and Beyond Networks*
**Authors:** Lima et al., IEEE ICC 2023

---

## Architecture Overview

### Stage 1 — LSTM RSRP Regression

| Parameter | Value |
|-----------|-------|
| Lookback | 100 samples |
| Layer 1 | LSTM(120, return_sequences=True) + Dropout(0.3) |
| Layer 2 | LSTM(50, return_sequences=True) + Dropout(0.3) |
| Layer 3 | LSTM(50, return_sequences=True) + Dropout(0.3) |
| Layer 4 | LSTM(50) + Dropout(0.3) |
| Output | Dense(1, activation='linear') |
| Optimizer | RMSprop |
| Loss | Mean Squared Error (MSE) |
| Max Epochs | 100 (EarlyStopping patience=10) |
| Batch size | 128 |
| Train/Test | 80% / 20% (chronological, no shuffle) |
| Total params | 133,691 |

### Stage 2 — Binary Classification for HO Prediction

| Parameter | Value |
|-----------|-------|
| Feature window | 50 LSTM-predicted RSRP samples |
| Class balancing | Tomek Links (under) → SMOTE (over) |
| Feature scaling | StandardScaler (z-score) |
| Classifiers | Random Forest (n=200), SVM (RBF, C=500), MLP ([120,120]), KNN (k=2) |
| Validation | 10 × StratifiedKFold(5 splits) = 50 folds each |
| Metrics | Accuracy, F1, Precision, Recall, Confusion Matrix |

---

## Dataset: `network_logs_1.csv`

| Property | Value |
|----------|-------|
| Raw rows | 10,570 |
| Network breakdown | LTE (4G): 10,301 · WCDMA (3G): 99 · GSM (2G): 98 · 5G NR: 60 · Not Registered: 12 |
| LTE rows used | **10,301** |
| Device | Samsung SM-S901E |
| Network provider | Airtel (India) |
| Data collection | Drive test |
| Timespan | 2025-08-28 14:36 to 2025-08-31 22:12 |
| RSRP range | −129 dBm to −60 dBm |
| Unique PCI values | 169 |
| Handover detection | PCI change between consecutive samples |

---

## Files Created

| File | Description |
|------|-------------|
| `network_logs_pipeline.py` | Main end-to-end pipeline script |
| `CLAUDE.md` | Project context, architecture, hardware notes |
| `README.md` | Full project README with results and usage |
| `paper/main.tex` | Complete IEEE journal paper (LaTeX) |
| `paper/lstm_training_loss.png` | Paper figure: LSTM loss curve |
| `paper/lstm_rsrp_prediction.png` | Paper figure: real vs predicted RSRP |
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
- CPU: All training ran on CPU
- GPU: NVIDIA RTX 4050 Laptop (6 GB VRAM, CUDA 13.0) — see GPU note below
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
| KNN (k=2) | 96.90% | ±0.24% | 96.91% | ±0.24% | 96.49% | 97.34% | 41.8 s |
| MLP (120-120) | 95.09% | ±0.39% | 95.12% | ±0.38% | 94.72% | 95.52% | 2606.4 s |
| SVM (RBF, C=500) | 88.24% | ±0.53% | 88.03% | ±0.54% | 89.64% | 86.49% | 1466.7 s |

#### Confusion Matrices (mean over 50 folds, ~3,920 samples per fold)

**Random Forest:**
```
              Predicted 0   Predicted 1
Actual 0      1876.52         83.88
Actual 1        17.54       1942.86
```

**KNN (k=2):**
```
              Predicted 0   Predicted 1
Actual 0      1891.00         69.40
Actual 1        52.18       1908.22
```

**MLP (120-120):**
```
              Predicted 0   Predicted 1
Actual 0      1855.90        104.50
Actual 1        87.82       1872.58
```

**SVM (RBF, C=500):**
```
              Predicted 0   Predicted 1
Actual 0      1764.24        196.16
Actual 1       264.84       1695.56
```

---

## GPU Setup Status — Final Diagnosis

| Step | Status |
|------|--------|
| Hardware | NVIDIA RTX 4050 Laptop, 6 GB VRAM |
| CUDA toolkit installed | CUDA 13.0 (nvcc v13.0, driver 592.27) |
| CUDA 12 pip packages | Installed: nvidia-cuda-runtime-cu12, nvidia-cublas-cu12, nvidia-cudnn-cu12 |
| DLL path fix | Added via `os.add_dll_directory()` for all nvidia/*/bin dirs |
| TF CUDA build | `tf.test.is_built_with_cuda()` returns **False** |
| Root cause | **TF 2.11+ dropped native Windows GPU support.** TF 2.10 was the last Windows GPU release. |
| Workaround | WSL2 + Ubuntu + CUDA inside Linux for GPU acceleration |
| Impact | Run 1 completed on CPU; results are fully valid |

---

## Paper — 2026-06-17

A complete IEEE journal-format paper was written in LaTeX covering the i-DECIDE implementation and results.

### Paper Contents

| Section | Content |
|---------|---------|
| Abstract | Dataset, two-stage method, key results |
| Introduction | HO problem, A3-event limitations, contributions |
| Dataset | Drive-test setup, RAT filtering, HO detection, statistics |
| i-DECIDE Architecture | Stage 1 LSTM + Stage 2 classifiers, TikZ block diagram |
| Experimental Setup | CV protocol, training configuration, hardware |
| Results | LSTM table, classification table (50-fold), confusion matrices, pgfplots bar chart |
| Conclusion | Summary + 4 future directions |
| References | 8 references (3GPP specs, i-DECIDE, LSTM, SMOTE, Tomek, RMSprop, RF, measurements) |

### Paper Files

| File | Description |
|------|-------------|
| `paper/main.tex` | Complete LaTeX source (IEEEtran journal class) |
| `paper/lstm_training_loss.png` | LSTM training/validation loss curve |
| `paper/lstm_rsrp_prediction.png` | Actual vs predicted RSRP on test set |

**To compile:** Upload all 3 files to [Overleaf](https://overleaf.com) → Compile.

### Known Issues Fixed
- Architecture TikZ figure overflowed page width → fixed with `\resizebox{\linewidth}{!}{}` wrapper and `raw` node included in Stage 1 bounding box (commit `72b517d`)

---

## Commit History

| Commit | Description |
|--------|-------------|
| `5c43c66` | Add two-stage handover prediction pipeline and PROGRESS.md |
| `7a0fd49` | Add CLAUDE.md project context and update PROGRESS.md |
| `e4ab5c6` | Enable GPU memory growth for RTX 4050 in pipeline |
| `ab5a174` | Update PROGRESS.md and CLAUDE.md with full Run 1 results |
| `a833e55` | Fix GPU DLL path setup and document TF Windows GPU limitation |
| `db093c3` | Add IEEE journal paper (main.tex) with full results |
| `6fbfc95` | Rewrite IEEE paper — focused purely on i-DECIDE implementation |
| `72b517d` | Fix architecture figure overflow — resizebox to linewidth |
| _(latest)_ | Document everything — update README, PROGRESS, CLAUDE |
