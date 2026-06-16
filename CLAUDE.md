# Handover Prediction — Project Context for Claude

## What This Project Does
Implements the two-stage deep learning handover prediction architecture from the IEEE paper (`AS_JNL_IEEE_iDecide_Backup_1.pdf`) applied to real-world drive-test network logs (`network_logs_1.csv`).

## User Preferences
- **No time restrictions** — always run on full dataset, never subsample or approximate
- **Best possible results** — use paper-exact hyperparameters and full cross-validation
- **GPU training** — use RTX 4050 (6 GB VRAM, CUDA 13.0) whenever possible
- **Document everything** — update PROGRESS.md and this file with every meaningful change, result, or decision

## Architecture (from paper)

### Stage 1 — LSTM RSRP Regression
| Component | Value |
|-----------|-------|
| Lookback window | 100 samples |
| LSTM Layer 1 | 120 units, return_sequences=True + Dropout(0.3) |
| LSTM Layer 2 | 50 units, return_sequences=True + Dropout(0.3) |
| LSTM Layer 3 | 50 units, return_sequences=True + Dropout(0.3) |
| LSTM Layer 4 | 50 units + Dropout(0.3) |
| Output | Dense(1, linear) |
| Optimizer | RMSprop |
| Loss | MSE |
| Epochs | 100 (EarlyStopping patience=10) |
| Batch size | 128 |

### Stage 2 — Binary Classification
| Component | Value |
|-----------|-------|
| Feature window | 50 consecutive LSTM predictions |
| Class balancing | Tomek Links (under) → SMOTE (over) |
| Scaling | StandardScaler |
| Classifiers | Random Forest (n=200), SVM (rbf, C=500), MLP ([120,120]), KNN (k=2) |
| Validation | 10 × StratifiedKFold(5 splits) = 50 folds each |

## Dataset
- File: `network_logs_1.csv`
- 10,570 raw rows → **10,301 LTE-4G samples** (after filtering WCDMA/5G/2G)
- Single device: Samsung SM-S901E on Airtel
- Timespan: 2025-08-28 to 2025-08-31 (drive test)
- **323 handover events** detected (PCI changes), HO rate: 3.14%
- RSRP range: -129 to -60 dBm
- 169 unique PCI values

## Hardware
- GPU: NVIDIA GeForce RTX 4050 Laptop, 6 GB VRAM
- CUDA 13.0 toolkit installed
- Driver version 592.27 (supports CUDA 13.1)
- TensorFlow 2.20.0 — GPU support being configured via `tensorflow[and-cuda]`

## Files
| File | Purpose |
|------|---------|
| `network_logs_pipeline.py` | Main end-to-end pipeline (run this) |
| `PROGRESS.md` | Detailed training log and results |
| `CLAUDE.md` | This file — project context for Claude |
| `results/preprocessed_data.csv` | Cleaned LTE-4G data with ho_trig labels |
| `results/classification_base.csv` | Raw 50-window classification features |
| `results/balanced_classification_base.csv` | Post Tomek+SMOTE balanced features |
| `results/lstm_model.keras` | Saved LSTM model weights |
| `results/lstm_training_loss.png` | Train vs val loss curve |
| `results/lstm_rsrp_prediction.png` | RSRP real vs predicted on test set |
| `results/results_summary.json` | All metrics in machine-readable format |
| `results/run_log.txt` | Full console output of last pipeline run |

## Run 1 Results (completed 2026-06-16)

### LSTM Stage
| Metric | Value |
|--------|-------|
| Epochs (early stopped) | 20 (best at epoch 10) |
| Best val_loss | 0.001393 |
| Test MAE | **1.4427 dBm** |
| Test RMSE | **2.7585 dBm** |

### Classification Stage (10×5-fold CV on 19,604 balanced samples)
| Classifier | Accuracy | F1 | Recall |
|------------|----------|----|--------|
| **Random Forest** | **97.41%** | **97.46%** | **99.11%** |
| KNN | 96.90% | 96.91% | 97.34% |
| MLP | 95.09% | 95.12% | 95.52% |
| SVM (RBF) | 88.24% | 88.03% | 86.49% |

**Best model: Random Forest** — achieves paper target of >97% accuracy.

## GPU Setup Status
- TF 2.20 cannot detect RTX 4050 — needs CUDA 12.x, system has CUDA 13.0
- PyTorch installed but CPU-only build
- CUDA 12 pip packages timed out during download
- **Next step:** retry CUDA 12 pip install with higher timeout for GPU LSTM retraining

## Key Decisions
- Filtered LTE-4G only (excluded WCDMA/3G, 5G NR, GSM/2G) — paper targets LTE/NR
- Sorted timestamps ascending (raw data was newest-first)
- Used EarlyStopping (patience=10) to prevent overfitting while respecting 100-epoch max
- SMOTE k_neighbors set dynamically based on minority class size
- All 4 classifiers evaluated at 10×5-fold CV as per paper (no shortcuts)
