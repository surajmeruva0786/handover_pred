# i-DECIDE Handover Prediction — Real-World LTE Drive-Test Implementation

Implementation of the **i-DECIDE two-stage deep learning pipeline** for proactive LTE handover prediction, applied to a real-world vehicular drive-test dataset (`network_logs_1.csv`) collected in India.

> **Architecture source:** Lima et al., "Deep Learning-Based Handover Prediction for 5G and Beyond Networks", IEEE ICC 2023 (`AS_JNL_IEEE_iDecide_Backup_1.pdf`)

---

## What This Project Does

1. **Stage 1 — LSTM RSRP Regression:** A stacked 4-layer LSTM predicts the next RSRP value from a 100-sample lookback window of historical signal measurements.
2. **Stage 2 — Binary Handover Classification:** 50-sample windows of LSTM-predicted RSRP are classified as handover (HO=1) or non-handover (HO=0) using four binary classifiers (Random Forest, SVM, MLP, KNN), with Tomek Links + SMOTE resampling to correct class imbalance.

---

## Dataset

| Property | Value |
|----------|-------|
| File | `network_logs_1.csv` |
| Raw rows | 10,570 |
| LTE-4G samples used | **10,301** |
| Handover events | **323** (3.14% HO rate) |
| Unique PCI values | 169 |
| RSRP range | −129 to −60 dBm |
| Device | Samsung SM-S901E |
| Network | Airtel (India) |
| Collection period | 28–31 August 2025 (drive test) |

---

## Results

### Stage 1 — LSTM

| Metric | Value |
|--------|-------|
| Test MAE | **1.4427 dBm** |
| Test RMSE | **2.7585 dBm** |
| Epochs (early stopped) | 20 (best at epoch 10) |
| Best val_loss | 0.001393 |

### Stage 2 — Classification (`10 × StratifiedKFold(5)`, 50 folds, 19,604 balanced samples)

| Classifier | Accuracy | F1-Score | Precision | Recall |
|------------|----------|----------|-----------|--------|
| **Random Forest** | **97.41%** ±0.26% | **97.46%** ±0.25% | 95.86% | **99.11%** |
| KNN (k=2) | 96.90% ±0.24% | 96.91% ±0.24% | 96.49% | 97.34% |
| MLP (120-120) | 95.09% ±0.39% | 95.12% ±0.38% | 94.72% | 95.52% |
| SVM (RBF, C=500) | 88.24% ±0.53% | 88.03% ±0.54% | 89.64% | 86.49% |

**Best model: Random Forest** — 97.41% accuracy, 99.11% HO recall (misses only 17.5 handover events per fold).

---

## Repository Structure

```
handover_pred/
├── network_logs_1.csv              # Drive-test LTE measurement log
├── network_logs_pipeline.py        # Main pipeline — run this
├── AS_JNL_IEEE_iDecide_Backup_1.pdf# Source architecture paper
├── CLAUDE.md                       # Project context and architecture reference
├── PROGRESS.md                     # Full training log and results
│
├── paper/                          # IEEE journal paper (LaTeX)
│   ├── main.tex                    # Complete paper source
│   ├── lstm_training_loss.png      # Figure: LSTM loss curve
│   └── lstm_rsrp_prediction.png    # Figure: Real vs predicted RSRP
│
├── results/                        # Pipeline outputs
│   ├── preprocessed_data.csv       # Cleaned LTE-4G data with ho_trig labels
│   ├── classification_base.csv     # Raw 50-window classification features
│   ├── balanced_classification_base.csv  # Post Tomek+SMOTE features
│   ├── lstm_model.keras            # Saved LSTM model weights
│   ├── lstm_training_loss.png      # Train vs val loss curve
│   ├── lstm_rsrp_prediction.png    # RSRP real vs predicted
│   ├── results_summary.json        # All metrics (machine-readable)
│   └── run_log.txt                 # Full console output of last run
│
├── real_network/                   # Original i-DECIDE real-network scripts
│   ├── scripts/
│   │   ├── anatel_data_processing.py
│   │   ├── anatel_lstm_rsrp.py
│   │   └── anatel_sampling_and_classify.py
│   └── real_data/
│       ├── drive_test_measurements01.csv
│       ├── drive_test_measurements02.csv
│       └── drive_test_measurements03.csv
│
└── simulation/                     # Original i-DECIDE simulation scripts
    ├── scripts/
    │   ├── data_processing.py
    │   ├── lstm_rsrp.py
    │   └── sampling_and_classify.py
    └── sim_data/
        └── DlRsrpSinrStats_*.txt   # ns-3 simulation RSRP/SINR logs
```

---

## How to Run

```bash
# Install dependencies
pip install tensorflow scikit-learn imbalanced-learn pandas numpy matplotlib

# Run the full pipeline (takes ~80 min on CPU)
python network_logs_pipeline.py
```

All outputs are saved to `results/`. The pipeline is fully reproducible (seed=42).

---

## Hardware Used

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 4050 Laptop, 6 GB VRAM |
| CUDA toolkit | 13.0 (driver 592.27) |
| TF GPU note | TF 2.11+ has no native Windows GPU support; all training ran on CPU |
| Training time | LSTM: ~545 s · All classifiers: ~4211 s |

---

## Paper

A full IEEE-format journal paper is available in `paper/main.tex`.  
To compile: upload `paper/` folder to [Overleaf](https://overleaf.com) and click Compile (IEEEtran class, no extra packages needed).

---

## Architecture (i-DECIDE)

### Stage 1 — LSTM

| Layer | Config |
|-------|--------|
| Input | shape (100, 1) |
| LSTM-1 | 120 units, return_sequences=True + Dropout(0.3) |
| LSTM-2 | 50 units, return_sequences=True + Dropout(0.3) |
| LSTM-3 | 50 units, return_sequences=True + Dropout(0.3) |
| LSTM-4 | 50 units + Dropout(0.3) |
| Output | Dense(1, linear) |
| Optimizer | RMSprop |
| Loss | MSE |
| Epochs | 100 max, EarlyStopping patience=10 |
| Batch size | 128 |

### Stage 2 — Classification

| Parameter | Value |
|-----------|-------|
| Feature window | 50 LSTM-predicted RSRP samples |
| Class balancing | Tomek Links (under) → SMOTE (over) |
| Scaling | StandardScaler |
| Classifiers | RF (n=200), SVM (RBF, C=500), MLP ([120,120]), KNN (k=2) |
| Validation | 10 × StratifiedKFold(5) = 50 folds each |

---

## Original Paper Citation

```bibtex
@inproceedings{lima2023idecide,
  title     = {Deep Learning-Based Handover Prediction for 5G and Beyond Networks},
  author    = {Lima, Jo{\~a}o P. S. H. and de Medeiros, Alvaro A. M. and
               de Aguiar, Eduardo P. and Silva, Edelberto F. and
               de Sousa Jr., Vicente A. and Nunes, Marcelo L. and Reis, Alysson L.},
  booktitle = {IEEE International Conference on Communications (ICC)},
  year      = {2023}
}
```
