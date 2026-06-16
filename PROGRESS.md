# Handover Prediction — Progress Log

## Paper Architecture Applied
**Source:** `AS_JNL_IEEE_iDecide_Backup_1.pdf`  
**Title:** *Deep Learning-Based Handover Prediction for 5G and Beyond Networks*  
**Authors:** Lima et al., IEEE ICC 2023 / IEEE Journal  

---

## Architecture Overview

The pipeline consists of two stages:

### Stage 1 — LSTM RSRP Regression
Predicts future Reference Signal Received Power (RSRP) values using a stacked LSTM.

| Parameter     | Value                          |
|---------------|--------------------------------|
| Lookback      | 100 samples                    |
| Layer 1       | LSTM(120, return_sequences=True) + Dropout(0.3) |
| Layer 2       | LSTM(50, return_sequences=True)  + Dropout(0.3) |
| Layer 3       | LSTM(50, return_sequences=True)  + Dropout(0.3) |
| Layer 4       | LSTM(50)                         + Dropout(0.3) |
| Output        | Dense(1, activation='linear')   |
| Optimizer     | RMSprop                        |
| Loss          | Mean Squared Error (MSE)       |
| Epochs        | 100 (w/ EarlyStopping, patience=10) |
| Batch size    | 128                            |
| Train/Test    | 80% / 20%                      |
| Metric        | MAE (dBm)                      |

### Stage 2 — Binary Classification for HO Prediction
Takes 50 consecutive LSTM-predicted RSRP values as feature vector, labels each as HO (1) or no-HO (0).

| Parameter        | Value                                     |
|------------------|-------------------------------------------|
| Feature window   | 50 LSTM-predicted RSRP samples            |
| Class balancing  | Tomek Links (under) → SMOTE (over)        |
| Feature scaling  | StandardScaler (z-score)                  |
| Classifiers      | Random Forest, SVM (RBF), MLP, KNN        |
| Validation       | 10 × StratifiedKFold (5 splits each = 50 folds) |
| Metrics          | Accuracy, F1, Precision, Recall, Confusion Matrix |

---

## Dataset: `network_logs_1.csv`

| Property            | Value                              |
|---------------------|------------------------------------|
| Raw rows            | 10,570                             |
| Columns             | 15 (Timestamp, DeviceID, RSRP, RSRQ, SINR, PCI, …) |
| Device              | Samsung SM-S901E (single device)   |
| Network types       | LTE (4G), WCDMA (3G)               |
| Data collection     | Drive test — Airtel network        |
| Timespan            | 2025-08-28 → 2025-08-31            |
| Handover detection  | PCI change between consecutive samples |

Only LTE (4G) rows with valid RSRP and PCI are used for training.

---

## Files Created

| File | Description |
|------|-------------|
| `network_logs_pipeline.py` | Main end-to-end pipeline script |
| `results/preprocessed_data.csv` | Cleaned LTE-4G dataset with ho_trig labels |
| `results/classification_base.csv` | Raw 50-window classification features |
| `results/balanced_classification_base.csv` | Class-balanced features |
| `results/lstm_model.keras` | Saved LSTM model |
| `results/lstm_training_loss.png` | Training vs validation loss curve |
| `results/lstm_rsrp_prediction.png` | Real vs predicted RSRP on test set |
| `results/results_summary.json` | All metrics in JSON format |
| `results/run_log.txt` | Full console output of the pipeline run |

---

## Run Log

### Run 1 — Initial training

> Status: **Running** (started 2026-06-16)

_Results will be filled in after training completes._

---

## Results

> Results are populated after `network_logs_pipeline.py` completes.

### Data Preprocessing

| Metric | Value |
|--------|-------|
| Raw rows | — |
| LTE-4G rows used | — |
| Handover events | — |
| HO rate | — |
| RSRP range | — |
| Unique PCI values | — |

### Stage 1 — LSTM RSRP Prediction

| Metric | Value |
|--------|-------|
| MAE (dBm) | — |
| RMSE (dBm) | — |
| Training epochs | — |

### Stage 2 — Classification Results (10×5-Fold CV)

| Classifier    | Accuracy | Std   | F1-Score | Precision | Recall |
|---------------|----------|-------|----------|-----------|--------|
| Random Forest | —        | —     | —        | —         | —      |
| SVM (RBF)     | —        | —     | —        | —         | —      |
| MLP           | —        | —     | —        | —         | —      |
| KNN           | —        | —     | —        | —         | —      |

---

## Commit History

| Commit | Description |
|--------|-------------|
| Initial | Added pipeline script and PROGRESS.md |

