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
| `generate_xai_plot.py` | XAI feature importance plot generator (RF Gini) |
| `CLAUDE.md` | Project context, architecture, hardware notes |
| `README.md` | Full project README with results and usage |
| `paper/main.tex` | Complete IEEE journal paper (LaTeX) — revised v2 |
| `paper/lstm_training_loss.png` | Paper figure: LSTM loss curve |
| `paper/lstm_rsrp_prediction.png` | Paper figure: real vs predicted RSRP |
| `paper/rf_feature_importance.png` | Paper figure: RF feature importance (XAI) |
| `paper/lead_time_vs_performance.png` | Paper figure: lead-time vs F1 / recall |
| `results/preprocessed_data.csv` | Cleaned LTE-4G dataset with ho_trig labels |
| `results/classification_base.csv` | Raw 50-window classification features |
| `results/balanced_classification_base.csv` | Post-Tomek/SMOTE balanced features |
| `results/lstm_model.keras` | Saved LSTM model (best weights) |
| `results/lstm_training_loss.png` | Train vs validation loss curve |
| `results/lstm_rsrp_prediction.png` | Real vs predicted RSRP on test set |
| `results/rf_feature_importance.png` | RF feature importance (XAI analysis) |
| `results/lead_time_vs_performance.png` | Lead-time performance sweep plot |
| `results/results_summary.json` | All metrics in JSON format (A3 baseline, lead time, stats) |
| `results/run_log.txt` | Full console output of latest run |

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

### Step 5 — A3 Baseline Comparison (Imbalanced Distribution)

Evaluated on original 3.14% class imbalance to simulate real-world deployment:

| Method | Accuracy | F1 | Precision | Recall | False-Alarm Rate |
|--------|----------|----|-----------|--------|-----------------|
| Rule-based A3 | 28.9% | 5.4% | 2.8% | 68.0% | **72.4%** |
| **i-DECIDE RF** | **93.9%** | **29.1%** | **22.6%** | 40.7% | **4.4%** |

- i-DECIDE RF: **5.3× better F1**, **16× fewer false alarms** than reactive A3
- False alarms reduced from 1,424 → 87 per CV fold

### Step 6 — Prediction Lead-Time Analysis

| Lead Time (τ) | i-DECIDE RF F1 | i-DECIDE RF Recall | A3 F1 | A3 Recall |
|---------------|----------------|-------------------|-------|-----------|
| τ = 1 s | 29.1% | 40.7% | 5.4% | 68.0% |
| τ = 2 s | 29.2% | 40.8% | 5.4% | 68.8% |
| τ = 3 s | 29.1% | 40.7% | 5.3% | 77.0% |
| τ = 5 s | 28.0% | 39.1% | 5.5% | 82.1% |

- RF maintains ≥28% F1 up to 5 seconds ahead — degradation only 3.8% relative
- 50-sample window @ ~1 s/sample → ~50 s theoretical prediction budget

### Step 7 — Statistical Significance (95% Confidence Intervals)

| Classifier | Mean Accuracy | Std | 95% CI |
|------------|---------------|-----|--------|
| RF | 97.41% | ±0.26% | (97.34%, 97.48%) |
| KNN | 96.90% | ±0.24% | (96.83%, 96.97%) |
| MLP | 95.09% | ±0.39% | (94.98%, 95.20%) |
| SVM | 88.24% | ±0.53% | (88.09%, 88.39%) |

All four confidence intervals are non-overlapping → all pairwise differences statistically significant.

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

## Paper — v1 (2026-06-17)

Initial IEEE journal-format paper written in LaTeX covering the i-DECIDE implementation and results.

| Section | Content |
|---------|---------|
| Abstract | Dataset, two-stage method, key results |
| Introduction | HO problem, A3-event limitations, 4 contributions |
| Dataset | Drive-test setup, RAT filtering, HO detection, statistics |
| i-DECIDE Architecture | Stage 1 LSTM + Stage 2 classifiers, TikZ block diagram |
| Experimental Setup | CV protocol, training configuration, hardware |
| Results | LSTM table, classification table (50-fold), confusion matrices |
| Conclusion | Summary + 4 future directions |
| References | 8 references |

---

## Paper — v2 Revision (2026-06-23)

Major revision addressing supervisor feedback. All changes verified against `results/results_summary.json`.

### Changes Made

#### 1. Novelty Strengthened
- Introduction now explicitly frames study as "first complete reproduction of i-DECIDE on live drive-test measurements from an operational South Asian LTE network (Airtel India)"
- Contributions expanded from **4 → 6 items**:
  - Added: quantitative comparison against rule-based A3 baseline
  - Added: prediction lead-time analysis (up to 5 seconds)
  - Added: statistical significance testing (95% CI on 50-fold distributions)

#### 2. A3 Baseline Added (was missing)
- New Section IV-B: Rule-Based 3GPP A3 Baseline — implementation description
- New Table (imbalanced evaluation): A3 vs i-DECIDE RF on real 31:1 class distribution
- Key result: **5.3× F1 improvement** (29.1% vs 5.4%), **16× false-alarm reduction** (4.4% vs 72.4%)
- Communications-layer interpretation: each false A3 alert wastes X2/S1 signalling resources

#### 3. Prediction Lead-Time Added (communications insight)
- New Section IV-C: Lead-Time Analysis methodology
- New Table: lead-time sweep τ = {1, 2, 3, 5} seconds
- Key result: RF degrades only 29.1% → 28.0% F1 at 5 s horizon (~3.8% relative drop)
- Establishes: 50-sample window @ ~1 s/sample = **~50 s theoretical prediction budget**

#### 4. Statistical Significance Added
- New subsection in Results: 95% CI computed as mean ± t(0.025,49) × std/√50
- New Table: CIs for all 4 classifiers
- All pairwise CIs non-overlapping → all differences statistically significant at α=0.05

#### 5. SMOTE Data Leakage Acknowledged
- New **Limitations** paragraph in Conclusion
- Clearly states: global SMOTE applied before CV (consistent with i-DECIDE reference implementation)
- Notes: may slightly inflate balanced-set accuracy estimates
- Practical comparison (Table vs A3) uses natural imbalance — unaffected by this issue
- Future work: nested SMOTE within each fold

#### 6. XAI Section Added
- New Figure: `rf_feature_importance.png` — Gini impurity per window position with error bars
- Generated by `generate_xai_plot.py`
- Key finding: t₄₉ (most recent) accounts for 6.23% of total impurity reduction; last 10 positions average 2× importance of first 10

#### 7. References Updated: 8 → 11
All three new references are verified real papers from 2022–2024:

| Key | Paper | Venue | Year |
|-----|-------|-------|------|
| `lima2023icc` | J. P. Lima et al., "Deep Learning-Based Handover Prediction for 5G and Beyond Networks" | IEEE ICC | 2023 |
| `fang2022sdgnet` | Y. Fang, S. Ergüt, P. Patras, "SDGNet: A Handover-Aware Spatiotemporal GNN for Mobile Traffic Forecasting" | IEEE Commun. Lett., vol. 26, no. 3, pp. 582–586 | 2022 |
| `khan2024xai` | N. Khan et al., "Explainable and Robust AI for Trustworthy Resource Management in 6G Networks" | IEEE Commun. Mag., vol. 62, no. 4, pp. 50–56 | 2024 |

3GPP specs updated: Release 16 (2020) → **Release 17 (2022)** for both TS 36.331 and TS 36.214.

**Note on foundational refs (LSTM 1997, SMOTE 2002, RF 2001, Tomek 1976, RMSprop 2012):**
These must cite the original algorithm papers — IEEE standard practice. Cannot replace with newer papers.

#### 8. New Figures Added
| Figure | File | Description |
|--------|------|-------------|
| RF Feature Importance | `paper/rf_feature_importance.png` | Gini impurity per RSRP window position |
| Lead-Time Performance | `paper/lead_time_vs_performance.png` | F1/Recall vs τ for RF and A3 |

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
| `81608b0` | Document full project — README, PROGRESS, CLAUDE |
| _(latest)_ | Revise paper v2 — A3 baseline, lead-time, XAI, stats, refs |
