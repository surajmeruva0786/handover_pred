"""
network_logs_pipeline.py

Two-stage deep learning pipeline for handover prediction applied to network_logs_1.csv.
Architecture from: "Deep Learning-Based Handover Prediction for 5G and Beyond Networks"
(AS_JNL_IEEE_iDecide_Backup_1.pdf)

Stage 1 — LSTM RSRP regression  : predicts future signal strength samples
Stage 2 — Binary classification  : predicts whether a handover will be triggered

Authors of original architecture: Lima et al., IEEE ICC 2023 / IEEE Journal
Adaptation: applied to real-world drive-test log (network_logs_1.csv)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Reproducibility
SEED = 42
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, mean_absolute_error)
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Directories ────────────────────────────────────────
os.makedirs('results', exist_ok=True)

RESULTS = {}
LOG_LINES = []

def log(msg):
    print(msg)
    LOG_LINES.append(msg)

def section(title):
    bar = "=" * 60
    log(f"\n{bar}\n{title}\n{bar}")

# ══════════════════════════════════════════════════════
# STEP 1 — Data Loading & Preprocessing
# ══════════════════════════════════════════════════════
section("STEP 1: DATA LOADING & PREPROCESSING")
t0 = time.time()

df_raw = pd.read_csv('network_logs_1.csv')
df_raw.columns = df_raw.columns.str.strip()
log(f"Raw dataset  : {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")
nt = df_raw['NetworkType'].str.strip().value_counts()
log(f"Network types:\n{nt.to_string()}")

RESULTS['raw_rows'] = int(df_raw.shape[0])
RESULTS['network_types'] = {k: int(v) for k, v in nt.items()}

# Keep only LTE (4G) rows
df = df_raw[df_raw['NetworkType'].str.strip() == 'LTE (4G)'].copy()
log(f"\nAfter LTE-4G filter : {len(df)} rows")

def parse_signal(val):
    """Strip unit suffixes and convert to float."""
    if pd.isna(val):
        return np.nan
    s = str(val).replace(' dBm', '').replace(' dB', '').strip()
    try:
        return float(s)
    except ValueError:
        return np.nan

for col in ['RSRP', 'RSRQ', 'SINR']:
    df[col] = df[col].apply(parse_signal)

df['PCI'] = pd.to_numeric(df['PCI'], errors='coerce')
df = df.dropna(subset=['RSRP', 'PCI']).copy()
df['PCI'] = df['PCI'].astype(int)

# Sort ascending by timestamp (raw data is newest-first)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.sort_values('Timestamp', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)

log(f"After cleaning (valid RSRP + PCI) : {len(df)} rows")
log(f"Timespan : {df['Timestamp'].min()} → {df['Timestamp'].max()}")
log(f"RSRP range: {df['RSRP'].min():.1f} to {df['RSRP'].max():.1f} dBm")
log(f"Unique PCI values: {sorted(df['PCI'].unique().tolist())}")

# Detect handovers: PCI change between consecutive samples
df['ho_trig'] = 0
for i in range(1, len(df)):
    if df.loc[i, 'PCI'] != df.loc[i - 1, 'PCI']:
        df.loc[i, 'ho_trig'] = 1

n_ho = int(df['ho_trig'].sum())
n_total = len(df)
log(f"\nHandover events (ho_trig=1) : {n_ho}")
log(f"No-HO samples (ho_trig=0)  : {n_total - n_ho}")
log(f"HO rate                    : {n_ho / n_total * 100:.2f}%")

RESULTS['lte_rows'] = n_total
RESULTS['handover_events'] = n_ho
RESULTS['no_ho_events'] = n_total - n_ho
RESULTS['pci_values'] = sorted(df['PCI'].unique().tolist())
RESULTS['rsrp_min'] = float(df['RSRP'].min())
RESULTS['rsrp_max'] = float(df['RSRP'].max())

df.to_csv('results/preprocessed_data.csv', index=False)
log("Saved → results/preprocessed_data.csv")
log(f"[Step 1 done in {time.time()-t0:.1f}s]")

# ══════════════════════════════════════════════════════
# STEP 2 — LSTM RSRP Prediction  (Stage 1)
# ══════════════════════════════════════════════════════
section("STEP 2: LSTM RSRP PREDICTION (STAGE 1)")
t0 = time.time()

LOOKBACK   = 100   # samples used to predict the next RSRP
EPOCHS     = 100
BATCH_SIZE = 128
TRAIN_RATIO = 0.80

rsrp    = df['RSRP'].values.reshape(-1, 1)
ho_trig = df['ho_trig'].values

scaler = MinMaxScaler(feature_range=(0, 1))
rsrp_norm = scaler.fit_transform(rsrp)

n_train = int(len(rsrp_norm) * TRAIN_RATIO)
rsrp_train = rsrp_norm[:n_train]
rsrp_test  = rsrp_norm[n_train:]
log(f"Train / Test split : {n_train} / {len(rsrp_norm) - n_train} samples")

# Build sliding-window sequences for LSTM training
X_train, y_train = [], []
for i in range(LOOKBACK, len(rsrp_train)):
    X_train.append(rsrp_train[i - LOOKBACK:i, 0])
    y_train.append(rsrp_train[i, 0])
X_train = np.array(X_train).reshape(-1, LOOKBACK, 1)
y_train = np.array(y_train)
log(f"LSTM train shape : X{X_train.shape}  y{y_train.shape}")

# ── LSTM Architecture (from paper) ────────────────────
#   Layer 1 : LSTM(120)  + Dropout(0.3)
#   Layer 2 : LSTM(50)   + Dropout(0.3)
#   Layer 3 : LSTM(50)   + Dropout(0.3)
#   Layer 4 : LSTM(50)   + Dropout(0.3)
#   Output  : Dense(1, linear)
#   Optimizer: RMSprop | Loss: MSE
model = Sequential([
    LSTM(120, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.3),
    LSTM(50, return_sequences=True),
    Dropout(0.3),
    LSTM(50, return_sequences=True),
    Dropout(0.3),
    LSTM(50),
    Dropout(0.3),
    Dense(1, activation='linear'),
])
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

log("\nLSTM Model Summary:")
model.summary(print_fn=log)

early_stop = EarlyStopping(monitor='val_loss', patience=10,
                            restore_best_weights=True, verbose=1)
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1,
)
actual_epochs = len(history.history['loss'])
log(f"\nTraining stopped at epoch {actual_epochs}/{EPOCHS}")
log(f"Best val_loss : {min(history.history['val_loss']):.6f}")

# ── Test set evaluation ────────────────────────────────
inputs_test = rsrp_norm[n_train - LOOKBACK:]
X_test = []
for i in range(LOOKBACK, len(inputs_test)):
    X_test.append(inputs_test[i - LOOKBACK:i, 0])
X_test = np.array(X_test).reshape(-1, LOOKBACK, 1)

pred_test_norm = model.predict(X_test, verbose=0)
pred_test      = scaler.inverse_transform(pred_test_norm)
real_test      = rsrp[n_train:]

mae  = float(mean_absolute_error(real_test, pred_test))
rmse = float(np.sqrt(np.mean((real_test - pred_test) ** 2)))
log(f"\nLSTM Test MAE  : {mae:.4f} dBm")
log(f"LSTM Test RMSE : {rmse:.4f} dBm")

RESULTS['lstm_mae']          = mae
RESULTS['lstm_rmse']         = rmse
RESULTS['lstm_epochs']       = actual_epochs
RESULTS['lstm_train_samples']= int(X_train.shape[0])
RESULTS['lstm_test_samples'] = int(X_test.shape[0])

# Save model
model.save('results/lstm_model.keras')
log("Saved → results/lstm_model.keras")

# Training loss plot
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('LSTM Training Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/lstm_training_loss.png', dpi=150)
plt.close()

# RSRP prediction plot
plt.figure(figsize=(12, 4))
plt.plot(real_test, color='red',  label='Real RSRP', alpha=0.7)
plt.plot(pred_test, color='blue', label='LSTM Prediction', alpha=0.7)
plt.title(f'RSRP Prediction — MAE = {mae:.2f} dBm')
plt.xlabel('Time (samples)')
plt.ylabel('RSRP (dBm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/lstm_rsrp_prediction.png', dpi=150)
plt.close()
log("Saved → results/lstm_training_loss.png, results/lstm_rsrp_prediction.png")
log(f"[Step 2 done in {time.time()-t0:.1f}s]")

# ══════════════════════════════════════════════════════
# STEP 3 — Build Classification Feature Dataset
# ══════════════════════════════════════════════════════
section("STEP 3: BUILDING CLASSIFICATION DATASET")
t0 = time.time()

# Predict RSRP for the FULL time series
X_all = []
for i in range(LOOKBACK, len(rsrp_norm)):
    X_all.append(rsrp_norm[i - LOOKBACK:i, 0])
X_all = np.array(X_all).reshape(-1, LOOKBACK, 1)

pred_all_norm = model.predict(X_all, verbose=0)
pred_all      = scaler.inverse_transform(pred_all_norm)   # shape (N-100, 1)
ho_all        = ho_trig[LOOKBACK:]                         # aligned labels

log(f"Full-set predictions shape : {pred_all.shape}")

# Sliding 50-sample window of LSTM predictions → one feature vector per window
WINDOW = 50
X_cls, y_cls = [], []
for i in range(WINDOW, len(pred_all)):
    X_cls.append(pred_all[i - WINDOW:i, 0])
    y_cls.append(int(ho_all[i - 1]))

X_cls = np.array(X_cls)
y_cls = np.array(y_cls)
log(f"Classification dataset : X{X_cls.shape}  y{y_cls.shape}")
log(f"Class distribution (raw): {Counter(y_cls)}")

cls_df = pd.DataFrame(X_cls)
cls_df['label'] = y_cls
cls_df.to_csv('results/classification_base.csv', index=False)
log("Saved → results/classification_base.csv")
log(f"[Step 3 done in {time.time()-t0:.1f}s]")

# ══════════════════════════════════════════════════════
# STEP 4 — Class Balancing (Tomek Links + SMOTE)
# ══════════════════════════════════════════════════════
section("STEP 4: CLASS BALANCING (TOMEK LINKS + SMOTE)")
t0 = time.time()

counts = Counter(y_cls)
minority_count = min(counts.values())
majority_count = max(counts.values())

if minority_count < 6:
    # Skip resampling if too few minority samples; just use raw
    log(f"WARNING: Too few minority samples ({minority_count}). Skipping resampling.")
    X_bal, y_bal = X_cls, y_cls
else:
    # Tomek Links undersampling
    tomek = TomekLinks()
    X_tomek, y_tomek = tomek.fit_resample(X_cls, y_cls)
    log(f"After Tomek Links : {Counter(y_tomek)}")

    # Determine safe k_neighbors for SMOTE
    min_after_tomek = min(Counter(y_tomek).values())
    k_neighbors = min(5, min_after_tomek - 1)
    if k_neighbors < 1:
        k_neighbors = 1

    smote = SMOTE(k_neighbors=k_neighbors, random_state=SEED)
    X_bal, y_bal = smote.fit_resample(X_tomek, y_tomek)
    log(f"After SMOTE       : {Counter(y_bal)}")

RESULTS['cls_before_balance'] = {str(k): int(v) for k, v in Counter(y_cls).items()}
RESULTS['cls_after_balance']  = {str(k): int(v) for k, v in Counter(y_bal).items()}

std_scaler  = StandardScaler()
X_bal_scaled = std_scaler.fit_transform(X_bal)

bal_df = pd.DataFrame(X_bal_scaled)
bal_df['label'] = y_bal
bal_df.to_csv('results/balanced_classification_base.csv', index=False)
log("Saved → results/balanced_classification_base.csv")
log(f"[Step 4 done in {time.time()-t0:.1f}s]")

# ══════════════════════════════════════════════════════
# STEP 5 — Binary Classification  (Stage 2)
# ══════════════════════════════════════════════════════
section("STEP 5: BINARY CLASSIFICATION (STAGE 2)")

def run_classifier(name, clf, X, y, n_repeats=10, n_splits=5):
    accs, f1s, precs, recs, mats = [], [], [], [], []
    for seed in range(n_repeats):
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_idx, te_idx in kfold.split(X, y):
            clf.fit(X[tr_idx], y[tr_idx])
            preds = clf.predict(X[te_idx])
            accs.append(accuracy_score(y[te_idx], preds))
            f1s.append(f1_score(y[te_idx], preds, zero_division=0))
            precs.append(precision_score(y[te_idx], preds, zero_division=0))
            recs.append(recall_score(y[te_idx], preds, zero_division=0))
            mats.append(confusion_matrix(y[te_idx], preds))

    res = {
        'accuracy_mean' : float(np.mean(accs)),
        'accuracy_std'  : float(np.std(accs)),
        'f1_mean'       : float(np.mean(f1s)),
        'f1_std'        : float(np.std(f1s)),
        'precision_mean': float(np.mean(precs)),
        'recall_mean'   : float(np.mean(recs)),
        'confusion_matrix': np.mean(mats, axis=0).tolist(),
    }
    log(f"\n[{name}]")
    log(f"  Accuracy  : {res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f}")
    log(f"  F1-Score  : {res['f1_mean']:.4f} ± {res['f1_std']:.4f}")
    log(f"  Precision : {res['precision_mean']:.4f}")
    log(f"  Recall    : {res['recall_mean']:.4f}")
    log(f"  Confusion matrix (mean):\n  {np.array(res['confusion_matrix'])}")
    return res

classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    'SVM (RBF)'   : SVC(kernel='rbf', C=500.0, random_state=SEED),
    'MLP'         : MLPClassifier(solver='lbfgs', max_iter=2000,
                                   hidden_layer_sizes=[120, 120], random_state=SEED),
    'KNN'         : KNeighborsClassifier(n_neighbors=2, p=1,
                                          algorithm='ball_tree', leaf_size=200),
}

cls_results = {}
for clf_name, clf in classifiers.items():
    t0 = time.time()
    log(f"\nRunning {clf_name} (10×5-fold CV) ...")
    cls_results[clf_name] = run_classifier(clf_name, clf, X_bal_scaled, y_bal)
    log(f"  [{clf_name} done in {time.time()-t0:.1f}s]")

RESULTS['classification'] = cls_results

# ══════════════════════════════════════════════════════
# STEP 6 — Summary & Save
# ══════════════════════════════════════════════════════
section("STEP 6: RESULTS SUMMARY")

with open('results/results_summary.json', 'w') as f:
    json.dump(RESULTS, f, indent=2)
log("Saved → results/results_summary.json")

with open('results/run_log.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG_LINES))
log("Saved → results/run_log.txt")

log("\n" + "─" * 60)
log("FINAL RESULTS")
log("─" * 60)
log(f"Dataset         : {RESULTS['raw_rows']} raw → {RESULTS['lte_rows']} LTE-4G samples")
log(f"Handover events : {RESULTS['handover_events']} ({RESULTS['handover_events']/RESULTS['lte_rows']*100:.2f}%)")
log(f"LSTM MAE        : {RESULTS['lstm_mae']:.4f} dBm")
log(f"LSTM RMSE       : {RESULTS['lstm_rmse']:.4f} dBm")
log(f"LSTM Epochs     : {RESULTS['lstm_epochs']}")
log("\nClassifier Results:")
log(f"{'Classifier':<22} {'Accuracy':>10} {'F1':>8} {'Precision':>11} {'Recall':>8}")
log("─" * 62)
for cname, res in cls_results.items():
    log(f"{cname:<22} {res['accuracy_mean']:>10.4f} {res['f1_mean']:>8.4f}"
        f" {res['precision_mean']:>11.4f} {res['recall_mean']:>8.4f}")
log("─" * 62)
log("\nPipeline complete.")
