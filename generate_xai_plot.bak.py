"""Generate RF feature importance XAI plot for the handover prediction paper."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load balanced data
df = pd.read_csv('results/balanced_classification_base.csv')
feature_cols = [str(i) for i in range(50)]
X = df[feature_cols].values
y = df['label'].values

# Data already scaled in pipeline — use as-is
X_scaled = X

# Train RF with same hyperparams as pipeline
print("Training RF for XAI...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_scaled, y)
print("Done.")

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
indices = np.arange(len(importances))  # 0..49

# Smooth trend via moving average
window = 5
smooth = np.convolve(importances, np.ones(window)/window, mode='same')

# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 3.4))

ax.bar(indices, importances, color='steelblue', alpha=0.55,
       yerr=std, error_kw=dict(elinewidth=0.5, alpha=0.4, capsize=0),
       label='Feature importance (±1 std)')
ax.plot(indices, smooth, color='crimson', linewidth=1.8,
        label=f'{window}-sample moving avg')

ax.set_xlabel('RSRP Window Position (0 = oldest, 49 = most recent)', fontsize=10)
ax.set_ylabel('Mean Decrease in Gini Impurity', fontsize=10)
ax.set_xlim(-1, 50)
ax.set_xticks([0, 9, 19, 29, 39, 49])
ax.set_xticklabels(['$t_0$', '$t_9$', '$t_{19}$', '$t_{29}$', '$t_{39}$', '$t_{49}$'])
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
ax.legend(fontsize=8, loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('results/rf_feature_importance.png', dpi=200, bbox_inches='tight')
plt.savefig('paper/rf_feature_importance.png',   dpi=200, bbox_inches='tight')
print("Saved: results/rf_feature_importance.png  paper/rf_feature_importance.png")

# Print summary stats for paper text
top5 = np.argsort(importances)[::-1][:5]
print("\nTop-5 most important features:")
for rank, idx in enumerate(top5, 1):
    print(f"  {rank}. t_{idx}  importance={importances[idx]:.5f}")

print(f"\nLast-10 positions (t40-t49) mean importance: {importances[40:].mean():.5f}")
print(f"First-10 positions (t0-t9)  mean importance: {importances[:10].mean():.5f}")
print(f"Ratio last/first: {importances[40:].mean()/importances[:10].mean():.2f}x")
