import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

print("Chargement des données...")
folder = "."
all_files = [f for f in os.listdir(folder) if f.endswith(".csv")]

dfs = []
for f in all_files:
    tmp = pd.read_csv(os.path.join(folder, f), low_memory=False)
    dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True)
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# ── Simuler un dataset très déséquilibré ─────────────────────
benign = df[df['Label'] == 'BENIGN']
attacks = df[df['Label'] != 'BENIGN'].sample(frac=0.01, random_state=42)
df_imbalanced = pd.concat([benign, attacks], ignore_index=True)

print(f"\nDataset déséquilibré simulé :")
print(df_imbalanced['Label'].value_counts())

le = LabelEncoder()
df_imbalanced['Label'] = le.fit_transform(df_imbalanced['Label'])

X = df_imbalanced.drop('Label', axis=1)
y = df_imbalanced['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Réduire pour SMOTE (éviter erreur mémoire) ───────────────
benign_label = le.transform(['BENIGN'])[0]
benign_idx  = np.where(y_train == benign_label)[0]
attack_idx  = np.where(y_train != benign_label)[0]
benign_sample = np.random.choice(benign_idx, size=min(50000, len(benign_idx)), replace=False)
keep_idx = np.concatenate([benign_sample, attack_idx])
X_train_small = X_train.iloc[keep_idx].reset_index(drop=True)
y_train_small = y_train.iloc[keep_idx].reset_index(drop=True)

print(f"\nDistribution avant SMOTE : {Counter(y_train_small)}")

# ── Sans SMOTE ───────────────────────────────────────────────
print("\nEntraînement SANS SMOTE...")
model_no_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_no_smote.fit(X_train_small, y_train_small)
y_pred_no_smote = model_no_smote.predict(X_test)

print("\n=== Résultats SANS SMOTE ===")
print(classification_report(y_test, y_pred_no_smote,
      target_names=le.classes_, zero_division=0))

# ── Avec SMOTE ───────────────────────────────────────────────
print("\nApplication de SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X_train_small, y_train_small)
print(f"Distribution après SMOTE : {Counter(y_resampled)}")

print("\nEntraînement AVEC SMOTE...")
model_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_smote.fit(X_resampled, y_resampled)
y_pred_smote = model_smote.predict(X_test)

print("\n=== Résultats AVEC SMOTE ===")
print(classification_report(y_test, y_pred_smote,
      target_names=le.classes_, zero_division=0))

# ── Comparaison visuelle ─────────────────────────────────────
labels_to_show = [c for c in le.classes_ if c != 'BENIGN'][:8]
label_indices  = [list(le.classes_).index(l) for l in labels_to_show]

f1_no_smote = f1_score(y_test, y_pred_no_smote, average=None, zero_division=0)[label_indices]
f1_smote    = f1_score(y_test, y_pred_smote,    average=None, zero_division=0)[label_indices]

x = np.arange(len(labels_to_show))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width/2, f1_no_smote, width, label='Sans SMOTE', color='#e74c3c')
ax.bar(x + width/2, f1_smote,    width, label='Avec SMOTE', color='#2ecc71')
ax.set_xlabel("Type d'attaque")
ax.set_ylabel("F1-Score")
ax.set_title("Impact de SMOTE sur la détection des attaques rares")
ax.set_xticks(x)
ax.set_xticklabels(labels_to_show, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig('smote_comparison.png')
print("\nGraphique sauvegardé : smote_comparison.png")