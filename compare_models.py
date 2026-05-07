import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

models = {
    "Random Forest":       (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), False),
    "XGBoost":             (XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0), False),
    "Neural Network":      (MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50, random_state=42), True),
    "Logistic Regression": (LogisticRegression(max_iter=200, random_state=42, n_jobs=-1), True),
}

# ── Charger les données ──────────────────────────────────────
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

# Échantillonner pour aller plus vite
df = df.sample(n=100000, random_state=42)
print(f"Dataset échantillonné : {df.shape}")

le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

X = df.drop('Label', axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normaliser pour Logistic Regression et MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Modèles à comparer ───────────────────────────────────────
models = {
    "Random Forest":       (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), False),
    "XGBoost":             (XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0), False),
    "Neural Network":      (MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50, random_state=42), True),
    "Logistic Regression": (LogisticRegression(max_iter=200, random_state=42, n_jobs=-1), True),
}

results = []

for name, (model, use_scaled) in models.items():
    print(f"\nEntraînement : {name}...")
    X_tr = X_train_scaled if use_scaled else X_train
    X_te = X_test_scaled  if use_scaled else X_test

    start = time.time()
    model.fit(X_tr, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_te)

    results.append({
        "Modèle":    name,
        "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "F1-Score":  round(f1_score(y_test, y_pred, average='weighted') * 100, 2),
        "Precision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
        "Recall":    round(recall_score(y_test, y_pred, average='weighted') * 100, 2),
        "Temps (s)": round(train_time, 2),
    })
    print(f"  ✅ Accuracy: {results[-1]['Accuracy']}% | F1: {results[-1]['F1-Score']}% | Temps: {train_time:.1f}s")

# ── Tableau comparatif ───────────────────────────────────────
results_df = pd.DataFrame(results)
print("\n=== COMPARAISON DES MODÈLES ===")
print(results_df.to_string(index=False))

# ── Graphique ────────────────────────────────────────────────
metrics = ["Accuracy", "F1-Score", "Precision", "Recall"]
x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
for i, row in results_df.iterrows():
    vals = [row[m] for m in metrics]
    ax.bar(x + i * width, vals, width, label=row["Modèle"])

ax.set_xlabel("Métriques")
ax.set_ylabel("Score (%)")
ax.set_title("Comparaison des modèles de détection d'intrusions")
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig("model_comparison.png")
print("\nGraphique sauvegardé : model_comparison.png")