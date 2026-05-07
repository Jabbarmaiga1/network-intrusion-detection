import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ── Charger tous les fichiers CSV ────────────────────────────
print("Chargement de tous les fichiers CSV...")
folder = "."
all_files = [f for f in os.listdir(folder) if f.endswith(".csv")]

dfs = []
for f in all_files:
    print(f"  Lecture de {f}...")
    tmp = pd.read_csv(os.path.join(folder, f), low_memory=False)
    dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True)
print(f"\nDataset complet : {df.shape}")

# ── Nettoyage ────────────────────────────────────────────────
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print(f"\nDistribution des attaques :")
print(df['Label'].value_counts())

# ── Encodage ─────────────────────────────────────────────────
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

X = df.drop('Label', axis=1)
y = df['Label']

# ── Train/Test split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nEntraînement sur {len(X_train)} samples...")
print("Patience, ça peut prendre 5-10 minutes...\n")

# ── Modèle ───────────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── Évaluation ───────────────────────────────────────────────
y_pred = model.predict(X_test)

print("=== Résultats Multiclasse ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Matrice de confusion ─────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matrice de confusion — Multiclasse')
plt.ylabel('Réel')
plt.xlabel('Prédit')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix_multiclass.png')
print("Matrice sauvegardée : confusion_matrix_multiclass.png")

# ── Feature importance ───────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=X.columns)
top10 = importances.nlargest(10)

plt.figure(figsize=(10, 6))
top10.plot(kind='barh', color='steelblue')
plt.title('Top 10 features les plus importantes')
plt.tight_layout()
plt.savefig('feature_importance_multiclass.png')
print("Feature importance sauvegardée : feature_importance_multiclass.png")

# Sauvegarder le modèle et l'encodeur pour les niveaux suivants
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("\nModèle sauvegardé : model.pkl")