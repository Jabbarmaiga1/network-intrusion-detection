import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Charger le fichier DDoS
df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", low_memory=False)

# Nettoyer les noms de colonnes
df.columns = df.columns.str.strip()

# Supprimer les valeurs infinies et NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print(f"Dataset après nettoyage : {df.shape}")
print(f"\nDistribution :\n{df['Label'].value_counts()}")

# Encoder les labels (BENIGN=0, DDoS=1)
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

# Séparer features et label
X = df.drop('Label', axis=1)
y = df['Label']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nEntraînement sur {len(X_train)} samples...")

# Entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Évaluer
y_pred = model.predict(X_test)

print("\n=== Résultats ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matrice de confusion')
plt.ylabel('Réel')
plt.xlabel('Prédit')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nMatrice de confusion sauvegardée : confusion_matrix.png")

# Top 10 features importantes
importances = pd.Series(model.feature_importances_, index=X.columns)
top10 = importances.nlargest(10)

plt.figure(figsize=(10, 6))
top10.plot(kind='barh')
plt.title('Top 10 features les plus importantes')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance sauvegardée : feature_importance.png")