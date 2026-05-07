import pandas as pd
import numpy as np
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Entraîner le modèle ──────────────────────────────────────
print("Chargement du dataset...")
df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", low_memory=False)
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

X = df.drop('Label', axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Entraînement du modèle...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("\n" + "="*60)
print("   SYSTÈME DE DÉTECTION D'INTRUSIONS — SIMULATION LIVE")
print("="*60)
print("Monitoring du trafic réseau en cours...\n")

# ── Simulation ───────────────────────────────────────────────
benign_samples = X_test[y_test == 0].values
ddos_samples   = X_test[y_test == 1].values

stats = {"BENIGN": 0, "DDoS": 0, "total": 0}

try:
    while True:
        # Choisir aléatoirement normal ou attaque
        is_attack = np.random.random() < 0.3  # 30% chance d'attaque

        if is_attack:
            sample = ddos_samples[np.random.randint(len(ddos_samples))]
        else:
            sample = benign_samples[np.random.randint(len(benign_samples))]

        # Prédire
        prediction = model.predict([sample])[0]
        proba = model.predict_proba([sample])[0]
        confidence = max(proba) * 100
        label = le.classes_[prediction]

        stats["total"] += 1
        stats[label] += 1

        # Affichage
        timestamp = pd.Timestamp.now().strftime("%H:%M:%S")

        if label == "DDoS":
            status = "🚨 ATTAQUE DÉTECTÉE"
            bar = "█" * int(confidence / 10)
        else:
            status = "✅ Trafic normal  "
            bar = "░" * int(confidence / 10)

        print(f"[{timestamp}] {status} | Confiance: {confidence:.1f}% {bar}")
        print(f"           Paquets analysés: {stats['total']} | "
              f"Normal: {stats['BENIGN']} | "
              f"Attaques: {stats['DDoS']}")
        print()

        time.sleep(1)

except KeyboardInterrupt:
    print("\n" + "="*60)
    print("RAPPORT FINAL")
    print("="*60)
    print(f"Total analysé  : {stats['total']} flux")
    print(f"Trafic normal  : {stats['BENIGN']}")
    print(f"Attaques DDoS  : {stats['DDoS']}")
    if stats['total'] > 0:
        print(f"Taux d'attaque : {stats['DDoS']/stats['total']*100:.1f}%")
    print("="*60)