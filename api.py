from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = FastAPI(
    title="Network Intrusion Detection API",
    description="API de détection d'intrusions réseau avec Machine Learning",
    version="1.0.0"
)

# ── Entraîner le modèle au démarrage ─────────────────────────
print("Initialisation du modèle...")
np.random.seed(42)
n = 10000
X = np.random.randn(n, 10)
y = np.zeros(n, dtype=int)
attack_idx = np.random.choice(n, size=int(n * 0.3), replace=False)
X[attack_idx] += 3
y[attack_idx] = np.random.randint(1, 8, size=len(attack_idx))

FEATURE_NAMES = [
    'Flow Duration', 'Packet Length Mean', 'Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packets/s',
    'Bwd Packets/s', 'SYN Flag Count', 'ACK Flag Count', 'Destination Port'
]

CLASSES = ['BENIGN', 'DDoS', 'DoS Hulk', 'PortScan',
           'FTP-Patator', 'SSH-Patator', 'Bot', 'Web Attack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Modèle prêt !")

# ── Schéma ───────────────────────────────────────────────────
class NetworkFlow(BaseModel):
    features: list[float]

# ── Routes ───────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Network Intrusion Detection API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/features", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForest", "classes": CLASSES}

@app.get("/features")
def features():
    return {"features": FEATURE_NAMES, "count": len(FEATURE_NAMES)}

@app.post("/predict")
def predict(flow: NetworkFlow):
    if len(flow.features) != len(FEATURE_NAMES):
        return {
            "error": f"Expected {len(FEATURE_NAMES)} features, got {len(flow.features)}"
        }

    features_array = np.array(flow.features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    probabilities = model.predict_proba(features_array)[0]
    confidence = float(max(probabilities)) * 100
    label = CLASSES[prediction]

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "is_attack": label != "BENIGN",
        "all_probabilities": {
            CLASSES[i]: round(float(p) * 100, 2)
            for i, p in enumerate(probabilities)
        }
    }