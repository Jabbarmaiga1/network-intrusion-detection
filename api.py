from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import os

app = FastAPI(
    title="Network Intrusion Detection API",
    description="API de détection d'intrusions réseau avec Machine Learning",
    version="1.0.0"
)

# ── Charger le modèle ────────────────────────────────────────
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# ── Schéma de la requête ─────────────────────────────────────
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
    return {"status": "ok", "model": "RandomForest", "classes": list(le.classes_)}

@app.get("/features")
def features():
    return {"features": feature_names, "count": len(feature_names)}

@app.post("/predict")
def predict(flow: NetworkFlow):
    if len(flow.features) != len(feature_names):
        return {
            "error": f"Expected {len(feature_names)} features, got {len(flow.features)}"
        }

    features_array = np.array(flow.features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    probabilities = model.predict_proba(features_array)[0]
    confidence = float(max(probabilities)) * 100
    label = le.classes_[prediction]

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "is_attack": label != "BENIGN",
        "all_probabilities": {
            le.classes_[i]: round(float(p) * 100, 2)
            for i, p in enumerate(probabilities)
        }
    }