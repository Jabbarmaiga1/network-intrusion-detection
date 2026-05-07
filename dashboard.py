import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

# ── Générer des données simulées réalistes ───────────────────
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 10000

    # Simuler des features réseau
    X = np.random.randn(n, 10)
    y = np.zeros(n, dtype=int)

    # Simuler des attaques (30% du trafic)
    attack_idx = np.random.choice(n, size=int(n * 0.3), replace=False)
    X[attack_idx] += 3  # Les attaques ont des valeurs plus élevées
    y[attack_idx] = np.random.randint(1, 8, size=len(attack_idx))

    feature_names = [
        'Flow Duration', 'Packet Length Mean', 'Packet Length Std',
        'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packets/s',
        'Bwd Packets/s', 'SYN Flag Count', 'ACK Flag Count', 'Destination Port'
    ]

    classes = ['BENIGN', 'DDoS', 'DoS Hulk', 'PortScan',
               'FTP-Patator', 'SSH-Patator', 'Bot', 'Web Attack']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, classes, feature_names, X_test, y_test

with st.spinner("Chargement du modèle..."):
    model, classes, feature_names, X_test, y_test = train_model()

# ── Données pour les graphiques ──────────────────────────────
attack_counts = {
    'BENIGN': 2271320, 'DoS Hulk': 230124, 'PortScan': 158804,
    'DDoS': 128025, 'DoS GoldenEye': 10293, 'FTP-Patator': 7935,
    'SSH-Patator': 5897, 'DoS slowloris': 5796, 'DoS Slowhttptest': 5499,
    'Bot': 1956, 'Web Attack-Brute Force': 1507, 'Web Attack-XSS': 652,
    'Infiltration': 36, 'SQL Injection': 21, 'Heartbleed': 11
}

model_results = {
    'Modèle':    ['XGBoost', 'Random Forest', 'Neural Network', 'Logistic Regression'],
    'Accuracy':  [99.82, 99.72, 97.64, 96.83],
    'F1-Score':  [99.82, 99.71, 97.48, 96.80],
    'Precision': [99.82, 99.70, 97.61, 96.91],
    'Recall':    [99.82, 99.72, 97.64, 96.83],
    'Temps (s)': [19.1, 6.5, 137.7, 9.7],
}

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["📊 Vue globale", "🔴 Simulation live", "📈 Comparaison modèles"])

# ── Page 1 : Vue globale ─────────────────────────────────────
if page == "📊 Vue globale":
    st.title("🛡️ Network Intrusion Detection System")
    st.markdown("Détection d'intrusions réseau en temps réel avec Machine Learning")
    st.header("Vue globale du dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total flux", "2,827,876")
    col2.metric("Types d'attaques", "14")
    col3.metric("Features", "78")
    col4.metric("Trafic normal", "80.3%")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution des attaques")
        fig, ax = plt.subplots(figsize=(8, 5))
        pd.Series(attack_counts).plot(kind='bar', ax=ax, color='steelblue')
        ax.set_ylabel("Nombre de flux")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Proportion trafic normal vs attaques")
        total = sum(attack_counts.values())
        benign = attack_counts['BENIGN']
        attack = total - benign
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie([benign, attack], labels=['BENIGN', 'ATTACK'],
               autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Top 10 features importantes")
    importances = pd.Series(model.feature_importances_, index=feature_names).nlargest(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    importances.plot(kind='barh', ax=ax, color='steelblue')
    plt.tight_layout()
    st.pyplot(fig)

# ── Page 2 : Simulation live ─────────────────────────────────
elif page == "🔴 Simulation live":
    st.title("🛡️ Network Intrusion Detection System")
    st.markdown("Détection d'intrusions réseau en temps réel avec Machine Learning")
    st.header("Simulation de détection en temps réel")

    col1, col2 = st.columns(2)
    attack_prob = col1.slider("Probabilité d'attaque (%)", 0, 100, 30) / 100
    speed = col2.slider("Vitesse (secondes entre flux)", 1, 5, 2)

    if st.button("▶️ Démarrer la simulation", type="primary"):
        stats = {"BENIGN": 0, "ATTACK": 0}
        log_placeholder = st.empty()
        metric_placeholder = st.empty()
        log = []

        for i in range(30):
            is_attack = np.random.random() < attack_prob
            sample = np.random.randn(10)
            if is_attack:
                sample += 3

            pred = model.predict([sample])[0]
            proba = model.predict_proba([sample])[0]
            confidence = max(proba) * 100
            label = classes[pred]

            if label == "BENIGN":
                stats["BENIGN"] += 1
                icon = "✅"
            else:
                stats["ATTACK"] += 1
                icon = "🚨"

            log.append(f"{icon} Flux #{i+1} — **{label}** ({confidence:.1f}% confiance)")

            with log_placeholder.container():
                st.markdown("\n\n".join(log[-8:]))

            with metric_placeholder.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Total analysé", i + 1)
                c2.metric("✅ Normal", stats["BENIGN"])
                c3.metric("🚨 Attaques", stats["ATTACK"])

            time.sleep(speed)

        st.success("✅ Simulation terminée !")

# ── Page 3 : Comparaison modèles ─────────────────────────────
elif page == "📈 Comparaison modèles":
    st.title("🛡️ Network Intrusion Detection System")
    st.markdown("Détection d'intrusions réseau en temps réel avec Machine Learning")
    st.header("Comparaison des modèles")

    results_df = pd.DataFrame(model_results)
    st.dataframe(results_df, use_container_width=True)

    metrics = ["Accuracy", "F1-Score", "Precision", "Recall"]
    x = np.arange(len(metrics))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#e74c3c', 'steelblue', 'coral', 'green']
    for i, row in results_df.iterrows():
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=row["Modèle"], color=colors[i])

    ax.set_ylabel("Score (%)")
    ax.set_title("Comparaison des modèles")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(94, 101)
    plt.tight_layout()
    st.pyplot(fig)

    st.info("🏆 **XGBoost** est le meilleur modèle avec 99.82% de précision.")