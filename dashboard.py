import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

# ── Charger le modèle ────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, le, features

@st.cache_data
def load_data():
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
    return df

# ── Interface ────────────────────────────────────────────────
st.title("🛡️ Network Intrusion Detection System")
st.markdown("Détection d'intrusions réseau en temps réel avec Machine Learning")

with st.spinner("Chargement du modèle et des données..."):
    model, le, features = load_model()
    df = load_data()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["📊 Vue globale", "🔴 Simulation live", "📈 Comparaison modèles"])

# ── Page 1 : Vue globale ─────────────────────────────────────
if page == "📊 Vue globale":
    st.header("Vue globale du dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total flux", f"{len(df):,}")
    col2.metric("Types d'attaques", df['Label'].nunique() - 1)
    col3.metric("Features", len(features))
    benign_pct = (df['Label'] == 'BENIGN').sum() / len(df) * 100
    col4.metric("Trafic normal", f"{benign_pct:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution des attaques")
        label_counts = df['Label'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        label_counts.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_xlabel("")
        ax.set_ylabel("Nombre de flux")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Proportion trafic normal vs attaques")
        attack_counts = df['Label'].apply(lambda x: 'BENIGN' if x == 'BENIGN' else 'ATTACK').value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(attack_counts, labels=attack_counts.index,
               autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Top 10 features importantes")
    importances = pd.Series(model.feature_importances_, index=features).nlargest(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    importances.plot(kind='barh', ax=ax, color='steelblue')
    plt.tight_layout()
    st.pyplot(fig)

# ── Page 2 : Simulation live ─────────────────────────────────
elif page == "🔴 Simulation live":
    st.header("Simulation de détection en temps réel")

    le2 = LabelEncoder()
    df_copy = df.copy()
    df_copy['Label'] = le2.fit_transform(df_copy['Label'])
    X = df_copy[features]
    y = df_copy['Label']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    benign_samples = X_test[y_test == le2.transform(['BENIGN'])[0]].values
    attack_samples = X_test[y_test != le2.transform(['BENIGN'])[0]].values

    col1, col2 = st.columns(2)
    attack_prob = col1.slider("Probabilité d'attaque (%)", 0, 100, 30) / 100
    speed = col2.slider("Vitesse (secondes entre flux)", 1, 5, 2)

    if st.button("▶️ Démarrer la simulation", type="primary"):
        stats = {"BENIGN": 0, "ATTACK": 0}
        log_placeholder = st.empty()
        chart_placeholder = st.empty()
        metric_placeholder = st.empty()
        log = []

        for i in range(30):
            is_attack = np.random.random() < attack_prob
            sample = attack_samples[np.random.randint(len(attack_samples))] if is_attack \
                     else benign_samples[np.random.randint(len(benign_samples))]

            pred = model.predict([sample])[0]
            proba = model.predict_proba([sample])[0]
            confidence = max(proba) * 100
            label = le.classes_[pred]

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
    st.header("Comparaison des modèles")

    results = {
        "Modèle":    ["Random Forest", "Neural Network", "Logistic Regression"],
        "Accuracy":  [99.72, 97.64, 96.83],
        "F1-Score":  [99.71, 97.48, 96.80],
        "Precision": [99.70, 97.61, 96.91],
        "Recall":    [99.72, 97.64, 96.83],
        "Temps (s)": [5.3, 49.6, 7.3],
    }
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)

    metrics = ["Accuracy", "F1-Score", "Precision", "Recall"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['steelblue', 'coral', 'green']
    for i, row in results_df.iterrows():
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=row["Modèle"], color=colors[i])

    ax.set_ylabel("Score (%)")
    ax.set_title("Comparaison des modèles")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(94, 101)
    plt.tight_layout()
    st.pyplot(fig)

    st.info("🏆 **Random Forest** est le meilleur modèle : plus précis et 10x plus rapide que le réseau de neurones.")