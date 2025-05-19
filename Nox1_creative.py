# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration de la page
st.set_page_config(page_title="🔮 NOx Prediction Dashboard", layout="wide", page_icon="🌫️")
st.title("🌫️ NOx Air Pollution Prediction")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 1. Chargement du fichier")
    uploaded_file = st.file_uploader("Importer un fichier CSV", type="csv")
    
    if not uploaded_file:
        st.warning("⛔ Veuillez charger un fichier CSV pour continuer.")
        st.stop()
    
    st.success("✅ Fichier chargé avec succès !")

# --- DONNÉES ---
@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file, na_values=["null", "NA"])
    df['date'] = pd.to_datetime(df['date'], format="%d.%m.%Y %H:%M")
    return df

with st.spinner("⏳ Chargement des données..."):
    df = load_data(uploaded_file)

# --- PRÉTRAITEMENT ---
X = df.drop(columns=['date', 'Nox_baf', 'Nox opsis'])
X = X.apply(pd.to_numeric, errors='coerce').fillna(X.mean())

# --- MODÈLES ---
model_baf = joblib.load("Nox1_modèle.pkl")
model_opsis = joblib.load("Nox_opsis_linearregression.pkl")

# --- PRÉDICTION ---
df['Nox_baf_pred'] = model_baf.predict(X)
df['Nox_opsis_pred'] = model_opsis.predict(X)

# --- ALERTES ---
def alerte(val, seuil_att, seuil_dang):
    if val >= seuil_dang:
        return "🚨 DANGER"
    elif val >= seuil_att:
        return "⚠️ ATTENTION"
    return "✅ OK"

df['Alerte_baf'] = df['Nox_baf_pred'].apply(alerte, args=(400, 500))
df['Alerte_opsis'] = df['Nox_opsis_pred'].apply(alerte, args=(350, 450))

# --- STATISTIQUES RAPIDES ---
st.markdown("### 📊 Statistiques globales")
col1, col2 = st.columns(2)
col1.metric("📈 Moyenne NOx BAF prédite", round(df['Nox_baf_pred'].mean(), 2))
col2.metric("📉 Moyenne NOx OPSIS prédite", round(df['Nox_opsis_pred'].mean(), 2))

# --- DISTRIBUTION DES ALERTES ---
st.markdown("### 🚦 Distribution des alertes")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['Alerte_baf'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue', title="BAF")
df['Alerte_opsis'].value_counts().plot(kind='bar', ax=axes[1], color='salmon', title="OPSIS")
st.pyplot(fig)

# --- VISUALISATION INTERACTIVE ---
st.markdown("### 🕵️ Analyse temporelle des NOx prédits")

target = st.radio("Sélection du modèle", ["BAF", "OPSIS"], horizontal=True)
date_range = st.date_input("📅 Plage de dates", [df.date.min().date(), df.date.max().date()])

start = pd.to_datetime(date_range[0])
end = pd.to_datetime(date_range[1])
df_filtered = df[(df['date'] >= start) & (df['date'] <= end)]

col_pred = 'Nox_baf_pred' if target == "BAF" else 'Nox_opsis_pred'
col_alert = 'Alerte_baf' if target == "BAF" else 'Alerte_opsis'
seuils = (400, 500) if target == "BAF" else (350, 450)

# Scatter plot
fig, ax = plt.subplots(figsize=(12, 5))
colors = {"✅ OK": "green", "⚠️ ATTENTION": "orange", "🚨 DANGER": "red"}
for lvl, c in colors.items():
    sub = df_filtered[df_filtered[col_alert] == lvl]
    ax.scatter(sub.date, sub[col_pred], label=lvl, color=c, s=10)
for s in seuils:
    ax.axhline(s, linestyle="--", color='gray')
ax.legend()
ax.set_title(f"📈 Évolution temporelle - {target}")
st.pyplot(fig)

# --- TABLEAU FINAL ---
st.markdown("### 📋 Tableau récapitulatif")
df['Alerte'] = df[['Alerte_opsis', 'Alerte_baf']].max(axis=1)
colonnes = ['date', 'Nox opsis', 'Nox_opsis_pred', 'Alerte_opsis',
            'Nox_baf', 'Nox_baf_pred', 'Alerte_baf', 'Alerte']
st.dataframe(df[colonnes], use_container_width=True)

# --- EXPORT CSV ---
st.markdown("### 💾 Exporter les résultats")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Télécharger les résultats complets", csv, "resultats_nox.csv", "text/csv")

