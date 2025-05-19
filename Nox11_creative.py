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
from PIL import Image
import time
import requests
from io import BytesIO
import plotly.express as px

# --- Page Config ---
st.set_page_config(page_title="NOx Monitor | Cimenterie", layout="wide", page_icon="🌫️")

# --- CSS Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        color: #333;
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
    }
    .title-style {
        font-size: 3em;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
    }
    .subheader-style {
        font-size: 1.3em;
        color: #7f8c8d;
        margin-bottom: 1rem;
    }
    .alert-box {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Loading Animation ---
with st.spinner("Chargement de l'application en cours..."):
    time.sleep(1.5)

# --- Header ---
st.markdown('<div class="title-style">🌫️ NOx Monitor Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader-style">Modélisation des NOx par approche chimiométrique dans l’industrie cimentière</div>', unsafe_allow_html=True)

# --- Project Description ---
st.markdown("## 🧾 Projet")
st.markdown("""
Ce projet a pour objectif de **modéliser et surveiller les émissions de NOx** dans une cimenterie grâce à une approche chimiométrique.

### 🚨 Pourquoi surveiller le NOx ?
- Le NOx (oxydes d'azote) est un gaz **hautement toxique** produit par les processus à haute température comme dans les cimenteries.
- Il provoque des **irritations respiratoires**, participe à la formation de **pluies acides** et de **smog**.

### 🎯 Objectifs :
- Prédire les niveaux de NOx à partir de données opérationnelles.
- Générer des alertes visuelles pour la gestion environnementale.
""")

# --- Satellite Image ---
st.markdown("## 🌍 Données Satellites")
try:
    url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144348/pollution_nox_omi_2021_lrg.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img, caption="Pollution mondiale au NOx (NASA, 2021)", use_column_width=True)
except:
    st.warning("Impossible de charger l'image.")

# --- Upload CSV ---
st.sidebar.header("1. Téléchargement de Données")
uploaded_file = st.sidebar.file_uploader("Choisir un fichier CSV", type="csv")
if not uploaded_file:
    st.sidebar.info("Veuillez charger un fichier pour commencer.")
    st.stop()

# --- Load and Preprocess Data ---
df = pd.read_csv(uploaded_file, na_values=["null", "NA"])
df['date'] = pd.to_datetime(df['date'], format="%d.%m.%Y %H:%M")
X = df.drop(columns=['date', 'Nox_baf', 'Nox opsis'], errors='ignore')
X = X.apply(pd.to_numeric, errors='coerce').fillna(X.mean())

# --- Load Models ---
model_baf = joblib.load("Nox1_modèle.pkl")
model_opsis = joblib.load("Nox_opsis_linearregression.pkl")

# --- Predictions ---
df['Nox_baf_pred'] = model_baf.predict(X)
df['Nox_opsis_pred'] = model_opsis.predict(X)

# --- Alert Levels ---
def get_alert(value, att, dang):
    if value >= dang: return "DANGER"
    elif value >= att: return "ATTENTION"
    return "OK"

df['Alerte_baf'] = df['Nox_baf_pred'].apply(get_alert, args=(400, 500))
df['Alerte_opsis'] = df['Nox_opsis_pred'].apply(get_alert, args=(350, 450))

# --- Graphs ---
st.markdown("## 📊 Visualisation des Prédictions")
target = st.selectbox("Choisir le capteur", ["BAF", "OPSIS"])
date_range = st.date_input("Plage de dates", [df.date.min().date(), df.date.max().date()])
start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
df_filtered = df[(df['date'] >= start) & (df['date'] <= end)]

pred_col = 'Nox_baf_pred' if target == "BAF" else 'Nox_opsis_pred'
alert_col = 'Alerte_baf' if target == "BAF" else 'Alerte_opsis'

fig = px.scatter(df_filtered, x="date", y=pred_col, color=alert_col,
                 color_discrete_map={"OK": "green", "ATTENTION": "orange", "DANGER": "red"},
                 title=f"NOx {target} - Prédictions dans le Temps")
st.plotly_chart(fig, use_container_width=True)

# --- Final Table ---
st.markdown("## 🧾 Récapitulatif des Données")
st.dataframe(df[['date', 'Nox opsis', 'Nox_opsis_pred', 'Alerte_opsis', 'Nox_baf', 'Nox_baf_pred', 'Alerte_baf']])

# --- Download Button ---
st.markdown("## 💾 Téléchargement")
st.download_button("Télécharger les résultats", df.to_csv(index=False).encode("utf-8"), "nox_resultats.csv", "text/csv")

