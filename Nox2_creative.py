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

# Configuration générale
st.set_page_config(page_title="NOx Monitor | Cimenterie", layout="wide")

# Animation de chargement
with st.spinner("Chargement de l'application..."):
    time.sleep(1.5)

# En-tête
st.title("🌫️ Prédiction de la pollution au NOx dans l’industrie cimentière")
st.markdown("**Par une approche chimiométrique** | 🇲🇦 ISACQ | 🧪 2025")

# --- 📖 Description du projet ---
st.markdown("### 🧾 À propos du projet")
st.write("""
Ce projet vise à **modéliser les émissions de NOx** (oxydes d'azote) dans une cimenterie à l’aide d’une **approche chimiométrique**. Les données collectées (température, concentration en gaz, etc.) sont utilisées pour entraîner des modèles prédictifs tels que la **régression linéaire**.

🔍 **Pourquoi c'est important ?**
- Le NOx contribue fortement à la pollution atmosphérique et à la formation de l’ozone troposphérique.
- Dans les cimenteries, sa production est influencée par la température du four et la combustion.

🚨 **Dangers du NOx :**
- Irritations pulmonaires
- Formation de pluies acides
- Contribution aux maladies respiratoires chroniques

👉 Ce site permet donc de **visualiser, prédire, et alerter** selon les seuils critiques de pollution.

""")

# --- 🛰️ Image météo/satellite ---
st.markdown("### 🌍 Images météo et satellite")
try:
    url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144348/pollution_nox_omi_2021_lrg.jpg"
    image_response = requests.get(url)
    img = Image.open(BytesIO(image_response.content))
    st.image(img, caption="Carte mondiale de la pollution au NOx (NASA, 2021)", use_column_width=True)
except:
    st.warning("Impossible de charger l'image satellite.")

# --- 🔄 Upload de données ---
st.sidebar.header("1️⃣ Téléchargement CSV")
uploaded_file = st.sidebar.file_uploader("💾 Votre fichier CSV", type="csv")

if not uploaded_file:
    st.sidebar.warning("Merci de charger un fichier CSV pour commencer.")
    st.stop()

df = pd.read_csv(uploaded_file, na_values=["null", "NA"])
df['date'] = pd.to_datetime(df['date'], format="%d.%m.%Y %H:%M")

# --- 🧪 Préparation des données ---
X = df.drop(columns=['date', 'Nox_baf', 'Nox opsis'])
X = X.apply(pd.to_numeric, errors='coerce').fillna(X.mean())

# --- 🤖 Chargement des modèles ---
model_baf = joblib.load("Nox1_modèle.pkl")
model_opsis = joblib.load("Nox_opsis_linearregression.pkl")

# --- 🔮 Prédictions ---
df['Nox_baf_pred'] = model_baf.predict(X)
df['Nox_opsis_pred'] = model_opsis.predict(X)

# --- 🚨 Alertes ---
def alerte(val, seuil_att, seuil_dang):
    if val >= seuil_dang: return "DANGER"
    elif val >= seuil_att: return "ATTENTION"
    return "OK"

df['Alerte_baf'] = df['Nox_baf_pred'].apply(alerte, args=(400, 500))
df['Alerte_opsis'] = df['Nox_opsis_pred'].apply(alerte, args=(350, 450))

# --- 📊 Distribution des alertes ---
st.markdown("### 📊 Distribution des alertes NOx")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['Alerte_baf'].value_counts().plot.bar(ax=axes[0], color="lightblue", title="BAF")
df['Alerte_opsis'].value_counts().plot.bar(ax=axes[1], color="salmon", title="OPSIS")
st.pyplot(fig)

# --- 📅 Visualisation temporelle ---
st.markdown("### 🕒 Visualisation temporelle interactive")
target = st.selectbox("Sélection du type de NOx", ["BAF", "OPSIS"])
date_range = st.date_input("Plage de dates", [df.date.min().date(), df.date.max().date()])
start = pd.to_datetime(date_range[0])
end = pd.to_datetime(date_range[1])

df_f = df[(df['date'] >= start) & (df['date'] <= end)]
col_pred = 'Nox_baf_pred' if target == "BAF" else 'Nox_opsis_pred'
col_alert = 'Alerte_baf' if target == "BAF" else 'Alerte_opsis'
seuils = (400, 500) if target == "BAF" else (350, 450)

fig, ax = plt.subplots(figsize=(12, 5))
colors = {"OK": "green", "ATTENTION": "orange", "DANGER": "red"}
for lvl, c in colors.items():
    sub = df_f[df_f[col_alert] == lvl]
    ax.scatter(sub.date, sub[col_pred], label=lvl, c=c, s=10)
for s in seuils:
    ax.axhline(s, linestyle="--", color="gray")
ax.set_title(f"{target} prédits dans le temps")
ax.legend()
st.pyplot(fig)

# --- 📋 Résumé final ---
st.markdown("### 🧾 Tableau récapitulatif")
if 'Alerte' not in df.columns:
    df['Alerte'] = df[['Alerte_opsis', 'Alerte_baf']].max(axis=1)

colonnes = ['date', 'Nox opsis', 'Nox_opsis_pred', 'Alerte_opsis',
            'Nox_baf', 'Nox_baf_pred', 'Alerte_baf', 'Alerte']
st.dataframe(df[colonnes])

# --- ⬇️ Téléchargement ---
st.markdown("### 💾 Télécharger les résultats")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Télécharger les résultats CSV", csv, "resultats_nox.csv", "text/csv")

