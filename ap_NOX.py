import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="NOx Prediction", layout="wide")
st.title("Prédiction de la pollution NOx")

# Sidebar
st.sidebar.header("1. Téléchargement CSV")
uploaded_file = st.sidebar.file_uploader("Votre fichier CSV", type="csv")

if not uploaded_file:
    st.sidebar.warning("Merci de charger un fichier CSV")
    st.stop()

# 2. Chargement et parsing
df = pd.read_csv(uploaded_file, na_values=["null","NA"])
df['date'] = pd.to_datetime(df['date'], format="%d.%m.%Y %H:%M")

# 3. Préparation des features
X = df.drop(columns=['date','Nox_baf','Nox opsis'])
X = X.apply(pd.to_numeric, errors='coerce').fillna(X.mean())

# 4. Chargement des modèles
model_baf = joblib.load("Nox1_modele.pkl")
model_opsis = joblib.load("Nox_opsis_linearregression.pkl")

# 5. Prédiction
df['Nox_baf_pred']   = model_baf.predict(X)
df['Nox_opsis_pred'] = model_opsis.predict(X)

# 6. Alertes
def alerte(val, seuil_att, seuil_dang):
    if val >= seuil_dang: return "DANGER"
    if val >= seuil_att: return "ATTENTION"
    return "OK"

df['Alerte_baf']   = df['Nox_baf_pred'].apply(alerte, args=(400,500))
df['Alerte_opsis'] = df['Nox_opsis_pred'].apply(alerte, args=(350,450))

# 7. Distribution des alertes
st.subheader("Distribution des alertes NOx")
fig, axes = plt.subplots(1,2, figsize=(12,4))
df['Alerte_baf'].value_counts().plot.bar(ax=axes[0], title="BAF")
df['Alerte_opsis'].value_counts().plot.bar(ax=axes[1], title="OPSIS")
st.pyplot(fig)

# 8. Filtre date & scatter
st.subheader("Visualisation interactive")
target = st.selectbox("Type NOx", ["BAF","OPSIS"])
start, end = st.date_input("Plage de dates", [df.date.min(), df.date.max()])
df_f = df[(df.date>=start)&(df.date<=end)]

col_pred  = 'Nox_baf_pred'   if target=="BAF"   else 'Nox_opsis_pred'
col_alert = 'Alerte_baf'     if target=="BAF"   else 'Alerte_opsis'
seuils    = (400,500)        if target=="BAF"   else (350,450)

fig, ax = plt.subplots(figsize=(12,5))
colors = {"OK":"green","ATTENTION":"orange","DANGER":"red"}
for lvl,c in colors.items():
    sub = df_f[df_f[col_alert]==lvl]
    ax.scatter(sub.date, sub[col_pred], label=lvl, c=c, s=10)
for s in seuils:
    ax.axhline(s, linestyle="--")
ax.legend(); ax.set_title(f"{target} prédits")
st.pyplot(fig)

# 9. Téléchargement CSV des résultats
st.subheader("Télécharger les résultats")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Télécharger", csv, "resultats_nox.csv", "text/csv")
