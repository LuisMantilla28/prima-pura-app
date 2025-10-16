import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import expit

# ==============================
# CONFIG
# ==============================
COBERTURAS = [
    "Gastos_Adicionales_siniestros_monto",
    "Gastos_Medicos_RC_siniestros_monto",
    "Resp_Civil_siniestros_monto",
    "Contenidos_siniestros_monto",
]

CAT_COLS = ['año_cursado', 'estudios_area', '2_o_mas_inquilinos',
            'en_campus', 'genero', 'extintor_incendios']
NUM_COLS = ['calif_promedio', 'distancia_al_campus']

MODEL_PATH = "/content/drive/Shareddrives/CAS_2025/Aplicativo/modelos_hurdle_tweedie.pkl"

# ==============================
# FUNCIONES AUXILIARES
# ==============================
@st.cache_resource(show_spinner=False)
def load_model_objects(path):
    objetos = joblib.load(path)
    return objetos["preprocess"], objetos["modelos_freq"], objetos["modelos_sev"]

def predecir_prima_pura_total(df_nuevos, num_cols, cat_cols, coberturas, preprocess, modelos_freq, modelos_sev):
    X_nuevos = preprocess.transform(df_nuevos[num_cols + cat_cols])
    pred_dict = {}
    for cobertura in coberturas:
        freq_pred = modelos_freq[cobertura].predict(X_nuevos)
        sev_pred  = np.clip(modelos_sev[cobertura].predict(X_nuevos), 0, None)
        pred_dict[cobertura] = freq_pred * sev_pred
    df_pred = pd.DataFrame(pred_dict, index=df_nuevos.index)
    df_pred["prima_pura_total"] = df_pred.sum(axis=1)
    return df_pred

def _to_int(x):
    if isinstance(x, bool): return int(x)
    if isinstance(x, (int, float)): return int(x)
    s = str(x).strip().lower()
    return 1 if s in {"si", "sí", "true", "1", "y", "s"} else 0

# ==============================
# INTERFAZ STREAMLIT
# ==============================
st.set_page_config(page_title="Estimador de Prima Pura", layout="centered")
st.title("🔢 Estimación de Prima Pura (Hurdle + Tweedie)")

try:
    preprocess, modelos_freq, modelos_sev = load_model_objects(MODEL_PATH)
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

st.write("### Ingrese los datos del estudiante:")

col1, col2 = st.columns(2)
with col1:
    anio = st.selectbox("Año cursado", ["1ro", "2do", "3ro", "4to", "5to"], index=3)
    area = st.selectbox("Área de estudios", ["Ingenierías", "Humanidades", "Ciencias", "Economía", "Artes", "Salud"], index=1)
    genero = st.selectbox("Género", ["Masculino", "Femenino", "Otro"], index=0)
    calif_prom = st.number_input("Calificación promedio", min_value=0.0, max_value=5.0, value=4.2, step=0.1)
with col2:
    dos_mas = st.selectbox("¿2 o más inquilinos?", ["No", "Sí"], index=0)
    en_campus = st.selectbox("¿Vive en campus?", ["No", "Sí"], index=1)
    extintor = st.selectbox("¿Tiene extintor?", ["No", "Sí"], index=1)
    dist_campus = st.number_input("Distancia al campus (km)", min_value=0.0, value=0.3, step=0.1)

if st.button("Calcular prima"):
    nuevo = pd.DataFrame({
        'año_cursado': [anio],
        'estudios_area': [area],
        '2_o_mas_inquilinos': [_to_int(dos_mas)],
        'en_campus': [_to_int(en_campus)],
        'genero': [genero],
        'extintor_incendios': [_to_int(extintor)],
        'calif_promedio': [calif_prom],
        'distancia_al_campus': [dist_campus]
    })
    df_pred = predecir_prima_pura_total(
        nuevo, NUM_COLS, CAT_COLS, COBERTURAS, preprocess, modelos_freq, modelos_sev
    )
    st.success("Predicción realizada con éxito ✅")
    st.write("**Prima por cobertura:**")
    st.dataframe(df_pred[COBERTURAS].round(4))
    st.metric("Prima pura total", f"{df_pred['prima_pura_total'].iloc[0]:,.4f}")
