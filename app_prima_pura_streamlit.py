# app_prima_pura_streamlit.py
import os
import importlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import expit, logit
from datetime import datetime
import plotly.graph_objects as go

# ==========================================
# CONFIG
# ==========================================
COBERTURAS = [
    "Gastos_Adicionales_siniestros_monto",
    "Gastos_Medicos_RC_siniestros_monto",
    "Resp_Civil_siniestros_monto",
    "Contenidos_siniestros_monto",
]

CAT_COLS = ['año_cursado', 'estudios_area', '2_o_mas_inquilinos',
            'en_campus', 'genero', 'extintor_incendios']
NUM_COLS = ['calif_promedio', 'distancia_al_campus']
REQ_COLS = NUM_COLS + CAT_COLS
MODEL_PATH = "modelos_hurdle_tweedie.pkl"

# ==========================================
# PARCHE COMPATIBILIDAD sklearn (pickles viejos)
# ==========================================
try:
    ct_mod = importlib.import_module("sklearn.compose._column_transformer")
    if not hasattr(ct_mod, "_RemainderColsList"):
        class _RemainderColsList(list): pass
        setattr(ct_mod, "_RemainderColsList", _RemainderColsList)
except Exception:
    pass

# ==========================================
# CLASE NECESARIA PARA DESERIALIZAR EL PICKLE
# ==========================================
from sklearn.linear_model import LogisticRegression, PoissonRegressor

class HurdleFrequency:
    def __init__(self, max_iter=300, logit_C=0.05, poisson_alpha=0.1, calibrate_mean=True):
        self.max_iter = max_iter
        self.logit_C = logit_C
        self.poisson_alpha = poisson_alpha
        self.calibrate_mean = calibrate_mean
        self.logit = LogisticRegression(max_iter=max_iter, C=logit_C, solver="lbfgs")
        self.poisson_pos = PoissonRegressor(max_iter=max_iter, alpha=poisson_alpha)
        self.has_pos_ = True
        self.delta_bias_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        y_occ = (y > 0).astype(int)
        self.logit.fit(X, y_occ)
        if self.calibrate_mean:
            p_hat = self.logit.predict_proba(X)[:, 1]
            pi_real = y_occ.mean() + 1e-12
            pi_pred = p_hat.mean() + 1e-12
            self.delta_bias_ = logit(pi_real) - logit(pi_pred)
        else:
            self.delta_bias_ = 0.0
        mask = y > 0
        if mask.sum() == 0:
            self.has_pos_ = False
        else:
            self.poisson_pos.fit(X[mask], y[mask])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        logit_raw = self.logit.decision_function(X)
        p_adj = expit(logit_raw + self.delta_bias_)
        if self.has_pos_:
            m_pos = np.clip(self.poisson_pos.predict(X), 0, None)
        else:
            m_pos = np.zeros(X.shape[0])
        return p_adj * m_pos

# ==========================================
# UTILIDADES
# ==========================================
def _to_int(x):
    if isinstance(x, bool): return int(x)
    if isinstance(x, (int, float)): return int(x)
    s = str(x).strip().lower()
    return 1 if s in {"si", "sí", "true", "1", "y", "s"} else 0

def validar_columnas(df):
    faltantes = [c for c in REQ_COLS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

def normalizar_binarias(df):
    for col in ['2_o_mas_inquilinos', 'en_campus', 'extintor_incendios']:
        if col in df.columns:
            df[col] = df[col].apply(_to_int)
    return df

def predecir_prima_pura_total(df_nuevos, num_cols, cat_cols, coberturas, preprocess, modelos_freq, modelos_sev):
    validar_columnas(df_nuevos)
    df_nuevos = normalizar_binarias(df_nuevos.copy())
    X_nuevos = preprocess.transform(df_nuevos[num_cols + cat_cols])
    pred = {}
    for c in coberturas:
        freq = modelos_freq[c].predict(X_nuevos)
        sev = np.clip(modelos_sev[c].predict(X_nuevos), 0, None)
        pred[c] = freq * sev
    out = pd.DataFrame(pred, index=df_nuevos.index)
    out["prima_pura_total"] = out.sum(axis=1)
    return out

# ==========================================
# CARGA DEL MODELO
# ==========================================
@st.cache_resource(show_spinner=True)
def load_model_objects():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    raise FileNotFoundError("No se encontró el modelo local.")

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Estimador de Prima Pura", layout="centered")

# ==== ENCABEZADO ====
st.markdown("""
<div style='background:linear-gradient(90deg,#002D62,#0055A4,#0078D7);color:white;
text-align:center;padding:1.5rem 1rem;border-radius:12px;margin-bottom:28px;'>
<h1>Póliza Dormitorios</h1><p><strong>Seguros Sigma</strong></p></div>
""", unsafe_allow_html=True)

# ==== FORMULARIO ====
st.write("👤 Ingrese los datos del estudiante:")

objetos = load_model_objects()
preprocess = objetos["preprocess"]
modelos_freq = objetos["modelos_freq"]
modelos_sev = objetos["modelos_sev"]

# ==== ESTADO INICIAL ====
if "gastos" not in st.session_state:
    st.session_state["gastos"] = 20
if "utilidad" not in st.session_state:
    st.session_state["utilidad"] = 10
if "impuestos" not in st.session_state:
    st.session_state["impuestos"] = 5
if "pred_ok" not in st.session_state:
    st.session_state["pred_ok"] = False

col1, col2 = st.columns(2)
with col1:
    anio = st.selectbox("🎓 Año cursado", ["1ro año", "2do año", "3ro año", "4to año", "posgrado"], index=3)
    area = st.selectbox("🏫 Área de estudios", ["Administracion", "Humanidades", "Ciencias", "Otro"], index=1)
    calif_prom = st.number_input("📊 Calificación promedio", 0.0, 10.0, 7.01, step=0.01)
    dos_mas = st.selectbox("👥 ¿2 o más inquilinos?", ["No", "Sí"], index=0)
with col2:
    en_campus = st.selectbox("🏠 ¿Vive fuera del campus?", ["No", "Sí"], index=1)
    if en_campus == "No":
        dist_campus = 0.0
    else:
        dist_campus = st.number_input("📏 Distancia al campus (km)", 0.0, value=1.111582, step=0.000001)
    genero = st.selectbox("⚧️ Género", ["Masculino", "Femenino", "Otro", "No respuesta"], index=0)
    extintor = st.selectbox("🧯 ¿Tiene extintor?", ["No", "Sí"], index=1)

# ==== BOTÓN CALCULAR ====
if st.button("🔢 Calcular prima pura"):
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

    try:
        df_pred = predecir_prima_pura_total(nuevo, NUM_COLS, CAT_COLS, COBERTURAS,
                                            preprocess, modelos_freq, modelos_sev)
        prima_total = df_pred["prima_pura_total"].iloc[0]
        st.session_state["prima_pura_total"] = prima_total
        st.session_state["pred_ok"] = True

        st.success("✅ Predicción realizada con éxito")

        # === Mostrar tabla ===
        st.markdown("### 💵 Prima por cobertura (USD)")
        st.dataframe(df_pred[COBERTURAS].round(3))
        st.metric("💰 Prima pura total (USD)", f"{prima_total:,.4f}")

    except Exception as e:
        st.error(f"Error al predecir: {e}")
        st.session_state["pred_ok"] = False

# ==== BLOQUE PRIMA COMERCIAL ====
if st.session_state.get("pred_ok", False):
    st.markdown("### 💸 Cálculo de Prima Comercial")

    gastos = st.slider("Gastos administrativos (%)", 0, 50, 
                       value=st.session_state["gastos"], key="gastos")
    utilidad = st.slider("Utilidad (%)", 0, 30, 
                         value=st.session_state["utilidad"], key="utilidad")
    impuestos = st.slider("Impuestos (%)", 0, 20, 
                          value=st.session_state["impuestos"], key="impuestos")

    prima_pura = st.session_state["prima_pura_total"]
    factor_total = 1 + (gastos + utilidad + impuestos) / 100
    prima_comercial = prima_pura * factor_total

    st.markdown(f"""
    | Concepto | % | Valor (USD) |
    |-----------|---|-------------|
    | Prima pura | — | {prima_pura:.2f} |
    | Gastos administrativos | {gastos}% | {prima_pura*gastos/100:.2f} |
    | Utilidad | {utilidad}% | {prima_pura*utilidad/100:.2f} |
    | Impuestos | {impuestos}% | {prima_pura*impuestos/100:.2f} |
    | **Prima comercial total** | — | **{prima_comercial:.2f}** |
    """)

# ==== PIE ====
st.markdown(f"""
<div style='background-color:#f2f2f2;color:#333;font-size:0.85rem;
text-align:center;padding:0.8rem;border-radius:8px;margin-top:40px;
border-top:2px solid #0078D7;'>
© {datetime.now().year} Desarrollado con 
<a href="https://streamlit.io" target="_blank">Streamlit</a> · 💡Equipo Riskbusters - Universidad Nacional de Colombia
</div>
""", unsafe_allow_html=True)
