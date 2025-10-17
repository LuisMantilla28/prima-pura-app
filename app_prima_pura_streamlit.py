# app_prima_pura_streamlit.py
import os
import importlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import warnings
from scipy.special import expit, logit

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

# Si el .pkl está en el mismo repo:
MODEL_PATH = "modelos_hurdle_tweedie.pkl"

# ==========================================
# PARCHE COMPATIBILIDAD sklearn (pickles viejos)
# ==========================================
try:
    ct_mod = importlib.import_module("sklearn.compose._column_transformer")
    if not hasattr(ct_mod, "_RemainderColsList"):
        class _RemainderColsList(list):
            """Shim para compatibilidad de pickles antiguos de ColumnTransformer."""
            pass
        setattr(ct_mod, "_RemainderColsList", _RemainderColsList)
except Exception:
    # Si falla el import, continuamos; el load puede funcionar igual
    pass

# ==========================================
# CLASE NECESARIA PARA DESERIALIZAR EL PICKLE
# (debe estar definida ANTES de joblib.load)
# ==========================================
from sklearn.linear_model import LogisticRegression, PoissonRegressor

class HurdleFrequency:
    """Frecuencia HURDLE: Logit (ocurrencia) + Poisson (cond. en N>0) con calibración de media."""
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

def validar_columnas(df: pd.DataFrame):
    faltantes = [c for c in REQ_COLS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

def normalizar_binarias(df: pd.DataFrame):
    for col in ['2_o_mas_inquilinos', 'en_campus', 'extintor_incendios']:
        if col in df.columns:
            df[col] = df[col].apply(_to_int)
    return df

def predecir_prima_pura_total(df_nuevos, num_cols, cat_cols, coberturas, preprocess, modelos_freq, modelos_sev):
    validar_columnas(df_nuevos)
    df_nuevos = normalizar_binarias(df_nuevos.copy())
    # IMPORTANTE: el preprocess fue entrenado con estas columnas en este orden
    X_nuevos = preprocess.transform(df_nuevos[num_cols + cat_cols])

    pred = {}
    for c in coberturas:
        try:
            freq = modelos_freq[c].predict(X_nuevos)
            sev  = np.clip(modelos_sev[c].predict(X_nuevos), 0, None)
        except Exception as e:
            raise RuntimeError(f"Error prediciendo cobertura '{c}': {e}")
        pred[c] = freq * sev
    out = pd.DataFrame(pred, index=df_nuevos.index)
    out["prima_pura_total"] = out.sum(axis=1)
    return out

# ==========================================
# CARGA ROBUSTA DEL MODELO (local o remota)
# ==========================================
@st.cache_resource(show_spinner=True)
def load_model_objects():
    # 1) Ruta local en el repo
    if os.path.exists(MODEL_PATH):
        objetos = joblib.load(MODEL_PATH)
        return objetos

    # 2) Secrets: URL directa o Google Drive ID
    model_url = st.secrets.get("MODEL_URL", None)
    gdrive_id = st.secrets.get("MODEL_GDRIVE_ID", None)

    if (model_url is None) and (gdrive_id is None):
        raise FileNotFoundError("No se encontró el modelo local y no hay MODEL_URL/MODEL_GDRIVE_ID en st.secrets.")

    tmp_path = "/tmp/modelos_hurdle_tweedie.pkl"
    if model_url is not None:
        import requests
        r = requests.get(model_url, timeout=120)
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.write(r.content)
    else:
        # Descargar desde Google Drive con gdown
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown
        gdown.download(id=gdrive_id, output=tmp_path, quiet=False)

    objetos = joblib.load(tmp_path)
    return objetos

# ==========================================
# STREAMLIT UI
# ==========================================
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Estimador de Prima Pura",
    page_icon="💼",
    layout="centered"
)

# ==== ESTILOS PERSONALIZADOS ====
st.markdown("""
    <style>
    /* Fondo general */
    .stApp {
        background-color: #f4f6f9;
    }

    /* Banner */
    .banner {
        background: linear-gradient(90deg, #002B5B 0%, #005B96 100%);
        color: white;
        padding: 30px 10px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 25px;
    }

    .banner h1 {
        font-size: 2.2em;
        margin-bottom: 5px;
    }

    .banner p {
        font-size: 1.1em;
        color: #e3e3e3;
        margin-top: 0;
    }

    /* Botones y métricas */
    div[data-testid="stMetricValue"] {
        color: #002B5B;
        font-weight: bold;
    }

    .stButton>button {
        background-color: #005B96;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        border: none;
    }

    .stButton>button:hover {
        background-color: #0074CC;
        color: white;
        border: none;
    }

    /* Pie de página */
    .footer {
        text-align: center;
        font-size: 0.85em;
        color: gray;
        margin-top: 40px;
        border-top: 1px solid #ddd;
        padding-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# ==== CABECERA ====
st.markdown("""
    <div class="banner">
        <h1>🔢 Estimador de Prima Pura</h1>
        <p>Modelo Actuarial Hurdle + Tweedie • Proyecto de Estimación Individual</p>
    </div>
""", unsafe_allow_html=True)


# ==== CARGAR MODELOS ====
try:
    objetos = load_model_objects()
    preprocess = objetos["preprocess"]
    modelos_freq = objetos["modelos_freq"]
    modelos_sev  = objetos["modelos_sev"]
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()


# ==== FORMULARIO ====
st.subheader("🧾 Ingrese los datos del estudiante")

with st.expander("👤 Información general", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        anio = st.selectbox("Año cursado", ["1ro año", "2do año", "3ro año", "4to año", "posgrado"], index=3)
        area = st.selectbox("Área de estudios", ["Administracion", "Humanidades", "Ciencias", "Otro"], index=1)
        calif_prom = st.number_input("Calificación promedio", min_value=0.0, max_value=10.0, value=7.01, step=0.01, format="%.2f")
        dos_mas = st.selectbox("¿2 o más inquilinos?", ["No", "Sí"], index=0)

    with col2:
        en_campus = st.selectbox("¿Vive fuera del campus?", ["No", "Sí"], index=1)

        # Bloquear el campo de distancia si vive en campus
        if en_campus == "No":
            dist_campus = st.number_input(
                "Distancia al campus (km)",
                min_value=0.0, max_value=0.0, value=0.0,
                step=0.0, format="%.6f",
                disabled=True
            )
        else:
            dist_campus = st.number_input(
                "Distancia al campus (km)",
                min_value=0.0, value=1.111582, step=0.000001, format="%.6f"
            )

        genero = st.selectbox("Género", ["Masculino", "Femenino", "Otro"], index=0)
        extintor = st.selectbox("¿Tiene extintor?", ["No", "Sí"], index=1)


# ==== BOTÓN DE CÁLCULO ====
if st.button("🚀 Calcular prima"):
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
        df_pred = predecir_prima_pura_total(
            nuevo, NUM_COLS, CAT_COLS, COBERTURAS, preprocess, modelos_freq, modelos_sev
        )

        st.success("✅ Predicción realizada con éxito")

        st.divider()
        st.subheader("📊 Resultados de la estimación")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("💰 Prima pura total", f"${df_pred['prima_pura_total'].iloc[0]:,.4f}")
        with col2:
            st.metric("📁 Número de coberturas", f"{len(COBERTURAS)}")

        st.write("### Detalle por cobertura")
        st.dataframe(df_pred[COBERTURAS].round(4), use_container_width=True)

        # Botón descarga
        st.download_button(
            "⬇️ Descargar CSV",
            data=df_pred.to_csv(index=False).encode("utf-8"),
            file_name="prediccion_individual.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error al predecir: {e}")
        st.stop()


# ==== INFORMACIÓN TÉCNICA ====
with st.expander("🔧 Información técnica"):
    import sklearn, numpy, scipy, pandas, joblib as jb
    st.write({
        "scikit_learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "joblib": jb.__version__
    })


# ==== PIE DE PÁGINA ====
st.markdown("""
    <div class="footer">
        Desarrollado con ❤️ en Streamlit · Modelo actuarial de prima pura Hurdle + Tweedie <br>
        <span style="font-size:12px;">© 2025 Proyecto académico - Universidad Nacional de Colombia</span>
    </div>
""", unsafe_allow_html=True)

