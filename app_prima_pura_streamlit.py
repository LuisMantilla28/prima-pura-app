import os
import importlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import warnings
from scipy.special import expit, logit
from datetime import datetime



# ==========================================
# CONFIG
# ==========================================
COBERTURAS = [
    "Gastos_Adicionales_siniestros_monto",
    "Gastos_Medicos_RC_siniestros_monto",
    "Resp_Civil_siniestros_monto",
    "Contenidos_siniestros_monto",
]

CAT_COLS = ['año_cursado', 'estudios_area', '2_o_mas_inquilinos', 'en_campus', 'genero', 'extintor_incendios']
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
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(x)
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
            sev = np.clip(modelos_sev[c].predict(X_nuevos), 0, None)
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
import plotly.graph_objects as go

# --- LOGO PARA LA PESTAÑA ---
LOGO_URL = "https://raw.githubusercontent.com/LuisMantilla28/prima-pura-app/main/losog_simple-removebg-preview.png"

# *** CONFIG DE PÁGINA: título + FAVICON (logo) ***
st.set_page_config(
    page_title="Estimador de Prima Pura",
    page_icon=LOGO_URL,     # <- icono de la pestaña
    layout="wide"
)


# ==== ESTILO GLOBAL: fondo blanco + contenedor centrado ====
css_container = '''
<style>
/* Fondo blanco forzado */
html, body, [class*="stAppViewContainer"], [class*="stApp"] {
  background-color: white !important;
  color: #002D62 !important;
}

/* Sidebar claro (si lo usas) */
[data-testid="stSidebar"] {
  background-color: #F8FAFF !important;
}

/* Contenedor central con ancho máximo */
.main-container {
  max-width: 1150px;  /* ajusta 900–1200 px a gusto */
  margin: 0 auto;     /* centra */
  padding: 0 1rem;    /* respiración lateral */
}
</style>

<div class="main-container">
'''
st.markdown(css_container, unsafe_allow_html=True)




# ==== ENCABEZADO ====
st.markdown("""
<style>
/* ======= Encabezado ======= */
.header {
    background: linear-gradient(90deg, #002D62, #0055A4, #0078D7);
    color: white;
    text-align: center;
    padding: 1.5rem 1rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    margin-bottom: 28px;
}
.header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
}
.header p {
    font-size: 1.3rem;
    color: white;
    margin-top: 8px;
    font-weight: 600;
}
.header::after {
    content: "";
    display: block;
    height: 3px;
    width: 75%;
    margin: 14px auto 0;
    background: linear-gradient(90deg, #66B2FF, #99CCFF);
    border-radius: 5px;
}
/* ======= Botón ======= */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #002D62, #0055A4, #0078D7);
    color: white;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1.2rem;
    box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #003D82, #0065BF, #199BFF);
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.25);
}
/* ======= Footer ======= */
.footer {
    background-color: #f2f2f2;
    color: #333;
    font-size: 0.85rem;
    text-align: center;
    padding: 0.8rem;
    border-radius: 8px;
    margin-top: 40px;
    border-top: 2px solid #0078D7;
}
.footer a {
    color: #0078D7;
    text-decoration: none;
    font-weight: 600;
}
.footer a:hover {
    text-decoration: underline;
}
/* ======= Ajuste espacio tabla ======= */
.block-container { padding-bottom: 0rem !important; }
[data-testid="stPlotlyChart"] { margin-bottom: -40px !important; }
/* ======= Responsivo ======= */
@media (max-width: 600px) {
    .header h1 { font-size: 1.5rem; }
    .header p { font-size: 1.1rem; }
}
</style>
<div class="header">
    <h1>Póliza Dormitorios</h1>
    <p><strong>Seguros Sigma</strong></p>
</div>
""", unsafe_allow_html=True)

# ==== FORMULARIO ====
st.write("👤 Ingrese los datos del estudiante:")

try:
    objetos = load_model_objects()
    preprocess = objetos["preprocess"]
    modelos_freq = objetos["modelos_freq"]
    modelos_sev = objetos["modelos_sev"]
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    anio = st.selectbox("🎓 Año cursado", ["1ro año", "2do año", "3ro año", "4to año", "posgrado"], index=3)
    area = st.selectbox("🏫 Área de estudios", ["Administracion", "Humanidades", "Ciencias", "Otro"], index=1)
    calif_prom = st.number_input("📊 Calificación promedio", min_value=0.0, max_value=10.0, value=7.01, step=0.01, format="%.2f")
    dos_mas = st.selectbox("👥 ¿2 o más inquilinos?", ["No", "Sí"], index=0)
with col2:
    en_campus = st.selectbox("🏠 ¿Vive fuera del campus?", ["No", "Sí"], index=1)
    if en_campus == "No":
        dist_campus = st.number_input("📏 Distancia al campus (km)", min_value=0.0, max_value=0.0, value=0.0, step=0.0, format="%.6f", disabled=True)
    else:
        dist_campus = st.number_input("📏 Distancia al campus (km)", min_value=0.0, value=1.111582, step=0.000001, format="%.6f")
    genero = st.selectbox("⚧️ Género", ["Masculino", "Femenino", "Otro", "No respuesta"], index=0)
    extintor = st.selectbox("🧯 ¿Tiene extintor?", ["No", "Sí"], index=1)

# ==== BOTÓN DE CÁLCULO ====
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
        df_pred = predecir_prima_pura_total(
            nuevo, NUM_COLS, CAT_COLS, COBERTURAS, preprocess, modelos_freq, modelos_sev
        )
        df_pred = df_pred.round(0)
        st.session_state["df_pred"] = df_pred
        st.session_state["prima_pura_total"] = df_pred["prima_pura_total"].iloc[0]
        st.session_state["calculada"] = True
        st.success("✅ Predicción realizada con éxito")
    except Exception as e:
        st.error(f"Error al predecir: {e}")
        st.stop()

# Mostrar resultados si ya se calculó
if st.session_state.get("calculada", False):
    df_pred = st.session_state["df_pred"]

    # ==== TABLA ====
    TITULOS = {
        "Gastos_Adicionales_siniestros_monto": "💼 Gastos Adicionales",
        "Contenidos_siniestros_monto": "🏠 Contenidos",
        "Resp_Civil_siniestros_monto": "⚖️ Responsabilidad Civil",
        "Gastos_Medicos_RC_siniestros_monto": "🩺 Gastos Médicos RC",
    }
    headers = [f"<b>{TITULOS.get(c, c)}</b>" for c in COBERTURAS]
    cells = [df_pred[c].round(4) for c in COBERTURAS]

    fig = go.Figure(data=[go.Table(
        header=dict(values=headers, fill_color="#0055A4", align="center",
                    font=dict(color="white", size=13)),
        cells=dict(values=cells, fill_color="#F8FAFF", align="center",
                   font=dict(color="#002D62", size=12))
    )])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=160)

    st.markdown("<h2 style='color:#002D62; font-weight:800;'>💵 Prima pura por cobertura (USD)</h2>",
                unsafe_allow_html=True)
    #st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


    
    # ==========================================================
    # 📋 Detalle de coberturas con descripción y monto máximo
    # ==========================================================
    st.markdown("<h3 style='color:#002D62; font-weight:800;'>🛡️ Detalle de coberturas y montos asegurados</h3>", unsafe_allow_html=True)
    
    # --- Crear DataFrame descriptivo ---
    df_detalle = pd.DataFrame({
        "Cobertura": [
            "💼 Gastos Adicionales",
            "🩺 Gastos Médicos RC",
            "⚖️ Responsabilidad Civil",
            "🏠 Contenidos"
        ],
        "Descripción": [
            "Cubre gastos de mudanza temporal, hotel y alimentación durante reparaciones.",
            "Cubre gastos médicos a terceros por accidentes o lesiones en la propiedad asegurada.",
            "Cubre al asegurado frente a demandas por daños o lesiones dentro de la vivienda",
            "Cubre pérdida o daño de pertenencias personales, como muebles, ropa y electrodomésticos."
        ],
        "Monto máximo (USD)": [20000, 150000, 300000, 10000],
        "Prima pura (USD)": [
            df_pred["Gastos_Adicionales_siniestros_monto"].iloc[0],
            df_pred["Gastos_Medicos_RC_siniestros_monto"].iloc[0],
            df_pred["Resp_Civil_siniestros_monto"].iloc[0],
            df_pred["Contenidos_siniestros_monto"].iloc[0],
        ]
    })
    

    # ==========================================================
    # 📋 Tabla y gráfico de torta lado a lado
    # ==========================================================
    #st.markdown("<h3 style='color:#002D62; font-weight:800;'>🛡️ Detalle de coberturas y montos asegurados</h3>", unsafe_allow_html=True)
    
    col_izq, col_der = st.columns([1.4, 1])
    
    with col_izq:
        st.markdown(df_detalle.style
            .set_table_styles([
                # --- ENCABEZADOS ---
                {"selector": "thead th", "props": [
                    ("background-color", "#0055A4"),
                    ("color", "white"),
                    ("font-weight", "700"),
                    ("text-align", "center"),
                    ("font-size", "1rem")
                ]},
                # --- FILAS IMPARES ---
                {"selector": "tbody tr:nth-child(odd)", "props": [
                    ("background-color", "#FFFFFF")
                ]},
                # --- FILAS PARES ---
                {"selector": "tbody tr:nth-child(even)", "props": [
                    ("background-color", "#F4F8FF")
                ]},
                # --- CELDAS GENERALES ---
                {"selector": "tbody td", "props": [
                    ("text-align", "center"),
                    ("color", "#002D62"),
                    ("font-size", "0.95rem"),
                    ("padding", "8px")
                ]}
            ])
            # --- APLICAR ESTILO A COLUMNA ESPECÍFICA ---
            .applymap(lambda v: "background-color:#E6F0FF; color:#003366; font-weight:700;", subset=["Prima pura (USD)"])
            .hide(axis="index")
            .format({
                "Monto máximo (USD)": "{:,.0f}",
                "Prima pura (USD)": "{:,.2f}"
            })._repr_html_(),
            unsafe_allow_html=True
        )
        #st.markdown(df_detalle.style
         #   .set_table_styles([
          #      {"selector": "thead th", "props": [("background-color", "#0055A4"), ("color", "white"),
           #                                        ("font-weight", "600"), ("text-align", "center")]},
            #    {"selector": "tbody td", "props": [("text-align", "center"), ("color", "#002D62"), ("font-size", "0.95rem")]},
             #   {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#F2F6FF")]}
            #])
            #.hide(axis="index")
            #.format({"Monto máximo (USD)": "{:,.0f}", "Prima pura (USD)": "{:,.2f}"})._repr_html_(),
            #unsafe_allow_html=True
        #)
    
    with col_der:
        fig_pie = go.Figure(
        data=[go.Pie(
            labels=df_detalle["Cobertura"],
            values=df_detalle["Prima pura (USD)"],
            hole=0.45,
            textinfo="label+percent",        # muestra etiqueta y porcentaje
            textposition="outside",          # coloca texto fuera del gráfico
            insidetextorientation="radial",  # asegura orientación legible
            marker=dict(colors=["#0078D7", "#3399FF", "#66B2FF", "#99CCFF"]),
            hoverinfo="label+value+percent",
            pull=[0, 0, 0, 0],         # sutil separación para claridad
            showlegend=False                 # quitamos leyenda para no duplicar
        )]
    )
    
        fig_pie.update_layout(
            title=dict(text="Distribución de la prima pura por cobertura", font=dict(size=15, color="#002D62")),
            showlegend=False,
            height=320,
            margin=dict(t=60, b=40, l=20, r=60)  # deja espacio para etiquetas externas
        )
        
        st.plotly_chart(fig_pie, use_container_width=False, config={"displayModeBar": False})
        

    # ==== MÉTRICA PRINCIPAL ====
    prima_pura = st.session_state["prima_pura_total"]
    st.markdown("<h2 style='color:#002D62; font-weight:800;'>💰 Prima pura total (USD)</h2>",
                unsafe_allow_html=True)
    

    st.markdown(f"""
    <div style='
        background:linear-gradient(90deg,#E8F1FF,#F4F8FF);
        border:2px solid #0078D7;
        border-radius:12px;
        padding:14px;
        text-align:center;
        font-size:1.6rem;
        font-weight:800;
        color:#003366;
        box-shadow:0 2px 6px rgba(0,0,0,0.1);
    '>
     {prima_pura:,.2f} USD
    </div>
    """, unsafe_allow_html=True)


    # ==== SLIDERS REACTIVOS ====
    st.markdown("<hr style='border: 1px solid #E6EAF0; margin: 24px 0;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#002D62; font-weight:800;'>💵 Prima Comercial (USD)</h2>",
                unsafe_allow_html=True)
    
    # === Sliders ===
    col1, col2, col3 = st.columns(3)
    with col1:
        gastos = st.slider("Gastos administrativos (%)", 0, 50, 20, key="gastos")
    with col2:
        utilidad = st.slider("Utilidad (%)", 0, 30, 10, key="utilidad")
    with col3:
        impuestos = st.slider("Impuestos (%)", 0, 20, 5, key="impuestos")
    
    # === Cálculo ===
    factor_total = 1 + (gastos + utilidad + impuestos) / 100
    prima_pura = st.session_state["prima_pura_total"]
    prima_comercial = prima_pura * factor_total

    # ==========================================================
    # 🧭 CLASIFICACIÓN DEL PERFIL DE RIESGO 
    # ==========================================================
    inq = int(_to_int(dos_mas))
    camp = int(_to_int(en_campus))
    ext = int(_to_int(extintor))
    
    # --- Clasificación y factores ---
    if inq == 1 and camp == 1 and ext == 0:
        nivel_riesgo = "Alto"
        factores = ["🏠 Vive <b>fuera del campus</b>.",
                    "👥 Tiene <b>2 o más inquilinos</b>.",
                    "🔥 No cuenta con <b>extintor</b>."]
    elif (inq == 1 and camp == 0 and ext == 0):
        nivel_riesgo = "Medio"
        factores = ["🏠 Vive <b>dentro del campus</b>.",
                    "👥 Tiene <b>2 o más inquilinos</b>.",
                    "🔥 No cuenta con <b>extintor</b>."]
    elif (inq == 0 and camp == 1 and ext == 0):
        nivel_riesgo = "Medio"
        factores = ["🏠 Vive <b>fuera del campus</b>.",
                    "👤 No comparte con otros inquilinos.",
                    "🔥 No cuenta con <b>extintor</b>."]
    elif inq == 1 and camp == 1 and ext == 1:
        nivel_riesgo = "Medio-alto"
        factores = ["🏠 Vive <b>fuera del campus</b>.",
                    "👥 Tiene <b>2 o más inquilinos</b>.",
                    "🧯 Cuenta con <b>extintor</b>."]
    elif (inq == 0 and camp == 1 and ext == 1):
        nivel_riesgo = "Medio-bajo"
        factores = ["🏠 Vive <b>fuera del campus</b>.",
                    "👤 No comparte con otros inquilinos.",
                    "🧯 Cuenta con <b>extintor</b>."]
    elif (inq == 1 and camp == 0 and ext == 1):
        nivel_riesgo = "Medio-bajo"
        factores = ["🏠 Vive <b>fuera del campus</b>.",
                    "👥 Tiene <b>2 o más inquilinos</b>.",
                    "🧯 Cuenta con <b>extintor</b>."]
    elif (inq == 0 and camp == 0 and ext == 0):
        nivel_riesgo = "Bajo"
        factores = ["🏠 Vive <b>dentro del campus</b>.",
                    "👤 No comparte con otros inquilinos.",
                    "🔥 No cuenta con <b>extintor</b>."]
    else:
        nivel_riesgo = "Bajo"
        factores = ["🏠 Vive <b>dentro del campus</b>.",
                    "👤 No comparte con otros inquilinos.",
                    "🧯 Tiene <b>extintor</b>."]
    
    # ==========================================================
    # 💵 Tabla Prima Comercial + Barra de Riesgo lado a lado
    # ==========================================================
    col_izq, col_der = st.columns([1, 1.1])  # relación ajustable (más espacio a la barra)
    
    with col_izq:
       st.markdown(f"""
        <table style="width:100%; border-collapse:collapse; margin-top:10px;">
        <thead style="background-color:#0055A4; color:white; font-weight:600;">
        <tr>
          <th style="padding:8px; text-align:center;">Concepto</th>
          <th style="padding:8px; text-align:center;">%</th>
          <th style="padding:8px; text-align:center;">Valor (USD)</th>
        </tr>
        </thead>
        <tbody style="background-color:#F8FAFF; color:#002D62; font-size:1.05rem;">
        <tr><td style="padding:6px;">Prima pura</td><td style="text-align:center;">—</td><td style="text-align:right;">{prima_pura:,.2f}</td></tr>
        <tr><td style="padding:6px;">Gastos administrativos</td><td style="text-align:center;">{gastos}%</td><td style="text-align:right;">{prima_pura*gastos/100:,.2f}</td></tr>
        <tr><td style="padding:6px;">Utilidad</td><td style="text-align:center;">{utilidad}%</td><td style="text-align:right;">{prima_pura*utilidad/100:,.2f}</td></tr>
        <tr><td style="padding:6px;">Impuestos</td><td style="text-align:center;">{impuestos}%</td><td style="text-align:right;">{prima_pura*impuestos/100:,.2f}</td></tr>
        
        <!-- 🔵 NUEVA FILA MEJORADA -->
        <tr style="background-color:#004AAD; color:white; font-weight:900; font-size:1.1rem;">
          <td style="padding:6px;">Prima comercial total</td>
          <td style="text-align:center;">—</td>
          <td style="text-align:right;">{prima_comercial:,.2f}</td>
        </tr>
        
        </tbody>
        </table>
        """, unsafe_allow_html=True)
            
    with col_der:
        niveles = ["Bajo", "Medio-bajo", "Medio", "Medio-alto", "Alto"]
        colores = ["#80CFA9", "#FFF176", "#FFD54F", "#FB8C00", "#E53935"]
        idx = niveles.index(nivel_riesgo)
    
        segmentos_html = "".join([
            f"<div class='segmento' style='background:{col}; opacity:{'1' if i==idx else '0.35'};'></div>"
            for i, col in enumerate(colores)
        ])
        factores_html = "".join([f"<li>{f}</li>" for f in factores])
    
        html_final = f"""
        <div class="tarjeta">
            <h3 class="titulo">🏷️ Nivel de Riesgo: {nivel_riesgo}</h3>
            <div class="barra-container">
                <div class="barra">{segmentos_html}</div>
                <div class="etiquetas">
                    {''.join([f"<span>{niv}</span>" for niv in niveles])}
                </div>
                <div class="flecha" style="left: calc({idx} * 20% + 10%);"></div>
            </div>
            <ul class="factores">{factores_html}</ul>
        </div>
        <style>
        .tarjeta {{
            background:#F8FAFF; border-radius:16px; padding:1.5rem;
            box-shadow:0 3px 12px rgba(0,0,0,0.15); margin-top:10px;
            animation: fadeIn 0.9s ease-in-out;
        }}
        .titulo {{ color:#003366; text-align:center; font-weight:800;
            font-size:1.4rem; margin-bottom:1rem; }}
        .barra-container {{ position:relative; margin-bottom:1.4rem; }}
        .barra {{ display:flex; height:24px; border-radius:6px; overflow:hidden; }}
        .segmento {{ flex:1; transition:opacity 0.4s ease; }}
        .etiquetas {{ display:flex; justify-content:space-between; margin-top:6px;
            font-size:0.9rem; font-weight:600; color:#003366; }}
        .flecha {{ position:absolute; top:24px; transform:translateX(-50%);
            width:0; height:0; border-left:9px solid transparent;
            border-right:9px solid transparent; border-top:12px solid #003366; }}
        .factores {{ margin-top:12px; margin-left:20px; color:#002D62;
            font-size:1.05rem; line-height:1.6; }}
        </style>
        """
        st.markdown(html_final, unsafe_allow_html=True)


        # ====== RESUMEN FINAL COMPACTO ======
        st.markdown(f"""
        <div style='text-align:center; margin-top:30px; font-size:1.1rem; color:#002D62;'>
        ✅ <b>Prima comercial total:</b> USD {prima_comercial:,.2f} |
        🏷️ <b>Nivel de riesgo:</b> {nivel_riesgo}
        </div>
        """, unsafe_allow_html=True)
   

        
# ==== INFO TÉCNICA ====
with st.expander("🔧 Información técnica"):
    import sklearn, numpy, scipy, pandas, joblib as jb
    st.write({
        "scikit_learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "joblib": jb.__version__
    })

st.markdown("</div>", unsafe_allow_html=True)




# ==== PIE DE PÁGINA ====
st.markdown(f"""
<div class="footer">
© {datetime.now().year} Desarrollado con <a href="https://streamlit.io" target="_blank">Streamlit</a>
·💡Equipo Riskbusters - Universidad Nacional de Colombia
</div>
""", unsafe_allow_html=True)
