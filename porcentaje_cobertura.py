# app.py
# -------------------------------------------------------------
# Dashboard de Coberturas y Métricas
# - Mapa de riesgo (Plotly, responsive)
# - Tablas "Factores de riesgo" (selección) y "Cambio TOTAL" sin %Cambio
# - Etiquetas amigables en tablas
# -------------------------------------------------------------
# Requisitos:
#   pip install streamlit requests pandas numpy openpyxl scipy plotly
# Ejecutar:  streamlit run app.py
# -------------------------------------------------------------

import io
import os
import sys
import tempfile
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler  # Importar Styler directamente
import requests
import streamlit as st
from datetime import datetime

import plotly.express as px
from plotly import graph_objects as go

# ================================
# CONFIG
# ================================
REMOTE_PY_URL = os.getenv("REMOTE_PY_URL", "")
REMOTE_MODULE_NAME = "modelo_remoto"
LOGO_URL = "https://raw.githubusercontent.com/LuisMantilla28/prima-pura-app/main/Logo_competencia.jpg"

EXCEL_URL = "https://raw.githubusercontent.com/LuisMantilla28/prima-pura-app/main/predicciones_train_test_una_hoja.xlsx"

COBERTURAS = [
    "Gastos_Adicionales_siniestros_monto",
    "Gastos_Medicos_RC_siniestros_monto",
    "Resp_Civil_siniestros_monto",
    "Contenidos_siniestros_monto",
]

# Variables que se muestran en tablas (selección)
VARS_BIN = [
    "num_bin__2_o_mas_inquilinos",
    "num_bin__en_campus",
    "num_bin__extintor_incendios",
]

# Etiquetas legibles para columnas "Variable"
VAR_LABELS = {
    "num_bin__2_o_mas_inquilinos": "Tener 2 o más inquilinos",
    "num_bin__en_campus": "Vivir fuera del campus",
    "num_bin__extintor_incendios": "Tener extintor",
}

# ================================
# ESTILO (CSS)
# ================================
EXECUTIVE_CSS = """
<style>
html, body, [class*="css"], .stMarkdown, .stText, .stDataFrame {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
}
.block-container { padding-top: 0.9rem; padding-bottom: 1.0rem; }
h1, .title-text { font-weight: 700; letter-spacing: -0.02em; }

/* KPI compactos (misma altura que "Cobertura") */
.kpi-card {
  background: #1E3A8A; border: 1px solid rgba(0,0,0,0.06); border-radius: 12px;
  padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); color:#fff; margin: 0;
}
.kpi-card .metric-label { font-size: 0.78rem; margin-bottom: 4px; opacity:0.9; }
.kpi-card .metric-value { font-size: 1.08rem; font-weight: 700; line-height: 1.2; }

h3, h4 { margin: 0.2rem 0 0.6rem 0; }
.caption { color: #6b7280 !important; text-transform: uppercase; letter-spacing: .03em; font-size: .78rem; }
.footer { margin-top: 0.6rem; color:#6b7280; }
</style>
"""

EXECUTIVE_CSS += """
<style>
/* Forzar modo claro */
html, body, [class*="stAppViewContainer"] {
  background-color: #FFFFFF !important;
  color: #111111 !important;
}
.stApp { background-color: #FFFFFF !important; color: #111111 !important; }
div[data-testid="stMarkdown"] p { color: #111111 !important; }
</style>
"""

# Paleta niveles
NIVELES_RIESGO = ["Bajo", "Medio-bajo", "Medio", "Medio-alto", "Alto"]
COLOR_MAP = {
    "Bajo": "#2E8B57",
    "Medio-bajo": "#F2C94C",
    "Medio": "#F5A623",
    "Medio-alto": "#D35400",
    "Alto": "#C0392B",
}

# Mapa fijo de perfiles
RISK_MAP_FIXED = {
    "0_0_1": "Bajo",
    "0_0_0": "Bajo",
    "0_1_1": "Medio-bajo",
    "1_0_1": "Medio-bajo",
    "0_1_0": "Medio",
    "1_0_0": "Medio",
    "1_1_1": "Medio-alto",
    "1_1_0": "Alto",
}

# -------------------------------------------------------------
# Módulo remoto opcional
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_remote_module(raw_url: str, module_name: str):
    if not raw_url:
        return None
    try:
        resp = requests.get(raw_url, timeout=20)
        resp.raise_for_status()
        code = resp.text
        tmpdir = tempfile.mkdtemp(prefix="remotepy_")
        module_path = os.path.join(tmpdir, f"{module_name}.py")
        with open(module_path, "w", encoding="utf-8") as f:
            f.write(code)
        if tmpdir not in sys.path:
            sys.path.insert(0, tmpdir)
        mod = __import__(module_name)
        return mod
    except Exception as e:
        st.warning(f"No se pudo cargar el módulo remoto: {e}")
        return None

# -------------------------------------------------------------
# Lectura Excel
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_excel_from_url(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=45)
    resp.raise_for_status()
    return pd.read_excel(io.BytesIO(resp.content))

def build_perfil(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["2_o_mas_inquilinos", "en_campus", "extintor_incendios"]:
        if c not in df.columns:
            raise ValueError(f"Falta la columna '{c}' en el Excel.")
        df[c] = df[c].apply(lambda x: 1 if str(x).strip().lower() in {"1","si","sí","true","y","s"} else 0)
    df["perfil_base"] = (
        df["2_o_mas_inquilinos"].astype(str) + "_" +
        df["en_campus"].astype(str) + "_" +
        df["extintor_incendios"].astype(str)
    )
    df["nivel_riesgo"] = df["perfil_base"].map(RISK_MAP_FIXED).fillna("Medio")
    return df

def ensure_pred_cols(df: pd.DataFrame, cobertura: str):
    col_freq = f"{cobertura}_freq_pred"
    col_sev  = f"{cobertura}_sev_pred"
    col_pri  = f"{cobertura}_prima_pred"
    for c in [col_freq, col_sev, col_pri]:
        if c not in df.columns:
            raise ValueError(f"Falta la columna '{c}' para la cobertura '{cobertura}'.")
    return col_freq, col_sev, col_pri

# -------------------------------------------------------------
# PLOT: Scatter Plotly (responsive, más pequeño y con borde blanco)
# -------------------------------------------------------------
def scatter_plotly(df: pd.DataFrame, cobertura: str, sample_max: int = 8000):
    col_freq, col_sev, col_pri = ensure_pred_cols(df, cobertura)
    df_plot = df.sample(sample_max, random_state=42) if len(df) > sample_max else df.copy()

    raw = pd.to_numeric(df_plot[col_pri], errors="coerce").fillna(0).values
    raw = np.clip(raw, np.nanpercentile(raw, 5), np.nanpercentile(raw, 95))

    SIZE_MIN = 4
    SIZE_MAX = 12  # más pequeño
    s_norm = SIZE_MIN + (raw - raw.min()) * (SIZE_MAX - SIZE_MIN) / (raw.max() - raw.min() + 1e-12)
    df_plot["_size_"] = s_norm

    c_label = cobertura.replace("_siniestros_monto", "").replace("_", " ").capitalize()
    fig = go.Figure()

    for nivel in NIVELES_RIESGO:
        sub = df_plot[df_plot["nivel_riesgo"] == nivel]
        if sub.empty:
            continue
        fig.add_trace(go.Scattergl(
            x=pd.to_numeric(sub[col_freq], errors="coerce"),
            y=pd.to_numeric(sub[col_sev], errors="coerce"),
            mode="markers",
            name=nivel,
            marker=dict(
                size=sub["_size_"],
                color=COLOR_MAP.get(nivel, "#999"),
                line=dict(width=0.8, color="white"),  # borde blanco
                opacity=0.9,
            ),
            text=sub["nivel_riesgo"],
            customdata=np.array(sub[col_pri]),
            hovertemplate=(
                "<b>Nivel:</b> %{text}<br>"
                "E[N]: %{x:.3f}<br>"
                "E[Y|N>0]: %{y:.2f}<br>"
                "Prima esperada: %{customdata:.2f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=f"Mapa de riesgo – {c_label}",
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
        legend_title_text="Nivel de riesgo",
        plot_bgcolor="#FBFBFB",
        hovermode="closest",
    )
    fig.update_xaxes(title_text="Frecuencia esperada E[N]", gridcolor="rgba(0,0,0,0.15)", zeroline=False)
    fig.update_yaxes(title_text="Severidad esperada E[Y | N>0]", gridcolor="rgba(0,0,0,0.15)", zeroline=False)
    return fig

# -------------------------------------------------------------
# Datos fallback (métricas y tablas)
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_fallback_data() -> Dict[str, Any]:
    header_metrics = {
        "Gastos_Adicionales_siniestros_monto": {"Media real de N": 0.055625,"Media predicha de N": 0.05318535298803486,"Severidad esperada media (predicha)": 5425.88,"Severidad real media (observada)": 5395.85,},
        "Gastos_Medicos_RC_siniestros_monto": {"Media real de N": 0.021875,"Media predicha de N": 0.023721117738488857,"Severidad esperada media (predicha)": 17170.89,"Severidad real media (observada)": 15154.66,},
        "Resp_Civil_siniestros_monto": {"Media real de N": 0.009375,"Media predicha de N": 0.008137502924526774,"Severidad esperada media (predicha)": 7629.31,"Severidad real media (observada)": 9311.54,},
        "Contenidos_siniestros_monto": {"Media real de N": 0.103125,"Media predicha de N": 0.09827778812479611,"Severidad esperada media (predicha)": 987.06,"Severidad real media (observada)": 971.40,},
    }

    cambio_por_cobertura = {
        "Gastos_Adicionales_siniestros_monto": pd.DataFrame({"Variable": ["num_bin__2_o_mas_inquilinos","num_bin__en_campus","multi__genero_No respuesta","multi__año_cursado_4to año","multi__año_cursado_posgrado","multi__año_cursado_3er año","multi__estudios_area_Otro","multi__genero_Masculino","num_bin__distancia_al_campus","multi__estudios_area_Humanidades","num_bin__calif_promedio","multi__estudios_area_Ciencias","multi__año_cursado_2do año","multi__genero_Otro","num_bin__extintor_incendios"],"%Cambio_prima": [354.8370,94.7330,46.7106,42.9091,36.2409,11.1749,10.2217,1.7919,0.4819,-2.8914,-2.9553,-5.3370,-12.7520,-17.4318,-46.2605]}),
        "Gastos_Medicos_RC_siniestros_monto": pd.DataFrame({"Variable": ["num_bin__2_o_mas_inquilinos","num_bin__en_campus","multi__año_cursado_posgrado","multi__año_cursado_3er año","multi__estudios_area_Humanidades","multi__genero_No respuesta","multi__año_cursado_2do año","num_bin__distancia_al_campus","multi__año_cursado_4to año","multi__genero_Otro","multi__estudios_area_Otro","num_bin__calif_promedio","multi__estudios_area_Ciencias","multi__genero_Masculino","num_bin__extintor_incendios"],"%Cambio_prima": [275.3948,187.5584,68.6651,42.8082,15.2331,10.5232,7.3051,5.3001,1.7714,0.8731,-4.1862,-7.7901,-13.3933,-18.6933,-47.4933]}),
        "Resp_Civil_siniestros_monto": pd.DataFrame({"Variable": ["num_bin__2_o_mas_inquilinos","multi__año_cursado_posgrado","num_bin__en_campus","num_bin__distancia_al_campus","num_bin__calif_promedio","multi__estudios_area_Otro","multi__año_cursado_3er año","multi__genero_Masculino","multi__estudios_area_Ciencias","multi__año_cursado_4to año","num_bin__extintor_incendios","multi__genero_No respuesta","multi__estudios_area_Humanidades","multi__genero_Otro","multi__año_cursado_2do año"],"%Cambio_prima": [448.2017,53.9895,23.9554,21.8542,10.3560,7.4916,-29.3219,-29.7620,-32.1020,-38.5175,-38.9797,-47.4722,-51.1230,-52.5578,-66.1921]}),
        "Contenidos_siniestros_monto": pd.DataFrame({"Variable": ["num_bin__2_o_mas_inquilinos","num_bin__en_campus","multi__año_cursado_3er año","multi__año_cursado_posgrado","multi__genero_No respuesta","multi__genero_Otro","multi__genero_Masculino","multi__año_cursado_2do año","num_bin__distancia_al_campus","num_bin__calif_promedio","multi__año_cursado_4to año","multi__estudios_area_Ciencias","multi__estudios_area_Otro","multi__estudios_area_Humanidades","num_bin__extintor_incendios"],"%Cambio_prima": [345.1322,119.3666,29.8415,22.5027,18.0567,10.3456,5.1074,2.1588,-1.0780,-2.0532,-2.9586,-7.2140,-8.6788,-26.5477,-29.9581]}),
    }

    cambio_total = pd.DataFrame({
        "Variable": ["num_bin__2_o_mas_inquilinos","num_bin__en_campus","multi__año_cursado_posgrado","multi__año_cursado_3er año","multi__genero_No respuesta","multi__año_cursado_4to año","num_bin__distancia_al_campus","multi__estudios_area_Otro","multi__estudios_area_Humanidades","num_bin__calif_promedio","multi__año_cursado_2do año","multi__genero_Otro","multi__genero_Masculino","multi__estudios_area_Ciencias","num_bin__extintor_incendios"],
        "Factor_total": [4.2280,2.3657,1.5125,1.2548,1.1920,1.1190,1.0414,1.0090,0.9932,0.9583,0.9466,0.9205,0.9017,0.8866,0.5566],
        "%Cambio_total": [322.8016,136.5736,51.2544,25.4803,19.1971,11.8959,4.1399,0.9005,-0.6814,-4.1669,-5.3377,-7.9533,-9.8288,-11.3409,-44.3401],
    })

    return {"header_metrics": header_metrics,
            "cambio_por_cobertura": cambio_por_cobertura,
            "cambio_total": cambio_total}

# -------------------------------------------------------------
# Acceso módulo remoto
# -------------------------------------------------------------
def try_remote_get_metrics(mod) -> Optional[Dict[str, Any]]:
    try:
        if mod and hasattr(mod, "get_metrics"):
            return mod.get_metrics()
    except Exception as e:
        st.warning(f"Fallo get_metrics() del módulo remoto: {e}")
    return None

# -------------------------------------------------------------
# UI helpers
# -------------------------------------------------------------
def fmt_float(x, nd=4):
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return x

def kpi(label: str, value, nd: int = 4):
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{fmt_float(value, nd)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_table(df: pd.DataFrame, caption: str, column_config: Dict[str, st.column_config.Column] = None):
    st.caption(caption)
    st.dataframe(df, use_container_width=True, hide_index=True, column_config=column_config or {})

# === Helper para mostrar Styler con encabezados NEGROS, sin índice y a 100% de ancho ===
def show_styler(sty: Styler):
    # 1) Ocultar la primera columna (índice)
    try:
        sty = sty.hide(axis="index")      # pandas >= 1.4
    except Exception:
        sty = sty.hide_index()            # compatibilidad con pandas más viejo

    # 2) Encabezados en negro (se mantiene)
    sty = sty.set_table_styles([
        {"selector": "th.col_heading", "props": [("color", "#000000"), ("font-weight", "600")]},
        {"selector": "th.row_heading", "props": [("color", "#000000"), ("font-weight", "600")]},
        {"selector": "th.blank",       "props": [("color", "#000000"), ("font-weight", "600")]},
    ], overwrite=False)

    # 3) Ancho completo de la tabla + alineación izquierda
    sty = sty.set_table_attributes('style="width:100%; table-layout:auto"')
    sty = sty.set_table_styles([
        {"selector": "table", "props": [("width", "100%")]},
        {"selector": "th",    "props": [("text-align", "left")]},
        {"selector": "td",    "props": [("text-align", "left")]},
    ], overwrite=False)

    # Render
    html = sty.to_html()
    st.markdown(f'<div style="width:100%">{html}</div>', unsafe_allow_html=True)


# ---------- Tabla fija de perfiles ----------
def get_niveles_table() -> pd.DataFrame:
    data = [
        {"nivel_riesgo":"Bajo",       "2 o más inquilinos":"No", "Ubicación":"Dentro de campus", "Extintor de incendios":"Sí",  "Prima esperada promedio":337,  "N° observaciones":2955},
        {"nivel_riesgo":"Bajo",       "2 o más inquilinos":"No", "Ubicación":"Dentro de campus", "Extintor de incendios":"No",  "Prima esperada promedio":529,  "N° observaciones":1227},
        {"nivel_riesgo":"Medio-bajo", "2 o más inquilinos":"No", "Ubicación":"Fuera de campus",  "Extintor de incendios":"Sí",  "Prima esperada promedio":816,  "N° observaciones":1586},
        {"nivel_riesgo":"Medio-bajo", "2 o más inquilinos":"Sí", "Ubicación":"Dentro de campus", "Extintor de incendios":"Sí",  "Prima esperada promedio":1006, "N° observaciones":725},
        {"nivel_riesgo":"Medio",      "2 o más inquilinos":"No", "Ubicación":"Fuera de campus",  "Extintor de incendios":"No",  "Prima esperada promedio":1248, "N° observaciones":671},
        {"nivel_riesgo":"Medio",      "2 o más inquilinos":"Sí", "Ubicación":"Dentro de campus", "Extintor de incendios":"No",  "Prima esperada promedio":1601, "N° observaciones":319},
        {"nivel_riesgo":"Medio-alto", "2 o más inquilinos":"Sí", "Ubicación":"Fuera de campus",  "Extintor de incendios":"Sí",  "Prima esperada promedio":2322, "N° observaciones":367},
        {"nivel_riesgo":"Alto",       "2 o más inquilinos":"Sí", "Ubicación":"Fuera de campus",  "Extintor de incendios":"No",  "Prima esperada promedio":3584, "N° observaciones":149},
    ]
    cols = ["nivel_riesgo","2 o más inquilinos","Ubicación","Extintor de incendios","Prima esperada promedio","N° observaciones"]
    return pd.DataFrame(data, columns=cols)

# ---------- Estilo por color de riesgo ----------
def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def _rgba_str(hex_color: str, alpha: float = 0.3) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return f"background-color: rgba({r}, {g}, {b}, {alpha});"

def style_by_risk(df: pd.DataFrame) -> Styler:
    def row_style(row):
        nivel = str(row["nivel_riesgo"])
        color_hex = COLOR_MAP.get(nivel, "#e5e7eb")
        bg = _rgba_str(color_hex, alpha=0.5)
        return [bg] * len(row)

    sty = (
        df.style
          .apply(row_style, axis=1)
          .format({"Prima esperada promedio": "{:,.0f}", "N° observaciones": "{:,}"})
    )

    # (El color negro del header se fuerza en show_styler)
    return sty

# ================================
# APP
# ================================
def main():
    st.set_page_config(
        page_title="Métricas de Prima por Cobertura",
        page_icon="https://raw.githubusercontent.com/LuisMantilla28/prima-pura-app/main/losog_simple-removebg-preview.png",
        layout="wide"
    )
    st.markdown(EXECUTIVE_CSS, unsafe_allow_html=True)

    # Header
    top_logo, top_title = st.columns([1, 6])
    with top_logo:
        if LOGO_URL:
            st.markdown("<br>", unsafe_allow_html=True)
            st.image(LOGO_URL, width=150, use_container_width=False)
    with top_title:
        st.markdown("<h1 class='title-text' style='margin-bottom:0rem; margin-top:3.5rem;'>Risk Profiling Dashboard</h1>", unsafe_allow_html=True)
        st.markdown(
            """
            <p style='font-size:0.95rem; color:#374151; margin-top:0.4rem;'>
            Esta herramienta interactiva permite visualizar y analizar el comportamiento del riesgo en pólizas de seguros para estudiantes universitarios. 
            A partir de los resultados de un modelo estadístico, el aplicativo integra datos de frecuencia, severidad y factores de riesgo para cada cobertura, 
            generando métricas y perfiles que facilitan la interpretación actuarial y la toma de decisiones basada en evidencia.
            </p>
            """,
            unsafe_allow_html=True
        )

    # Datos
    mod = load_remote_module(REMOTE_PY_URL, REMOTE_MODULE_NAME)
    data = try_remote_get_metrics(mod) or get_fallback_data()

    header_metrics: Dict[str, Dict[str, float]] = data["header_metrics"]
    cambio_por_cobertura: Dict[str, pd.DataFrame] = data.get("cambio_por_cobertura", {})
    cambio_total: pd.DataFrame = data.get("cambio_total", pd.DataFrame())

    # TABLA de perfiles (con Styler -> HTML) para tener encabezados en negro
    with st.container(border=False):
        df_perf = get_niveles_table()
        styler = style_by_risk(df_perf)
        show_styler(styler)

    # FILA SUPERIOR
    row1_left, row1_right = st.columns([1.2, 3.8], gap="large")
    with row1_left:
        with st.container(border=False):
            st.markdown("### Cobertura")
            cobertura = st.selectbox(
                "Selecciona cobertura", COBERTURAS, index=0,
                label_visibility="collapsed",
                format_func=lambda s: s.replace("_siniestros_monto", "").replace("_", " ")
            )
    with row1_right:
        with st.container(border=False):
            st.markdown("<br>", unsafe_allow_html=True)
            metrics = header_metrics.get(cobertura, {})
            g1, g2, g3, g4 = st.columns(4)
            with g1: kpi("Frecuencia media observada", metrics.get("Media real de N", np.nan))
            with g2: kpi("Frecuencia media  predicha", metrics.get("Media predicha de N", np.nan))
            with g3: kpi("Severidad real media (observada)", metrics.get("Severidad real media (observada)", np.nan), nd=2)
            with g4: kpi("Severidad esperada media (predicha)", metrics.get("Severidad esperada media (predicha)", np.nan), nd=2)

    # Excel
    try:
        df_all = read_excel_from_url(EXCEL_URL)
    except Exception as e:
        st.error(f"No se pudo leer el Excel desde GitHub RAW: {e}")
        df_all = None

    if df_all is not None:
        try:
            df_all = build_perfil(df_all)

            # FILA MEDIA: Izq scatter / Der TABLA selección
            row2_left, row2_right = st.columns([2.5, 2.0], gap="large")

            with row2_left:
                with st.container(border=True):
                    st.markdown("### Mapa de riesgo por cobertura")
                    fig_scatter = scatter_plotly(df_all, cobertura, sample_max=8000)
                    st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

                    col_freq, col_sev, col_pri = ensure_pred_cols(df_all, cobertura)
                    df_plot = df_all[[col_freq, col_sev, col_pri, "nivel_riesgo"]].copy()
                    st.download_button(
                        label="⬇️ Descargar datos de la muestra graficada (CSV)",
                        data=df_plot.to_csv(index=False).encode("utf-8"),
                        file_name=f"muestra_mapa_riesgo_{cobertura}.csv",
                        mime="text/csv"
                    )

            with row2_right:
                with st.container(border=True):
                    st.markdown("### Factores de riesgo")
                    df_cob = cambio_por_cobertura.get(cobertura, pd.DataFrame(columns=["Variable"])).copy()
                    if "%Cambio_prima" in df_cob.columns:
                        df_cob["Factor"] = (pd.to_numeric(df_cob["%Cambio_prima"], errors="coerce") / 100 + 1).round(4)

                    df_sel = df_cob[df_cob["Variable"].isin(VARS_BIN)].copy()
                    if df_sel.empty:
                        st.info("No hay datos de variables seleccionadas para esta cobertura.")
                    else:
                        df_sel["Variable"] = df_sel["Variable"].replace(VAR_LABELS)
                        cols = ["Variable", "Factor"] if "Factor" in df_sel.columns else ["Variable"]
                        df_sel = df_sel[cols].sort_values("Variable").reset_index(drop=True)

                        # === Mostrar con Styler para encabezados en negro ===
                        sty_sel = df_sel.style.format({"Factor": "{:.4f}"} if "Factor" in df_sel.columns else {})
                        show_styler(sty_sel)

                        st.markdown(
                            """
                            **¿Cómo interpretar el _Factor_?**
                            - **Factor > 1.00**: incrementa la prima esperada (p. ej., 1.25 ⇒ +25%).  
                            - **≈ 1.00**: efecto neutro o marginal sobre la prima.  
                            - **< 1.00**: reduce la prima esperada (p. ej., 0.80 ⇒ −20%).  
                            - Los factores se estiman condicionales al modelo y a la cobertura seleccionada.
                            """
                        )

            # FILA INFERIOR: TABLA total (sin %Cambio_total)
            with st.container(border=True):
                st.markdown("### Factores de riesgo prima esperada total")
                if cambio_total is None or cambio_total.empty:
                    st.info("No hay datos de cambio total disponibles.")
                else:
                    df_total_sel = cambio_total[cambio_total["Variable"].isin(VARS_BIN)].copy()
                    if df_total_sel.empty:
                        st.info("No hay datos para las variables seleccionadas en el cambio total.")
                    else:
                        df_total_sel = df_total_sel[["Variable", "Factor_total"]].sort_values("Variable").reset_index(drop=True)
                        df_total_sel["Variable"] = df_total_sel["Variable"].replace(VAR_LABELS)

                        # === Mostrar con Styler para encabezados en negro ===
                        sty_total = df_total_sel.style.format({"Factor_total": "{:.4f}"})
                        show_styler(sty_total)

        except Exception as e:
            st.error(f"Error al preparar la visualización: {e}")

    # Footer
    st.markdown(f"""
    <div class="footer">
        © {datetime.now().year} Desarrollado con 
        <a href="https://streamlit.io" target="_blank">Streamlit</a> · 💡Equipo Riskbusters - Universidad Nacional de Colombia
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
