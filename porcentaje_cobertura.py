# app.py
# -------------------------------------------------------------
# App Streamlit ejecutivo para métricas y coberturas
# + mapas y barras (matplotlib) con Excel RAW de GitHub.
# -------------------------------------------------------------
# Requisitos:
#   pip install streamlit requests pandas numpy matplotlib openpyxl scipy
# Ejecutar:  streamlit run app.py
# -------------------------------------------------------------

import io
import os
import sys
import tempfile
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ================================
# CONFIG
# ================================
REMOTE_PY_URL = os.getenv("REMOTE_PY_URL", "")
REMOTE_MODULE_NAME = "modelo_remoto"
LOGO_URL = os.getenv("LOGO_URL", "")
EXCEL_URL = "https://raw.githubusercontent.com/LuisMantilla28/prima-pura-app/main/predicciones_train_test_una_hoja.xlsx"

COBERTURAS = [
    "Gastos_Adicionales_siniestros_monto",
    "Gastos_Medicos_RC_siniestros_monto",
    "Resp_Civil_siniestros_monto",
    "Contenidos_siniestros_monto",
]

VARS_BIN = [
    "num_bin__2_o_mas_inquilinos",
    "num_bin__en_campus",
    "num_bin__extintor_incendios",
]

# ================================
# ESTILO (CSS)
# ================================
EXECUTIVE_CSS = """
<style>
html, body, [class*="css"], .stMarkdown, .stText, .stDataFrame {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
}
.block-container { padding-top: 0.9rem; padding-bottom: 1.4rem; }
h1, .title-text { font-weight: 700; letter-spacing: -0.02em; }

/* KPI más compactos (misma altura visual que el bloque "Cobertura") */
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

def ensure_pred_cols(df: pd.DataFrame, cobertura: str) -> Tuple[str, str, str]:
    col_freq = f"{cobertura}_freq_pred"
    col_sev  = f"{cobertura}_sev_pred"
    col_pri  = f"{cobertura}_prima_pred"
    for c in [col_freq, col_sev, col_pri]:
        if c not in df.columns:
            raise ValueError(f"Falta la columna '{c}' para la cobertura '{cobertura}'.")
    return col_freq, col_sev, col_pri

# -------------------------------------------------------------
# Helpers de visualización (clave: usar st.image con buffer PNG)
# -------------------------------------------------------------
def fig_to_stimage(fig, *, dpi: int = 220):
    """Renderiza un fig de matplotlib como PNG en memoria y lo muestra con st.image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, use_column_width=True)

# -------------------------------------------------------------
# PLOTS (matplotlib)
# -------------------------------------------------------------
def make_scatter_matplotlib(df: pd.DataFrame, cobertura: str, sample_max: int = 8000):
    """Scatter con contorno blanco y leyenda integrada (render estable)."""
    col_freq, col_sev, col_pri = ensure_pred_cols(df, cobertura)
    df_plot = df.sample(sample_max, random_state=42) if len(df) > sample_max else df.copy()

    # Tamaños robustos
    s = pd.to_numeric(df_plot[col_pri], errors="coerce").fillna(0).values
    s = np.clip(s, np.nanpercentile(s, 5), np.nanpercentile(s, 95))
    s_norm = 40 + (s - s.min()) * (300 - 40) / (s.max() - s.min() + 1e-9)

    colores = [COLOR_MAP.get(n, "#999999") for n in df_plot["nivel_riesgo"].values]

    # tamaño generoso; no dependemos de use_container_width
    fig, ax = plt.subplots(figsize=(11.5, 6.6))  # ancho x alto
    ax.scatter(
        pd.to_numeric(df_plot[col_freq], errors="coerce"),
        pd.to_numeric(df_plot[col_sev], errors="coerce"),
        s=s_norm,
        c=colores,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.6
    )

    c_label = cobertura.replace("_siniestros_monto", "").replace("_", " ").capitalize()
    ax.set_title(f"Mapa de riesgo – {c_label}", fontweight="bold", color="#003366")
    ax.set_xlabel("Frecuencia esperada E[N]")
    ax.set_ylabel("Severidad esperada E[Y | N>0]")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_facecolor("#FBFBFB")

    # Leyenda dentro del eje (no comprime el canvas)
    legend_patches = [mpatches.Patch(color=COLOR_MAP[n], label=n) for n in NIVELES_RIESGO]
    ax.legend(handles=legend_patches, title="Nivel de riesgo", loc="upper left",
              frameon=True, fontsize=9, title_fontsize=10)

    return fig, df_plot[[col_freq, col_sev, col_pri, "nivel_riesgo"]].copy()

def plot_bars(df: pd.DataFrame, x_col: str, y_col: str, title: str,
              xtick_rotation: int = 25, width: Tuple[float, float]=(11.0, 6.2)):
    """Barras con layout consistente y ejes legibles (render como imagen)."""
    fig, ax = plt.subplots(figsize=width)
    x_vals = df[x_col].astype(str).tolist()
    y_vals = pd.to_numeric(df[y_col], errors="coerce").fillna(0).tolist()

    ax.bar(x_vals, y_vals)
    ax.set_title(title, fontweight="bold", color="#003366")
    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel(y_col.replace("_", " "))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha="right")

    return fig

# -------------------------------------------------------------
# Datos fallback
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_fallback_data() -> Dict[str, Any]:
    header_metrics = {
        "Gastos_Adicionales_siniestros_monto": {
            "Media real de N": 0.055625,
            "Media predicha de N": 0.05318535298803486,
            "Severidad esperada media (predicha)": 5425.88,
            "Severidad real media (observada)": 5395.85,
        },
        "Gastos_Medicos_RC_siniestros_monto": {
            "Media real de N": 0.021875,
            "Media predicha de N": 0.023721117738488857,
            "Severidad esperada media (predicha)": 17170.89,
            "Severidad real media (observada)": 15154.66,
        },
        "Resp_Civil_siniestros_monto": {
            "Media real de N": 0.009375,
            "Media predicha de N": 0.008137502924526774,
            "Severidad esperada media (predicha)": 7629.31,
            "Severidad real media (observada)": 9311.54,
        },
        "Contenidos_siniestros_monto": {
            "Media real de N": 0.103125,
            "Media predicha de N": 0.09827778812479611,
            "Severidad esperada media (predicha)": 987.06,
            "Severidad real media (observada)": 971.40,
        },
    }

    cambio_por_cobertura = {
        "Gastos_Adicionales_siniestros_monto": pd.DataFrame(
            {"Variable": ["num_bin__2_o_mas_inquilinos","num_bin__en_campus","multi__genero_No respuesta","multi__año_cursado_4to año","multi__año_cursado_posgrado","multi__año_cursado_3er año","multi__estudios_area_Otro","multi__genero_Masculino","num_bin__distancia_al_campus","multi__estudios_area_Humanidades","num_bin__calif_promedio","multi__estudios_area_Ciencias","multi__año_cursado_2do año","multi__genero_Otro","num_bin__extintor_incendios"],
             "%Cambio_prima": [354.8370,94.7330,46.7106,42.9091,36.2409,11.1749,10.2217,1.7919,0.4819,-2.8914,-2.9553,-5.3370,-12.7520,-17.4318,-46.2605]}
        ),
        "Gastos_Medicos_RC_siniestros_monto": pd.DataFrame(
            {"Variable": ["num_bin__2_o_mas_inquilinos","num_bin__en_campus","multi__año_cursado_posgrado","multi__año_cursado_3er año","multi__estudios_area_Humanidades","multi__genero_No respuesta","multi__año_cursado_2do año","num_bin__distancia_al_campus","multi__año_cursado_4to año","multi__genero_Otro","multi__estudios_area_Otro","num_bin__calif_promedio","multi__estudios_area_Ciencias","multi__genero_Masculino","num_bin__extintor_incendios"],
             "%Cambio_prima": [275.3948,187.5584,68.6651,42.8082,15.2331,10.5232,7.3051,5.3001,1.7714,0.8731,-4.1862,-7.7901,-13.3933,-18.6933,-47.4933]}
        ),
        "Resp_Civil_siniestros_monto": pd.DataFrame(
            {"Variable": ["num_bin__2_o_mas_inquilinos","multi__año_cursado_posgrado","num_bin__en_campus","num_bin__distancia_al_campus","num_bin__calif_promedio","multi__estudios_area_Otro","multi__año_cursado_3er año","multi__genero_Masculino","multi__estudios_area_Ciencias","multi__año_cursado_4to año","num_bin__extintor_incendios","multi__genero_No respuesta","multi__estudios_area_Humanidades","multi__genero_Otro","multi__año_cursado_2do año"],
             "%Cambio_prima": [448.2017,53.9895,23.9554,21.8542,10.3560,7.4916,-29.3219,-29.7620,-32.1020,-38.5175,-38.9797,-47.4722,-51.1230,-52.5578,-66.1921]}
        ),
        "Contenidos_siniestros_monto": pd.DataFrame(
            {"Variable": ["num_bin__2_o_mas_inquilinos","num_bin__en_campus","multi__año_cursado_3er año","multi__año_cursado_posgrado","multi__genero_No respuesta","multi__genero_Otro","multi__genero_Masculino","multi__año_cursado_2do año","num_bin__distancia_al_campus","num_bin__calif_promedio","multi__año_cursado_4to año","multi__estudios_area_Ciencias","multi__estudios_area_Otro","multi__estudios_area_Humanidades","num_bin__extintor_incendios"],
             "%Cambio_prima": [345.1322,119.3666,29.8415,22.5027,18.0567,10.3456,5.1074,2.1588,-1.0780,-2.0532,-2.9586,-7.2140,-8.6788,-26.5477,-29.9581]}
        ),
    }

    cambio_total = pd.DataFrame(
        {
            "Variable": [
                "num_bin__2_o_mas_inquilinos","num_bin__en_campus","multi__año_cursado_posgrado",
                "multi__año_cursado_3er año","multi__genero_No respuesta","multi__año_cursado_4to año",
                "num_bin__distancia_al_campus","multi__estudios_area_Otro","multi__estudios_area_Humanidades",
                "num_bin__calif_promedio","multi__año_cursado_2do año","multi__genero_Otro",
                "multi__genero_Masculino","multi__estudios_area_Ciencias","num_bin__extintor_incendios",
            ],
            "Factor_total": [4.2280,2.3657,1.5125,1.2548,1.1920,1.1190,1.0414,1.0090,0.9932,0.9583,0.9466,0.9205,0.9017,0.8866,0.5566],
            "%Cambio_total": [322.8016,136.5736,51.2544,25.4803,19.1971,11.8959,4.1399,0.9005,-0.6814,-4.1669,-5.3377,-7.9533,-9.8288,-11.3409,-44.3401],
        }
    )

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

def kpi(label: str, value):
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{fmt_float(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ================================
# APP
# ================================
def main():
    st.set_page_config(page_title="Métricas de Prima por Cobertura", page_icon="📊", layout="wide")
    st.markdown(EXECUTIVE_CSS, unsafe_allow_html=True)

    # Header
    top_logo, top_title = st.columns([1, 6])
    with top_logo:
        if LOGO_URL:
            st.image(LOGO_URL, width=64)
    with top_title:
        st.markdown("<h1 class='title-text'>Dashboard de Coberturas y Métricas</h1>", unsafe_allow_html=True)
        st.markdown("<span style='color:#6b7280'>Frecuencia · Severidad · Prima esperada</span>", unsafe_allow_html=True)

    # Datos
    mod = load_remote_module(REMOTE_PY_URL, REMOTE_MODULE_NAME)
    data = try_remote_get_metrics(mod) or get_fallback_data()

    header_metrics: Dict[str, Dict[str, float]] = data["header_metrics"]
    cambio_por_cobertura: Dict[str, pd.DataFrame] = data.get("cambio_por_cobertura", {})
    cambio_total: pd.DataFrame = data.get("cambio_total", pd.DataFrame())

    # FILA SUPERIOR: Izq selector / Der métricas (compacto)
    row1_left, row1_right = st.columns([1.2, 3.8], gap="large")
    with row1_left:
        with st.container(border=True):
            st.markdown("### Cobertura")
            cobertura = st.selectbox(
                "Selecciona cobertura", COBERTURAS, index=0,
                label_visibility="collapsed",
                format_func=lambda s: s.replace("_siniestros_monto", "").replace("_", " ")
            )
    with row1_right:
        with st.container(border=True):
            st.markdown("### Métricas clave")
            metrics = header_metrics.get(cobertura, {})
            g1, g2, g3, g4 = st.columns(4)
            with g1: kpi("Media real de N", metrics.get("Media real de N", np.nan))
            with g2: kpi("Media predicha de N", metrics.get("Media predicha de N", np.nan))
            with g3: kpi("Severidad esperada media (predicha)", metrics.get("Severidad esperada media (predicha)", np.nan))
            with g4: kpi("Severidad real media (observada)", metrics.get("Severidad real media (observada)", np.nan))

    # Excel
    df_all = None
    try:
        df_all = read_excel_from_url(EXCEL_URL)
    except Exception as e:
        st.error(f"No se pudo leer el Excel desde GitHub RAW: {e}")

    if df_all is not None:
        try:
            df_all = build_perfil(df_all)

            # FILA MEDIA: Izq scatter / Der barras selección
            row2_left, row2_right = st.columns([2.5, 2.0], gap="large")

            with row2_left:
                with st.container(border=True):
                    st.markdown("### Mapa de riesgo por cobertura")
                    fig_scatter, df_sample = make_scatter_matplotlib(df_all, cobertura, sample_max=8000)
                    fig_to_stimage(fig_scatter, dpi=220)   # 👈 render robusto
                    plt.close(fig_scatter)

                    st.download_button(
                        label="⬇️ Descargar datos de la muestra graficada (CSV)",
                        data=df_sample.to_csv(index=False).encode("utf-8"),
                        file_name=f"muestra_mapa_riesgo_{cobertura}.csv",
                        mime="text/csv"
                    )

            with row2_right:
                with st.container(border=True):
                    st.markdown("### Cambio porcentual (selección) → Factor")
                    df_cob = cambio_por_cobertura.get(cobertura, pd.DataFrame(columns=["Variable", "%Cambio_prima"])).copy()
                    df_sel = df_cob[df_cob["Variable"].isin(VARS_BIN)].copy()
                    if df_sel.empty:
                        st.info("No hay datos de cambio por variable para esta cobertura.")
                    else:
                        df_sel["Factor"] = (pd.to_numeric(df_sel["%Cambio_prima"], errors="coerce") / 100 + 1).round(4)
                        df_sel = df_sel[["Variable", "Factor"]].sort_values("Variable").reset_index(drop=True)
                        fig_bar_sel = plot_bars(df_sel, x_col="Variable", y_col="Factor",
                                                title="Factor por variable (selección)",
                                                xtick_rotation=20, width=(11.0, 6.2))
                        fig_to_stimage(fig_bar_sel, dpi=220)  # 👈 render robusto
                        plt.close(fig_bar_sel)

            # FILA INFERIOR (más pequeña)
            with st.container(border=True):
                st.markdown("### Cambio de la PRIMA ESPERADA TOTAL (ponderado por prima base)")
                if cambio_total is None or cambio_total.empty:
                    st.info("No hay datos de cambio total disponibles.")
                else:
                    df_total_sel = cambio_total.copy()
                    df_total_sel = df_total_sel[df_total_sel["Variable"].isin(VARS_BIN)].copy()  # solo selección
                    if df_total_sel.empty:
                        st.info("No hay datos para las variables seleccionadas en el cambio total.")
                    else:
                        df_total_sel = df_total_sel[["Variable", "Factor_total"]].sort_values("Variable").reset_index(drop=True)
                        # Figura intencionalmente más baja
                        fig_bar_total = plot_bars(df_total_sel, x_col="Variable", y_col="Factor_total",
                                                  title="Factor total por variable (ponderado por prima base)",
                                                  xtick_rotation=20, width=(10.0, 3.6))
                        fig_to_stimage(fig_bar_total, dpi=220)  # 👈 render robusto
                        plt.close(fig_bar_total)

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
