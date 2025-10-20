# app.py
# -------------------------------------------------------------
# App Streamlit ejecutivo para métricas y coberturas
# con opción de conectar un módulo .py remoto (GitHub RAW)
# -------------------------------------------------------------
# Requisitos sugeridos:
#   pip install streamlit requests pandas numpy
# Ejecutar:  streamlit run app.py
# -------------------------------------------------------------

import io
import os
import sys
import tempfile
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ================================
# CONFIGURACIÓN
# ================================
# 1) URL del .py crudo en GitHub (raw). Cambia esta constante a tu enlace RAW.
#    Ejemplo: "https://raw.githubusercontent.com/usuario/repositorio/rama/carpeta/modulo_modelo.py"
REMOTE_PY_URL = os.getenv("REMOTE_PY_URL", "")  # <-- pon la URL RAW aquí o via variable de entorno
REMOTE_MODULE_NAME = "modelo_remoto"            # nombre interno con el que importaremos el módulo

# 2) Logo opcional de tu organización para el header (URL pública). Deja vacío para ocultar.
LOGO_URL = os.getenv("LOGO_URL", "")

# 3) Nombres canónicos de coberturas (deben coincidir con las claves de los datos)
COBERTURAS = [
    "Gastos_Adicionales_siniestros_monto",
    "Gastos_Medicos_RC_siniestros_monto",
    "Resp_Civil_siniestros_monto",
    "Contenidos_siniestros_monto",
]

# 4) Variables a mostrar en tablas según requerimiento
VARS_BIN = [
    "num_bin__2_o_mas_inquilinos",
    "num_bin__en_campus",
    "num_bin__extintor_incendios",
]

# ================================
# ESTILO (CSS) — look ejecutivo
# ================================
EXECUTIVE_CSS = """
<style>
/***** Tipografía y base *****/
html, body, [class*="css"], .stMarkdown, .stText, .stDataFrame { font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji"; }

/***** Contenedores *****/
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/***** Títulos *****/
h1, .title-text { font-weight: 700; letter-spacing: -0.02em; }

/***** KPI cards (métricas) *****/
.kpi-card { background: #1E3A8A; border: 1px solid rgba(0,0,0,0.06); border-radius: 14px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); } .kpi-card .metric-label { font-size: 0.85rem; margin-bottom: 6px; } .kpi-card .metric-value { font-size: 1.35rem; font-weight: 700;

/***** Tablas *****/
caption { color: #6b7280 !important; text-transform: uppercase; letter-spacing: .03em; font-size: .78rem; }

/***** Selector pegable (sticky) *****/
.sticky { position: sticky; top: 0.5rem; z-index: 999; }

</style>
"""

# -------------------------------------------------------------
# Utilidad: cargar un .py remoto (raw GitHub) y convertirlo en módulo importable
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_remote_module(raw_url: str, module_name: str):
    """Descarga un .py desde raw_url y lo importa como módulo con nombre module_name.
    Retorna el módulo o None si no fue posible cargarlo.
    """
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
# Fallback de datos (del mensaje del usuario) en caso de que el módulo remoto no provea funciones
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
            {
                "Variable": [
                    "num_bin__2_o_mas_inquilinos",
                    "num_bin__en_campus",
                    "multi__genero_No respuesta",
                    "multi__año_cursado_4to año",
                    "multi__año_cursado_posgrado",
                    "multi__año_cursado_3er año",
                    "multi__estudios_area_Otro",
                    "multi__genero_Masculino",
                    "num_bin__distancia_al_campus",
                    "multi__estudios_area_Humanidades",
                    "num_bin__calif_promedio",
                    "multi__estudios_area_Ciencias",
                    "multi__año_cursado_2do año",
                    "multi__genero_Otro",
                    "num_bin__extintor_incendios",
                ],
                "%Cambio_prima": [
                    354.8370,
                    94.7330,
                    46.7106,
                    42.9091,
                    36.2409,
                    11.1749,
                    10.2217,
                    1.7919,
                    0.4819,
                    -2.8914,
                    -2.9553,
                    -5.3370,
                    -12.7520,
                    -17.4318,
                    -46.2605,
                ],
            }
        ),
        "Gastos_Medicos_RC_siniestros_monto": pd.DataFrame(
            {
                "Variable": [
                    "num_bin__2_o_mas_inquilinos",
                    "num_bin__en_campus",
                    "multi__año_cursado_posgrado",
                    "multi__año_cursado_3er año",
                    "multi__estudios_area_Humanidades",
                    "multi__genero_No respuesta",
                    "multi__año_cursado_2do año",
                    "num_bin__distancia_al_campus",
                    "multi__año_cursado_4to año",
                    "multi__genero_Otro",
                    "multi__estudios_area_Otro",
                    "num_bin__calif_promedio",
                    "multi__estudios_area_Ciencias",
                    "multi__genero_Masculino",
                    "num_bin__extintor_incendios",
                ],
                "%Cambio_prima": [
                    275.3948,
                    187.5584,
                    68.6651,
                    42.8082,
                    15.2331,
                    10.5232,
                    7.3051,
                    5.3001,
                    1.7714,
                    0.8731,
                    -4.1862,
                    -7.7901,
                    -13.3933,
                    -18.6933,
                    -47.4933,
                ],
            }
        ),
        "Resp_Civil_siniestros_monto": pd.DataFrame(
            {
                "Variable": [
                    "num_bin__2_o_mas_inquilinos",
                    "multi__año_cursado_posgrado",
                    "num_bin__en_campus",
                    "num_bin__distancia_al_campus",
                    "num_bin__calif_promedio",
                    "multi__estudios_area_Otro",
                    "multi__año_cursado_3er año",
                    "multi__genero_Masculino",
                    "multi__estudios_area_Ciencias",
                    "multi__año_cursado_4to año",
                    "num_bin__extintor_incendios",
                    "multi__genero_No respuesta",
                    "multi__estudios_area_Humanidades",
                    "multi__genero_Otro",
                    "multi__año_cursado_2do año",
                ],
                "%Cambio_prima": [
                    448.2017,
                    53.9895,
                    23.9554,
                    21.8542,
                    10.3560,
                    7.4916,
                    -29.3219,
                    -29.7620,
                    -32.1020,
                    -38.5175,
                    -38.9797,
                    -47.4722,
                    -51.1230,
                    -52.5578,
                    -66.1921,
                ],
            }
        ),
        "Contenidos_siniestros_monto": pd.DataFrame(
            {
                "Variable": [
                    "num_bin__2_o_mas_inquilinos",
                    "num_bin__en_campus",
                    "multi__año_cursado_3er año",
                    "multi__año_cursado_posgrado",
                    "multi__genero_No respuesta",
                    "multi__genero_Otro",
                    "multi__genero_Masculino",
                    "multi__año_cursado_2do año",
                    "num_bin__distancia_al_campus",
                    "num_bin__calif_promedio",
                    "multi__año_cursado_4to año",
                    "multi__estudios_area_Ciencias",
                    "multi__estudios_area_Otro",
                    "multi__estudios_area_Humanidades",
                    "num_bin__extintor_incendios",
                ],
                "%Cambio_prima": [
                    345.1322,
                    119.3666,
                    29.8415,
                    22.5027,
                    18.0567,
                    10.3456,
                    5.1074,
                    2.1588,
                    -1.0780,
                    -2.0532,
                    -2.9586,
                    -7.2140,
                    -8.6788,
                    -26.5477,
                    -29.9581,
                ],
            }
        ),
    }

    cambio_total = pd.DataFrame(
        {
            "Variable": [
                "num_bin__2_o_mas_inquilinos",
                "num_bin__en_campus",
                "multi__año_cursado_posgrado",
                "multi__año_cursado_3er año",
                "multi__genero_No respuesta",
                "multi__año_cursado_4to año",
                "num_bin__distancia_al_campus",
                "multi__estudios_area_Otro",
                "multi__estudios_area_Humanidades",
                "num_bin__calif_promedio",
                "multi__año_cursado_2do año",
                "multi__genero_Otro",
                "multi__genero_Masculino",
                "multi__estudios_area_Ciencias",
                "num_bin__extintor_incendios",
            ],
            "Factor_total": [
                4.2280,
                2.3657,
                1.5125,
                1.2548,
                1.1920,
                1.1190,
                1.0414,
                1.0090,
                0.9932,
                0.9583,
                0.9466,
                0.9205,
                0.9017,
                0.8866,
                0.5566,
            ],
            "%Cambio_total": [
                322.8016,
                136.5736,
                51.2544,
                25.4803,
                19.1971,
                11.8959,
                4.1399,
                0.9005,
                -0.6814,
                -4.1669,
                -5.3377,
                -7.9533,
                -9.8288,
                -11.3409,
                -44.3401,
            ],
        }
    )

    return {
        "header_metrics": header_metrics,
        "cambio_por_cobertura": cambio_por_cobertura,
        "cambio_total": cambio_total,
    }

# -------------------------------------------------------------
# Funciones de acceso (módulo remoto o fallback)
# -------------------------------------------------------------

def try_remote_get_metrics(mod) -> Optional[Dict[str, Any]]:
    """Si el módulo remoto expone get_metrics() retorna su resultado.
    Espera un dict con claves: 'header_metrics', 'cambio_por_cobertura', 'cambio_total'.
    """
    try:
        if mod and hasattr(mod, "get_metrics"):
            data = mod.get_metrics()  # el módulo del usuario debe implementarlo
            return data
    except Exception as e:
        st.warning(f"Fallo get_metrics() del módulo remoto: {e}")
    return None

# -------------------------------------------------------------
# UI HELPERS
# -------------------------------------------------------------

def fmt_float(x, nd=4):
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return x


def kpi(label: str, value):
    """Tarjeta KPI con estilo consistente."""
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{fmt_float(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_small_table(df: pd.DataFrame, caption: str):
    df2 = df.copy()
    # Dar formato a columnas con %
    for col in df2.columns:
        if "%" in col:
            df2[col] = df2[col].apply(lambda v: float(v))
    st.caption(caption)
    st.dataframe(
        df2,
        use_container_width=True,
        hide_index=True,
        column_config={
            "%Cambio_prima": st.column_config.NumberColumn("%Cambio prima", format="%.4f"),
            "%Cambio_total": st.column_config.NumberColumn("%Cambio total", format="%.4f"),
            "Factor_total": st.column_config.NumberColumn("Factor total", format="%.4f"),
        },
    )


# ================================
# APP
# ================================

def main():
    st.set_page_config(page_title="Métricas de Prima por Cobertura", page_icon="📊", layout="wide")
    st.markdown(EXECUTIVE_CSS, unsafe_allow_html=True)

    # Header ejecutivo con logo y título
    hcol1, hcol2 = st.columns([1, 6])
    with hcol1:
        if LOGO_URL:
            st.image(LOGO_URL, width=72)
    with hcol2:
        st.markdown("<h1 class='title-text'>Dashboard de Coberturas y Métricas</h1>", unsafe_allow_html=True)
        st.markdown("<span style='color:#6b7280'>Frecuencia · Severidad · Prima esperada</span>", unsafe_allow_html=True)

    # Cargar módulo remoto si se configuró REMOTE_PY_URL
    mod = load_remote_module(REMOTE_PY_URL, REMOTE_MODULE_NAME)

    # Datos: intento módulo remoto -> fallback
    data = try_remote_get_metrics(mod)
    if data is None:
        data = get_fallback_data()

    header_metrics: Dict[str, Dict[str, float]] = data["header_metrics"]
    cambio_por_cobertura: Dict[str, pd.DataFrame] = data["cambio_por_cobertura"]
    cambio_total: pd.DataFrame = data["cambio_total"]

    # =====================
    # LAYOUT SUPERIOR: selector (20%) | panel (80%)
    # =====================
    colL, colR = st.columns([1, 4], gap="large")

    with colL:
        with st.container(border=True):
            st.markdown("### Cobertura")
            cobertura = st.selectbox(
                "Selecciona cobertura",
                COBERTURAS,
                index=0,
                format_func=lambda s: s.replace("_siniestros_monto", "").replace("_", " ")
            )

    with colR:
        with st.container(border=True):
            st.markdown("### Métricas clave")
            metrics = header_metrics.get(cobertura, {})
            g1, g2, g3, g4 = st.columns(4)
            with g1: kpi("Media real de N", metrics.get("Media real de N", np.nan))
            with g2: kpi("Media predicha de N", metrics.get("Media predicha de N", np.nan))
            with g3: kpi("Severidad esperada media (predicha)", metrics.get("Severidad esperada media (predicha)", np.nan))
            with g4: kpi("Severidad real media (observada)", metrics.get("Severidad real media (observada)", np.nan))

            st.markdown("---")
            df_cob = cambio_por_cobertura.get(cobertura, pd.DataFrame(columns=["Variable", "%Cambio_prima"]))
            tabla_vars = df_cob[df_cob["Variable"].isin(VARS_BIN)].copy()
            tabla_vars["Factor"] = (pd.to_numeric(tabla_vars["%Cambio_prima"], errors="coerce") / 100 + 1).round(4)
            tabla_vars = tabla_vars.sort_values("Variable").reset_index(drop=True)
            tabla_vars = tabla_vars[["Variable", "%Cambio_prima", "Factor"]]
            render_small_table(tabla_vars, "Cambio porcentual de la PRIMA ESPERADA por variable (selección)"column_config={
                "%Cambio_prima": st.column_config.NumberColumn("%Cambio prima", format="%.4f"),
                "%Cambio_total": st.column_config.NumberColumn("%Cambio total", format="%.4f"),
                "Factor_total": st.column_config.NumberColumn("Factor total", format="%.4f"),
                "Factor": st.column_config.NumberColumn("Factor", format="%.4f"),  # <-- añade esto
            },
            )

    # =====================
    # LAYOUT SUPERIOR CENTRAL: tabla con 3 variables desde %Cambio_total
    # =====================
    with st.container(border=True):
        st.markdown("### Impacto total ponderado por prima base (variables seleccionadas)")
        tabla_total = cambio_total[cambio_total["Variable"].isin(VARS_BIN)].copy()
        tabla_total = tabla_total[["Variable", "Factor_total", "%Cambio_total"]].sort_values("Variable").reset_index(drop=True)
        render_small_table(tabla_total, "Cambio de la PRIMA ESPERADA TOTAL (ponderado por prima base)")

    # =====================
    # Descarga de datos
    # =====================
    with st.expander("Descargar tablas como CSV"):
        colA, colB = st.columns(2)
        with colA:
            if 'tabla_vars' in locals() and not tabla_vars.empty:
                st.download_button(
                    label="Descargar selección por cobertura (CSV)",
                    data=tabla_vars.to_csv(index=False).encode('utf-8'),
                    file_name=f"cambio_por_variable_{cobertura}.csv",
                    mime="text/csv",
                )
        with colB:
            st.download_button(
                label="Descargar impacto total (CSV)",
                data=tabla_total.to_csv(index=False).encode('utf-8'),
                file_name="cambio_total_seleccion.csv",
                mime="text/csv",
            )

    # =====================
    # Pie de página
    # =====================
    st.markdown("""
    <div style='margin-top:1rem;color:#9ca3af;font-size:0.85rem'>
      © {year} — Métricas de suscripción · Actuarial Analytics
    </div>
    """.format(year=pd.Timestamp.today().year), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
