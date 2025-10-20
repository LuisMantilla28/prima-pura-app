# app.py
# -------------------------------------------------------------
# App Streamlit para visualizar métricas por cobertura
# y conectarse (opcionalmente) a un módulo remoto .py en GitHub
# -------------------------------------------------------------
# Requisitos sugeridos en el entorno:
#   pip install streamlit requests pandas numpy
# Ejecutar localmente:  streamlit run app.py
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

# 2) Nombres canónicos de coberturas (las claves deben coincidir con lo que devuelva el módulo remoto)
COBERTURAS = [
    "Gastos_Adicionales_siniestros_monto",
    "Gastos_Medicos_RC_siniestros_monto",
    "Resp_Civil_siniestros_monto",
    "Contenidos_siniestros_monto",
]

# 3) Variables a mostrar en tablas según el requerimiento
VARS_BIN = [
    "num_bin__2_o_mas_inquilinos",
    "num_bin__en_campus",
    "num_bin__extintor_incendios",
]

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
    # Métricas de la cabecera por cobertura
    header_metrics = {
        "Gastos_Adicionales_siniestros_monto": {
            "Media real de N": 0.055625,
            "Media E[N]": 0.05318535298803486,
            "Severidad esperada media (predicha)": 5425.88,
            "Severidad real media (observada)": 5395.85,
        },
        "Gastos_Medicos_RC_siniestros_monto": {
            "Media real de N": 0.021875,
            "Media E[N]": 0.023721117738488857,
            "Severidad esperada media (predicha)": 17170.89,
            "Severidad real media (observada)": 15154.66,
        },
        "Resp_Civil_siniestros_monto": {
            "Media real de N": 0.009375,
            "Media E[N]": 0.008137502924526774,
            "Severidad esperada media (predicha)": 7629.31,
            "Severidad real media (observada)": 9311.54,
        },
        "Contenidos_siniestros_monto": {
            "Media real de N": 0.103125,
            "Media E[N]": 0.09827778812479611,
            "Severidad esperada media (predicha)": 987.06,
            "Severidad real media (observada)": 971.40,
        },
    }

    # %Cambio de la PRIMA ESPERADA por variable (individual por cobertura)
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

    # %Cambio de la PRIMA ESPERADA TOTAL (ponderado por prima base)
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


def top_header_panel(metrics: Dict[str, float]):
    """Render de la cabecera (80% derecha):
    - Media real de N, Media E[N]
    - Severidad esperada media (predicha), Severidad real media (observada)
    """
    st.subheader("Métricas clave")
    cols = st.columns(4, gap="small")
    labels = [
        "Media real de N",
        "Media E[N]",
        "Severidad esperada media (predicha)",
        "Severidad real media (observada)",
    ]
    for i, lab in enumerate(labels):
        val = metrics.get(lab, np.nan)
        cols[i].metric(lab, fmt_float(val))


def render_small_table(df: pd.DataFrame, caption: str):
    df2 = df.copy()
    # si existe columna de %
    for col in df2.columns:
        if "%" in col:
            df2[col] = df2[col].apply(lambda v: fmt_float(v, 4))
    st.caption(caption)
    st.dataframe(df2, use_container_width=True, hide_index=True)


# ================================
# APP
# ================================

def main():
    st.set_page_config(page_title="Métricas de Prima por Cobertura", layout="wide")
    st.title("Dashboard de Coberturas y Métricas")

    # Cargar módulo remoto si se configuró REMOTE_PY_URL
    mod = load_remote_module(REMOTE_PY_URL, REMOTE_MODULE_NAME)

    # Datos: intento módulo remoto -> fallback
    data = try_remote_get_metrics(mod)
    if data is None:
        data = get_fallback_data()
        st.info(
            "Usando datos de ejemplo embebidos. Si agregas un módulo remoto con `get_metrics()`, "
            "el dashboard consumirá tus cifras reales."
        )

    header_metrics: Dict[str, Dict[str, float]] = data["header_metrics"]
    cambio_por_cobertura: Dict[str, pd.DataFrame] = data["cambio_por_cobertura"]
    cambio_total: pd.DataFrame = data["cambio_total"]

    # =====================
    # LAYOUT SUPERIOR: selector (20%) | panel (80%)
    # =====================
    colL, colR = st.columns([1, 4], gap="large")

    with colL:
        st.subheader("Cobertura")
        cobertura = st.selectbox(
            "Selecciona cobertura",
            COBERTURAS,
            index=0,
            format_func=lambda s: s.replace("_siniestros_monto", "").replace("_", " ")
        )

    with colR:
        # 1) Métricas de la cabecera
        metrics = header_metrics.get(cobertura, {})
        top_header_panel(metrics)

        # 2) Tabla con 3 variables (Cambio % de la PRIMA ESPERADA por variable)
        df_cob = cambio_por_cobertura.get(cobertura, pd.DataFrame(columns=["Variable", "%Cambio_prima"]))
        tabla_vars = df_cob[df_cob["Variable"].isin(VARS_BIN)].copy()
        tabla_vars = tabla_vars.sort_values("Variable").reset_index(drop=True)
        render_small_table(tabla_vars, "Cambio porcentual de la PRIMA ESPERADA por variable (selección)")

    # =====================
    # LAYOUT SUPERIOR CENTRAL: tabla con 3 variables desde %Cambio_total
    # =====================
    st.markdown("---")
    st.subheader("Impacto total ponderado por prima base (variables seleccionadas)")
    tabla_total = cambio_total[cambio_total["Variable"].isin(VARS_BIN)].copy()
    tabla_total = tabla_total[["Variable", "Factor_total", "%Cambio_total"]].sort_values("Variable").reset_index(drop=True)
    render_small_table(tabla_total, "Cambio de la PRIMA ESPERADA TOTAL (ponderado por prima base)")

    # =====================
    # Información de la fuente remota
    # =====================
    with st.expander("Conexión al módulo remoto (.py)"):
        st.write(
            """
            **Cómo usar el módulo remoto**

            1. Sube a GitHub (en modo público o con raw accesible) un archivo `.py` que
               exponga la función `get_metrics()` devolviendo un `dict` con las claves:
               - `header_metrics`: `dict[str, dict[str, float]]`
               - `cambio_por_cobertura`: `dict[str, pd.DataFrame]` (o lista de dicts convertible a DataFrame)
               - `cambio_total`: `pd.DataFrame`

            2. Obtén la URL **RAW** de ese `.py` y colócala en la variable de entorno `REMOTE_PY_URL`
               o modifica la constante `REMOTE_PY_URL` arriba.

            3. Estructura de ejemplo para `get_metrics()`:

            ```python
            import pandas as pd

            def get_metrics():
                header_metrics = {
                    "Gastos_Adicionales_siniestros_monto": {
                        "Media real de N": 0.0556,
                        "Media E[N]": 0.0532,
                        "Severidad esperada media (predicha)": 5425.88,
                        "Severidad real media (observada)": 5395.85,
                    },
                    # ... resto de coberturas ...
                }

                cambio_por_cobertura = {
                    "Gastos_Adicionales_siniestros_monto": pd.DataFrame(
                        {"Variable": ["num_bin__2_o_mas_inquilinos", ...], "%Cambio_prima": [354.83, ...]}
                    ),
                    # ... resto ...
                }

                cambio_total = pd.DataFrame({
                    "Variable": ["num_bin__2_o_mas_inquilinos", ...],
                    "Factor_total": [4.2280, ...],
                    "%Cambio_total": [322.8016, ...],
                })

                return {
                    "header_metrics": header_metrics,
                    "cambio_por_cobertura": cambio_por_cobertura,
                    "cambio_total": cambio_total,
                }
            ```
            """
        )


if __name__ == "__main__":
    main()
