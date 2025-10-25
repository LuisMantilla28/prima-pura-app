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
import requests
import streamlit as st
from datetime import datetime

import plotly.express as px

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
# PLOT: Scatter Plotly (responsive, más pequeño)
# -------------------------------------------------------------
def scatter_plotly(df: pd.DataFrame, cobertu_
