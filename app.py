"""PAVEMENT AI · SENATI Huánuco · PIAD 530
Predicción del Índice de Condición del Pavimento (PCI) con Regresión Lineal.
UI/UX: Neumorfismo + Glassmorfismo sobre Streamlit.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ==================== CONFIGURACIÓN ====================
st.set_page_config(
    page_title="PAVEMENT AI · SENATI Huánuco",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, "dataset_pavimento.csv")
REGISTROS = os.path.join(BASE_DIR, "registros_inspeccion.csv")
FEATURES = ["antiguedad", "transito_esa", "grietas_long", "grietas_trans"]

# ==================== CSS: NEUMORFISMO + GLASSMORFISMO ====================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root{
  --bg1:#0b1626; --bg2:#13243b; --accent:#73d8ff; --accent2:#8b9bff;
  --text:#e6eef7; --muted:#9db0c6;
  --neo-light:#22395a; --neo-dark:#060d18;
  --glass:rgba(255,255,255,.07); --glass-border:rgba(255,255,255,.16);
}
html, body, [class*="css"], .stApp { font-family:'Inter',sans-serif; }

/* Fondo con "manchas" de color para que el vidrio tenga qué desenfocar */
.stApp{
  background:
    radial-gradient(circle at 12% 8%,  rgba(56,189,248,.28), transparent 38%),
    radial-gradient(circle at 88% 18%, rgba(139,92,246,.26), transparent 40%),
    radial-gradient(circle at 55% 95%, rgba(20,184,166,.20), transparent 45%),
    linear-gradient(135deg,var(--bg1),var(--bg2) 55%,#0a1422);
  background-attachment:fixed;
  color:var(--text);
}
header[data-testid="stHeader"]{ background:transparent; }
.block-container{ padding-top:2rem; max-width:1250px; }

/* ---------- GLASSMORFISMO: tarjetas (st.container con borde) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--glass);
  backdrop-filter:blur(18px) saturate(140%);
  -webkit-backdrop-filter:blur(18px) saturate(140%);
  border:1px solid var(--glass-border) !important;
  border-radius:22px !important;
  box-shadow:0 10px 36px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.12);
  padding:.6rem .9rem;
}
.hero{
  background:linear-gradient(135deg,rgba(115,216,255,.16),rgba(139,155,255,.10));
  backdrop-filter:blur(18px); -webkit-backdrop-filter:blur(18px);
  border:1px solid var(--glass-border); border-radius:26px;
  padding:28px 32px; margin-bottom:18px;
  box-shadow:0 12px 40px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.14);
}
.hero h1{ margin:0; font-size:2rem; }
.hero p{ margin:6px 0 0; color:var(--muted); }

/* ---------- NEUMORFISMO: métricas en relieve ---------- */
.metric-box{
  background:linear-gradient(145deg,#1b3050,#12233b);
  border-radius:20px; padding:18px 14px; text-align:center;
  box-shadow:9px 9px 18px var(--neo-dark), -9px -9px 18px var(--neo-light);
  transition:transform .25s ease, box-shadow .25s ease;
}
.metric-box:hover{ transform:translateY(-3px); }
.metric-label{ font-size:.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:1.4px; }
.metric-value{ font-size:2.1rem; font-weight:700; margin-top:6px; color:var(--accent); }

/* Botones en relieve → hundidos al presionar */
.stButton > button, .stFormSubmitButton > button{
  background:linear-gradient(145deg,#1d3556,#12233b) !important;
  color:var(--accent) !important; font-weight:600; letter-spacing:.3px;
  border:none !important; border-radius:14px !important; padding:.65rem 1.4rem;
  box-shadow:6px 6px 12px var(--neo-dark), -6px -6px 12px var(--neo-light);
  transition:all .25s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover{ color:#fff !important; }
.stButton > button:active, .stFormSubmitButton > button:active{
  box-shadow:inset 5px 5px 10px var(--neo-dark), inset -5px -5px 10px var(--neo-light);
}

/* Campos hundidos (efecto "inset") */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
div[data-baseweb="select"] > div{
  background:#0f1d33 !important; color:var(--text) !important;
  border:none !important; border-radius:14px !important;
  box-shadow:inset 4px 4px 9px var(--neo-dark), inset -4px -4px 9px #1a2f4d !important;
}
div[data-baseweb="input"], div[data-baseweb="base-input"]{ background:transparent !important; border:none !important; }

/* Sliders */
div[data-testid="stSlider"] [role="slider"]{
  background:var(--accent) !important; box-shadow:0 0 0 5px rgba(115,216,255,.18);
}

/* Sidebar de vidrio */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,rgba(13,24,38,.82),rgba(21,37,51,.78));
  backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
  border-right:1px solid var(--glass-border);
}
section[data-testid="stSidebar"] label[data-baseweb="radio"]{
  background:rgba(255,255,255,.04); border-radius:14px; padding:10px 14px; margin-bottom:8px;
  transition:all .25s ease; width:100%;
}
section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover{
  box-shadow:5px 5px 10px var(--neo-dark), -5px -5px 10px var(--neo-light);
}

h1,h2,h3{ color:var(--accent) !important; font-weight:600 !important; }
.stAlert{ background:var(--glass) !important; border:1px solid rgba(115,216,255,.28) !important; border-radius:16px !important; }
.road-bar{
  height:8px; border-radius:4px; margin:14px 0 22px;
  background:repeating-linear-gradient(90deg,#5a6e84 0 18%,#1d2c3d 18% 20%);
  box-shadow:inset 1px 1px 3px rgba(0,0,0,.5);
}
.badge{
  display:inline-block; padding:4px 14px; border-radius:999px; font-weight:600; font-size:.85rem;
  background:rgba(255,255,255,.08); border:1px solid var(--glass-border);
}
</style>
""",
    unsafe_allow_html=True,
)

# ==================== DATOS ====================
@st.cache_data
def load_data(_version: float = 0.0) -> pd.DataFrame:
    """Dataset base + inspecciones registradas desde la app."""
    base = pd.read_csv(DATASET)
    if os.path.exists(REGISTROS):
        base = pd.concat([base, pd.read_csv(REGISTROS)], ignore_index=True)
    base["fecha"] = pd.to_datetime(base["fecha"])
    return base


version = os.path.getmtime(REGISTROS) if os.path.exists(REGISTROS) else 0.0
df = load_data(version)


# ==================== MODELO ====================
@st.cache_resource
def train_model(data: pd.DataFrame):
    X = data[FEATURES].values
    y = data["pci_observado"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LinearRegression().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "r2": r2_score(y_te, pred),
        "mae": mean_absolute_error(y_te, pred),
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
    }
    return model, metrics, X_te, y_te, pred


model, metrics, X_test, y_test, y_pred = train_model(df)


# ==================== AUXILIARES ====================
def clasificar_pci(pci):
    if pci >= 85: return "Bueno", "#22c55e"
    if pci >= 70: return "Satisfactorio", "#84cc16"
    if pci >= 55: return "Regular", "#eab308"
    if pci >= 40: return "Pobre", "#f97316"
    if pci >= 25: return "Muy pobre", "#ef4444"
    if pci >= 10: return "Grave", "#dc2626"
    return "Fallado", "#7f1d1d"


def neo_metric(label, value, color="#73d8ff"):
    st.markdown(
        f"""<div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div></div>""",
        unsafe_allow_html=True,
    )


def estilo_plot(fig, **kw):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6eef7", family="Inter"),
        margin=dict(l=10, r=10, t=40, b=10), **kw,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.08)")
    return fig


def hero(titulo, subtitulo):
    st.markdown(
        f"<div class='hero'><h1>{titulo}</h1><p>{subtitulo}</p></div>",
        unsafe_allow_html=True,
    )


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown(
        """<div style="text-align:center;padding:18px 0 6px;">
        <h2 style="margin:0;">🛣️ PAVEMENT AI</h2>
        <p style="color:#73d8ff;margin:4px 0;">SENATI · Huánuco</p>
        <p style="font-size:.8rem;color:#8aa0b8;">Predicción de deterioro de pavimento rígido</p></div>""",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    pagina = st.radio(
        "Navegación",
        ["🏠 Dashboard", "📝 Registrar Patología", "🔮 Predicción PCI",
         "📊 Análisis del Modelo"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Datos simulados para fines educativos")
    st.caption(f"Registros: {len(df)} · Modelo: Regresión Lineal")

# ==================== PÁGINAS ====================
if pagina == "🏠 Dashboard":
    hero("Centro de Análisis Vial", "Resumen del estado de las vías de concreto de Huánuco")
    st.markdown("<div class='road-bar'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: neo_metric("Registros", len(df))
    with c2: neo_metric("PCI promedio", f"{df['pci_observado'].mean():.1f}")
    with c3: neo_metric("R² del modelo", f"{metrics['r2']:.3f}")
    with c4: neo_metric("MAE", f"{metrics['mae']:.2f}")
    st.markdown("<br>", unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        with st.container(border=True):
            st.subheader("Distribución del PCI observado")
            fig = px.histogram(df, x="pci_observado", nbins=15, color_discrete_sequence=["#73d8ff"])
            st.plotly_chart(estilo_plot(fig), use_container_width=True)
    with cb:
        with st.container(border=True):
            st.subheader("PCI vs antigüedad")
            fig2 = px.scatter(df, x="antiguedad", y="pci_observado", color="severidad",
                              color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(estilo_plot(fig2), use_container_width=True)

    with st.container(border=True):
        st.subheader("Últimos registros")
        st.dataframe(df.tail(8), use_container_width=True, hide_index=True)

elif pagina == "📝 Registrar Patología":
    hero("Registro de patologías", "Ingresa los datos de la inspección de un tramo")
    with st.container(border=True):
        with st.form("registro"):
            c1, c2 = st.columns(2)
            with c1:
                tramo = st.text_input("Identificación del tramo", "TR-XX")
                antiguedad = st.number_input("Antigüedad (años)", 1, 20, 5)
                transito = st.number_input("Tránsito (ESA)", 100, 10000, 2000, step=100)
            with c2:
                grietas_l = st.number_input("Grietas longitudinales (m)", 0, 100, 10)
                grietas_t = st.number_input("Grietas transversales (m)", 0, 50, 5)
                severidad = st.selectbox("Severidad general", ["Baja", "Media", "Alta", "Muy Alta"])

            patologias = st.multiselect(
                "Patologías observadas",
                ["Grietas longitudinales", "Grietas transversales", "Grietas de esquina",
                 "Losa dividida", "Despostillamiento", "Escalonamiento", "Fisuras", "Descascaramiento"],
            )
            observaciones = st.text_area("Observaciones adicionales")
            pci_manual = st.number_input("PCI observado (evaluación técnica)", 0.0, 100.0, 70.0, step=0.5)
            enviado = st.form_submit_button("Registrar inspección", use_container_width=True)

        if enviado:
            nuevo = {
                "antiguedad": antiguedad, "transito_esa": transito,
                "grietas_long": grietas_l, "grietas_trans": grietas_t,
                "severidad": severidad, "pci_observado": pci_manual,
                "tramo": tramo, "fecha": datetime.now().strftime("%Y-%m-%d"),
                "patologias": ", ".join(patologias), "observaciones": observaciones,
            }
            fila = pd.DataFrame([nuevo])
            fila.to_csv(REGISTROS, mode="a", header=not os.path.exists(REGISTROS), index=False)
            load_data.clear()
            train_model.clear()
            estado, color = clasificar_pci(pci_manual)
            st.success(f"Inspección guardada para el tramo {tramo} en registros_inspeccion.csv")
            st.markdown(f"Condición: <span class='badge' style='color:{color}'>{estado}</span>",
                        unsafe_allow_html=True)
            st.json(nuevo)

elif pagina == "🔮 Predicción PCI":
    hero("Simulador de predicción PCI", "Estima la condición actual y futura de un tramo")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            ant = st.slider("Antigüedad actual (años)", 1, 15, 6)
            trans = st.slider("Tránsito (ESA)", 500, 5000, 2200, step=100)
        with c2:
            gl = st.slider("Grietas longitudinales (m)", 0, 60, 15)
            gt = st.slider("Grietas transversales (m)", 0, 30, 8)
        horizonte = st.slider("Horizonte de predicción (años adicionales)", 1, 8, 3)

    pci_actual = float(np.clip(model.predict(np.array([[ant, trans, gl, gt]]))[0], 0, 100))
    factor = 1 + horizonte * 0.08  # supuesto: crecimiento proporcional de cargas y grietas
    pci_futuro = float(np.clip(
        model.predict(np.array([[ant + horizonte, trans * factor, gl * factor, gt * factor]]))[0], 0, 100))
    est_act, col_act = clasificar_pci(pci_actual)
    est_fut, col_fut = clasificar_pci(pci_futuro)

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1: neo_metric("PCI actual estimado", f"{pci_actual:.1f}", col_act)
    with m2: neo_metric(f"PCI en {horizonte} años", f"{pci_futuro:.1f}", col_fut)
    with m3: neo_metric("Condición futura", est_fut, col_fut)
    st.markdown("<br>", unsafe_allow_html=True)

    anios = list(range(ant, ant + horizonte + 1))
    evol = []
    for a in anios:
        f = 1 + (a - ant) * 0.08
        evol.append(float(np.clip(model.predict(np.array([[a, trans * f, gl * f, gt * f]]))[0], 0, 100)))

    with st.container(border=True):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=anios, y=evol, mode="lines+markers", name="PCI estimado",
                                 line=dict(color="#73d8ff", width=3), marker=dict(size=9)))
        fig.add_hline(y=70, line_dash="dash", line_color="#eab308", annotation_text="Umbral satisfactorio")
        fig.add_hline(y=55, line_dash="dash", line_color="#f97316", annotation_text="Umbral regular")
        estilo_plot(fig, title="Evolución estimada del PCI", xaxis_title="Años",
                    yaxis_title="PCI", yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True)

    if pci_futuro < 55:
        st.info("📅 Recomendación: programar inspección y mantenimiento antes del horizonte indicado.")
    st.warning("⚠️ Datos simulados. El modelo no reemplaza una inspección técnica ni la metodología PCI oficial.")

elif pagina == "📊 Análisis del Modelo":
    hero("Evaluación del modelo", "Métricas sobre el 25 % de datos reservado para prueba")
    c1, c2, c3 = st.columns(3)
    with c1: neo_metric("R²", f"{metrics['r2']:.4f}")
    with c2: neo_metric("MAE", f"{metrics['mae']:.2f}")
    with c3: neo_metric("RMSE", f"{metrics['rmse']:.2f}")
    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Predicciones vs valores reales (conjunto de prueba)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name="Predicciones",
                                 marker=dict(color="#73d8ff", size=10)))
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Ideal",
                                 line=dict(color="#ffc45c", dash="dash")))
        estilo_plot(fig, xaxis_title="PCI real", yaxis_title="PCI predicho")
        st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.subheader("Coeficientes del modelo")
        coefs = pd.DataFrame({
            "Variable": ["Antigüedad", "Tránsito (ESA)", "Grietas long.", "Grietas trans."],
            "Coeficiente": model.coef_,
            "Interpretación": ["Años de vida del pavimento", "Carga acumulada equivalente",
                               "Longitud de grietas longitudinales", "Longitud de grietas transversales"],
        })
        st.dataframe(coefs.style.format({"Coeficiente": "{:.4f}"}), use_container_width=True, hide_index=True)
        st.markdown(f"**Intercepto:** {model.intercept_:.2f}")

