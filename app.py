"""FerroVision - industrial reliability analytics dashboard."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Setting up paths for datasets, trained models and evaluation results
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RUL_MODELS = ROOT / "models" / "rul"
VISION_MODELS = ROOT / "models" / "vision"
METRICS = ROOT / "outputs" / "metrics"
TRAIN = RAW / "train_data.csv"
TEST = RAW / "test_data.csv"

# Configuring the Streamlit dashboard
st.set_page_config(page_title="FerroVision",
                   page_icon="◈",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Custom CSS styling for the dashboard
st.markdown("""
<style>
:root { --ink:#101828; --muted:#667085; --line:#e4e7ec; --violet:#6d5dfc; --cyan:#23b9e6; --paper:#f8faff; }
.stApp { background: radial-gradient(circle at 76% -10%, #e9e7ff 0, transparent 29%), linear-gradient(135deg,#fbfcff 0%,#f4f7ff 100%); color:var(--ink); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#151b38,#20285a); }
[data-testid="stSidebar"] * { color:#f8f9ff !important; }
[data-testid="stSidebar"] .stRadio label { border-radius:12px; padding:6px 9px; }
.hero { padding:1.6rem 0 .7rem; } .eyebrow { color:#6558e8; font-size:.78rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
.hero h1 { font-size:3rem; margin:.15rem 0 .35rem; letter-spacing:-.06em; color:#101828; } .hero p { color:#667085; font-size:1.05rem; max-width:760px; }
.card { background:rgba(255,255,255,.82); border:1px solid #e7e9f3; border-radius:18px; padding:1.1rem 1.15rem; box-shadow:0 8px 28px rgba(26,36,80,.06); }
.metric-label { color:#667085; font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; } .metric-value { color:#101828; font-size:1.85rem; font-weight:800; margin-top:.2rem; }
.stButton>button { background:linear-gradient(100deg,#6558e8,#4e9df6); color:white; border:0; border-radius:10px; font-weight:700; padding:.55rem 1.1rem; }
[data-testid="stFileUploader"] { background:rgba(255,255,255,.8); border:1px dashed #9ba4d1; border-radius:14px; padding:.6rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_frame(path: str, require_rul: bool = False) -> pd.DataFrame:

    # Loading the dataset and checking whether required columns are available
    frame = pd.read_csv(path)
    required = {"engine_id", "cycle"} | ({"RUL"} if require_rul else set())
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    return frame.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

@st.cache_resource
def load_model(path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)

def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in {"engine_id", "cycle", "RUL"}]

# Reusable functions for dashboard headings and metric cards
def section(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='hero'><div class='eyebrow'>FerroVision / Industrial intelligence</div><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)

def metric_card(column, label: str, value: str) -> None:
    column.markdown(f"<div class='card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)

def model_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.keras")) if folder.exists() else []

def model_prefix(path: Path) -> str:
    return path.name.replace("best_", "").replace("_final.keras", "").replace("_casting.keras", "")

def render_eda() -> None:
    section("Asset health, at a glance", "Explore the turbofan degradation data used to train FerroVision’s RUL models.")

    # Loading training data and displaying basic dataset statistics
    frame = load_frame(str(TRAIN), require_rul=True)
    c1, c2, c3, c4 = st.columns(4)

    metric_card(c1,"Sensor records",f"{len(frame):,}")
    metric_card(c2,"Engine units",str(frame.engine_id.nunique()))
    metric_card(c3,"Input sensors",str(len(feature_columns(frame)) + 1))
    metric_card(c4,"Missing values",f"{int(frame.isna().sum().sum()):,}")

    # Displaying RUL distribution and allowing users to explore sensor readings
    st.markdown("### Remaining useful life distribution")
    bins = pd.cut(frame["RUL"], bins=30).value_counts().sort_index()
    chart = pd.DataFrame({"RUL range": bins.index.astype(str), "Records": bins.values}).set_index("RUL range")
    st.bar_chart(chart, color="#6558e8")
    st.markdown("### Engine signal explorer")
    left, right = st.columns(2)
    engine = left.selectbox("Engine", sorted(frame.engine_id.unique()))
    sensor = right.selectbox("Sensor", feature_columns(frame), index=min(10, len(feature_columns(frame))-1))
    history = frame[frame.engine_id == engine].set_index("cycle")
    st.line_chart(history[[sensor]], color="#23b9e6")
    with st.expander("View source data"):
        st.dataframe(frame.head(50), use_container_width=True)

def render_rul() -> None:
    section("RUL forecast", "Estimate remaining operating cycles from an engine’s latest available sensor history.")

    if not TEST.exists():
        st.error("The unlabeled inference file data/raw/test_data.csv is missing.")
        return

    # Loading available RUL models and selecting one for prediction
    choices = model_files(RUL_MODELS)

    if not choices:
        st.error("No RUL model artifacts were found.")
        return
    
    frame = load_frame(str(TEST), require_rul=False)
    left, right = st.columns([1,1])
    selected = left.selectbox("Forecasting model", choices, format_func=lambda p: p.name.replace("best_", "").replace("_final.keras", "").upper())
    model = load_model(str(selected))
    shape = model.input_shape
    prefix = model_prefix(selected)
    feature_file = RUL_MODELS / f"{prefix}_feature_columns.pkl"
    scaler_file = RUL_MODELS / f"{prefix}_feature_scaler.pkl"

    if not feature_file.exists() or not scaler_file.exists():
        st.error(f"Missing preprocessing artifact for {selected.name}.")
        return

    # Loading the feature list and scaler saved during model training
    with open(feature_file,"rb") as stream:
        columns = pickle.load(stream)
    with open(scaler_file,"rb") as stream:
        scaler = pickle.load(stream)

    if not set(columns).issubset(frame.columns):
        st.error("The inference data does not contain this model’s required features.")
        return
    
    engine = right.selectbox("Engine history", sorted(frame.engine_id.unique()))
    history = frame[frame.engine_id == engine].sort_values("cycle")
    expected_rank = len(shape)

    # Preparing input according to whether the selected model is ANN, RNN or LSTM
    if expected_rank == 2:
        sample = history[columns].tail(1).to_numpy("float32")
        model_kind, required = "ANN", 1

    else:
        required = int(shape[1])

        if len(history) < required:
            st.warning(f"Engine {engine} has {len(history)} cycles; {required} are required for this model.")
            return
        
        sample = history[columns].tail(required).to_numpy("float32")[np.newaxis,...]
        model_kind = "RNN" if "rnn" in selected.name else "LSTM"

    # Applying the same scaling used during model training    
    if expected_rank == 2:
        sample = scaler.transform(sample)

    else:
        sample = scaler.transform(sample[0])[np.newaxis,...]

    if st.button("Generate forecast", type="primary", use_container_width=True):
        prediction = max(0.0, float(model.predict(sample, verbose=0).squeeze()))
        a,b,c = st.columns(3)
        metric_card(a,"Predicted RUL",f"{prediction:.1f} cycles")
        metric_card(b,"Model",model_kind)
        metric_card(c,"Input history",f"{required} cycle{'s' if required != 1 else ''}")
        st.caption("Prediction is model-based guidance, not a maintenance authorization.")

    # Plotting the latest sensor readings for the selected engine    
    st.markdown("### Latest sensor trace")
    signal_columns = [
        column for column in columns
        if column.startswith("sensor_")
        ][:3]
    st.line_chart(history.set_index("cycle")[signal_columns], color=["#6558e8", "#23b9e6", "#f59e0b"])

def render_vision() -> None:
    section("Visual defect inspection", "Classify a casting component using the validated image-classification models.")

    # Loading available image classification models
    options = model_files(VISION_MODELS)

    if not options:
        st.error("No vision model artifacts were found.")
        return
    choice = st.selectbox("Inspection model", options, format_func=lambda p: p.name.replace("_casting.keras", "").replace("_", " ").title())
    upload = st.file_uploader("Upload a casting image", type=["jpg","jpeg","png"])

    if not upload:
        st.info("Use an image from the casting inspection workflow to receive a Normal or Defective prediction.")
        return

    # Preparing the uploaded image for model prediction
    image = Image.open(upload).convert("RGB")
    left, right = st.columns([1,1])
    left.image(image, caption="Submitted component", use_container_width=True)
    resized = np.asarray(image.resize((160,160)), dtype="float32")[np.newaxis,...]
    model = load_model(str(choice))
    threshold = 0.31 if choice.name.startswith("mobilenetv2") else 0.50
    probability = float(model.predict(resized, verbose=0).squeeze())

    # Comparing predicted probability with the selected classification threshold
    label = "Defective" if probability >= threshold else "Normal"
    color = "#d92d20" if label == "Defective" else "#039855"

    right.markdown(f"<div class='card'><div class='metric-label'>Inspection result</div><div class='metric-value' style='color:{color}'>{label}</div><p style='color:#667085'>Defect confidence: {probability*100:.1f}%<br>Decision threshold: {threshold:.2f}</p></div>", unsafe_allow_html=True)
    right.caption("MobileNetV2 uses the validation-calibrated 0.31 threshold. Confirm flagged parts with your quality process.")

def render_comparison() -> None:
    section("Model comparison", "Held-out evaluation results from the corrected train, validation, and test workflow.")

    # Collecting RUL evaluation results from the available model metric files
    rul_rows = []

    for name in ("ann","rnn","lstm"):
        file = METRICS/f"{name}_final_metrics.csv"

        if file.exists():
            rul_rows.append(pd.read_csv(file))

    if rul_rows:
        st.markdown("### Remaining useful life")
        rul=pd.concat(rul_rows,ignore_index=True)
        st.dataframe(rul[["model","window","features","MAE","RMSE","R2"]],use_container_width=True,hide_index=True)

    # Collecting image classification results for comparison
    vision_rows = []

    for name in ("custom","mobilenetv2"):
        file = METRICS/f"{name}_casting_metrics.csv"
        if file.exists():
            vision_rows.append(pd.read_csv(file))

    if vision_rows:
        st.markdown("### Casting defect classification")
        vision=pd.concat(vision_rows,ignore_index=True)
        keep=[c for c in ["model","accuracy","precision","recall","f1","threshold"] if c in vision.columns]
        st.dataframe(vision[keep],use_container_width=True,hide_index=True)

# Setting up sidebar navigation between the dashboard sections
st.sidebar.markdown("## FerroVision")
st.sidebar.caption("Predictive maintenance and quality inspection")
page = st.sidebar.radio("Navigate",["Overview","RUL Forecast","Defect Classification","Model Comparison"])
st.sidebar.markdown("---")
st.sidebar.caption("TensorFlow models • corrected splits • validation-selected checkpoints")

# Calling the function associated with the selected dashboard page
{"Overview":render_eda,"RUL Forecast":render_rul,"Defect Classification":render_vision,"Model Comparison":render_comparison}[page]()