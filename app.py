"""
CropCompass - Soil & Climate Based Crop Advisor
------------------------------------------------
Streamlit front-end for the LightGBM crop recommendation project.

The model, split and hyper-parameters mirror the notebook
(crop_recommendation_system.ipynb):
    * 70 / 30 stratified split, random_state = 42
    * LightGBM multiclass, 300 trees, lr 0.05, depth 8, 31 leaves
    * Baselines: Decision Tree, Naive Bayes, KNN, SVM, Random Forest

Run with:   streamlit run app.py
Needs:      Crop_recommendation.csv in the same folder as app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import shap

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# 1. APP IDENTITY + CONSTANTS  (rename things here in one place)
# ============================================================

APP_NAME = "CropCompass"
APP_TAGLINE = "Soil and climate based crop advisor"
APP_ICON = "🧭"

DATA_FILE = "Crop_recommendation.csv"
RANDOM_STATE = 42

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

FEATURE_INFO = {
    "N": ("Nitrogen (N)", "kg/ha", "🟢"),
    "P": ("Phosphorus (P)", "kg/ha", "🟠"),
    "K": ("Potassium (K)", "kg/ha", "🟣"),
    "temperature": ("Temperature", "°C", "🌡️"),
    "humidity": ("Relative humidity", "%", "💧"),
    "ph": ("Soil pH", "", "🧪"),
    "rainfall": ("Rainfall", "mm", "🌧️"),
}

CROP_EMOJI = {
    "apple": "🍎", "banana": "🍌", "blackgram": "🫘", "chickpea": "🫛",
    "coconut": "🥥", "coffee": "☕", "cotton": "☁️", "grapes": "🍇",
    "jute": "🧵", "kidneybeans": "🫘", "lentil": "🥣", "maize": "🌽",
    "mango": "🥭", "mothbeans": "🌱", "mungbean": "🌱", "muskmelon": "🍈",
    "orange": "🍊", "papaya": "🍈", "pigeonpeas": "🌿", "pomegranate": "🍎",
    "rice": "🌾", "watermelon": "🍉",
}

# Colour palette
C_DARK = "#14342b"
C_MID = "#2d6a4f"
C_LIGHT = "#95d5b2"
C_SOFT = "#d8f3dc"
C_AMBER = "#e9a23b"
C_RED = "#c8553d"
C_BG = "#f4f7f1"

st.set_page_config(
    page_title=f"{APP_NAME} | {APP_TAGLINE}",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. STYLING
# ============================================================

st.markdown(
    f"""
    <style>
    .stApp {{ background: {C_BG}; color: #1c2b24; }}
    .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1250px; }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{ background: {C_DARK}; }}
    section[data-testid="stSidebar"] * {{ color: #e8f5ec; }}
    section[data-testid="stSidebar"] [data-baseweb="select"] * {{ color: #1c2b24; }}
    section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.15); }}

    /* Hero banner */
    .hero {{
        background: linear-gradient(120deg, {C_DARK} 0%, {C_MID} 60%, #52b788 100%);
        border-radius: 22px; padding: 30px 36px; margin-bottom: 22px;
        box-shadow: 0 10px 28px rgba(20,52,43,0.22);
    }}
    .hero h1 {{ color: #fff; margin: 0; font-size: 2.3rem; letter-spacing: 0.3px; }}
    .hero p  {{ color: #d8f3dc; margin: 6px 0 0 0; font-size: 1.02rem; }}
    .hero .pill {{
        display: inline-block; margin: 14px 8px 0 0; padding: 4px 13px;
        background: rgba(255,255,255,0.14); color: #fff; border-radius: 999px;
        font-size: 0.8rem; border: 1px solid rgba(255,255,255,0.25);
    }}

    /* Result card */
    .result-card {{
        background: #fff; border-radius: 20px; padding: 26px 28px;
        border-left: 8px solid {C_MID}; box-shadow: 0 6px 20px rgba(0,0,0,0.07);
    }}
    .result-card .label {{ color: #6b7d73; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 1.2px; }}
    .result-card .crop  {{ color: {C_DARK}; font-size: 2.6rem; font-weight: 800; line-height: 1.15; margin: 4px 0 10px 0; }}
    .result-card .conf  {{ color: {C_MID}; font-weight: 700; font-size: 1.15rem; }}
    .bar-bg {{ background: #e6efe9; border-radius: 99px; height: 12px; margin-top: 10px; overflow: hidden; }}
    .bar-fg {{ background: linear-gradient(90deg, #52b788, {C_MID}); height: 12px; border-radius: 99px; }}

    /* Generic cards */
    .card {{
        background: #fff; border-radius: 16px; padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06); margin-bottom: 12px;
    }}
    .card h4 {{ margin: 0 0 6px 0; color: {C_DARK}; }}
    .muted {{ color: #6b7d73; font-size: 0.92rem; }}

    .chip {{
        display: inline-block; padding: 5px 12px; border-radius: 999px; margin: 3px 6px 3px 0;
        background: {C_SOFT}; color: {C_DARK}; font-size: 0.88rem; font-weight: 600;
    }}
    .chip.alt {{ background: #fff1d6; color: #7a4d00; }}

    .seg {{
        background: #fff; border-radius: 12px; padding: 10px 14px; margin-bottom: 8px;
        border-left: 5px solid {C_LIGHT}; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    .seg.now {{ border-left-color: {C_MID}; background: {C_SOFT}; }}

    /* Metrics */
    div[data-testid="stMetric"] {{
        background: #fff; border-radius: 14px; padding: 12px 16px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }}
    div[data-testid="stMetricLabel"] p {{ color: #6b7d73; }}
    div[data-testid="stMetricValue"] {{ color: {C_DARK}; }}

    /* Tabs */
    button[data-baseweb="tab"] {{ font-weight: 600; }}
    button[data-baseweb="tab"] p {{ color: #3d5a4c !important; font-weight: 600; }}
    button[data-baseweb="tab"][aria-selected="true"] p {{ color: {C_MID} !important; }}
    [data-testid="stMain"] [data-testid="stMarkdownContainer"],
    [data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-testid="stMain"] label p {{ color: #1c2b24; }}
    .footer {{ text-align:center; color:#8a9a91; font-size:0.82rem; margin-top:36px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_axis(ax):
    """Common clean look for all matplotlib charts."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b7c4bc")
    ax.spines["bottom"].set_color("#b7c4bc")
    ax.tick_params(colors="#42544b")
    ax.xaxis.label.set_color("#42544b")
    ax.yaxis.label.set_color("#42544b")
    ax.title.set_color(C_DARK)
    ax.set_facecolor("white")


def pretty(crop):
    """Crop name with emoji, e.g. 'rice' -> '🌾 Rice'."""
    return f"{CROP_EMOJI.get(crop, '🌱')} {crop.capitalize()}"


# ============================================================
# 3. DATA + MODEL (cached)
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():
    """Read the dataset and validate the schema used in the notebook."""
    candidates = [Path(__file__).parent / DATA_FILE, Path(DATA_FILE)]
    for path in candidates:
        if path.exists():
            data = pd.read_csv(path)
            needed = FEATURES + ["label"]
            missing = [c for c in needed if c not in data.columns]
            if missing:
                raise ValueError(f"Dataset is missing columns: {missing}")
            return data.dropna(subset=needed).reset_index(drop=True)
    return None


@st.cache_resource(show_spinner="Training the crop model, this happens only once...")
def build_pipeline(data):
    """
    Same pipeline as the notebook: stratified 70/30 split, five baseline
    models and the final LightGBM classifier.
    """
    X = data[FEATURES].copy()
    y = data["label"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y, shuffle=True
    )

    # ---- Baselines ----
    baselines = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    rows = []
    for name, clf in baselines.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="macro", zero_division=0),
            "Recall": recall_score(y_test, pred, average="macro", zero_division=0),
            "F1-Score": f1_score(y_test, pred, average="macro", zero_division=0),
        })

    # ---- Final model ----
    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    rows.append({
        "Model": "LightGBM (final)",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average="macro", zero_division=0),
    })
    comparison = (
        pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    )

    classes = list(model.classes_)
    y_test_bin = label_binarize(y_test, classes=classes)
    roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")

    metrics = {
        "train_acc": accuracy_score(y_train, model.predict(X_train)),
        "test_acc": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc,
    }

    return {
        "model": model,
        "classes": classes,
        "X": X, "y": y,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred,
        "comparison": comparison,
        "metrics": metrics,
        "explainer": shap.TreeExplainer(model),
    }


@st.cache_data(show_spinner="Running 5-fold cross-validation...")
def run_cross_validation(_X, _y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_model = LGBMClassifier(
        objective="multiclass", n_estimators=300, learning_rate=0.05,
        max_depth=8, num_leaves=31, random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=-1,
    )
    return cross_val_score(cv_model, _X, _y, cv=cv, scoring="accuracy", n_jobs=1)


@st.cache_data(show_spinner="Adding noise to the test set and re-scoring...")
def run_noise_test(_model, _X_test, _y_test, _X_train,
                   levels=(0.0, 0.01, 0.05, 0.10), repeats=20):
    feature_std = _X_train.std(axis=0).replace(0, 1)
    lo, hi = _X_train.min(axis=0), _X_train.max(axis=0)
    rng = np.random.default_rng(RANDOM_STATE)

    out = []
    for sigma in levels:
        scores = []
        for _ in range(repeats):
            noise = rng.normal(0, sigma, size=_X_test.shape)
            noisy = pd.DataFrame(
                _X_test.values + noise * feature_std.values,
                columns=_X_test.columns, index=_X_test.index,
            ).clip(lower=lo, upper=hi, axis="columns")
            scores.append(accuracy_score(_y_test, _model.predict(noisy)))
        out.append({"Noise (σ)": sigma, "Mean accuracy": np.mean(scores), "Std": np.std(scores)})
    return pd.DataFrame(out)


@st.cache_data(show_spinner="Computing global SHAP importance...")
def global_shap_importance(_explainer, _X_test):
    sample = _X_test.sample(min(300, len(_X_test)), random_state=RANDOM_STATE)
    values = _explainer.shap_values(sample)
    if isinstance(values, list):
        stacked = np.stack([np.asarray(v) for v in values], axis=0)
        imp = np.mean(np.abs(stacked), axis=(0, 1))
    else:
        arr = np.asarray(values)
        imp = np.mean(np.abs(arr), axis=(0, 2)) if arr.ndim == 3 else np.mean(np.abs(arr), axis=0)
    return pd.Series(imp, index=FEATURES).sort_values()


def local_shap_for_class(shap_values, class_index, row=0):
    """Handle the different SHAP output shapes for multiclass models."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_index])[row]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[row, :, class_index]
    return arr[row]


# ============================================================
# 4. LOAD EVERYTHING
# ============================================================

try:
    df = load_data()
except ValueError as err:
    st.error(str(err))
    st.stop()

if df is None:
    st.markdown(
        f"""
        <div class="hero"><h1>{APP_ICON} {APP_NAME}</h1>
        <p>{APP_TAGLINE}</p></div>
        """,
        unsafe_allow_html=True,
    )
    st.error(
        f"**{DATA_FILE}** was not found. Please keep the CSV file in the same folder as app.py and rerun."
    )
    st.stop()

bundle = build_pipeline(df)
model = bundle["model"]
classes = bundle["classes"]
metrics = bundle["metrics"]
crop_profile = df.groupby("label")[FEATURES].mean()
crop_min = df.groupby("label")[FEATURES].min()
crop_max = df.groupby("label")[FEATURES].max()


# ============================================================
# 5. HEADER
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <h1>{APP_ICON} {APP_NAME}</h1>
        <p>{APP_TAGLINE}. Describe your soil and weather, and get a crop suggestion
        together with the reasoning behind it.</p>
        <span class="pill">{len(df):,} field records</span>
        <span class="pill">{df['label'].nunique()} crops</span>
        <span class="pill">LightGBM classifier</span>
        <span class="pill">{metrics['test_acc']:.1%} test accuracy</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6. SIDEBAR - FIELD CONDITIONS
# ============================================================

def feature_bounds(name):
    lo, hi = float(df[name].min()), float(df[name].max())
    return lo, hi


def init_inputs():
    for f in FEATURES:
        key = f"in_{f}"
        if key not in st.session_state:
            med = float(df[f].median())
            st.session_state[key] = int(med) if f in ("N", "P", "K") else round(med, 1)


def apply_preset():
    choice = st.session_state.get("preset_choice")
    if choice and choice != "Custom values":
        for f in FEATURES:
            lo, hi = feature_bounds(f)
            val = float(np.clip(crop_profile.loc[choice, f], lo, hi))
            st.session_state[f"in_{f}"] = int(round(val)) if f in ("N", "P", "K") else round(val, 1)


init_inputs()

with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_NAME}")
    st.caption("Set your field conditions below. Results update instantly.")
    st.divider()

    st.selectbox(
        "Quick start from a typical crop profile",
        ["Custom values"] + sorted(classes),
        key="preset_choice",
        on_change=apply_preset,
        format_func=lambda c: c if c == "Custom values" else pretty(c),
        help="Loads the average soil and climate values of a crop, then you can tweak them.",
    )

    st.markdown("#### Soil nutrients")
    for f in ("N", "P", "K"):
        name, unit, icon = FEATURE_INFO[f]
        lo, hi = feature_bounds(f)
        st.slider(f"{icon} {name} ({unit})", int(lo), int(hi), key=f"in_{f}")

    st.markdown("#### Weather and soil reaction")
    for f in ("temperature", "humidity", "ph", "rainfall"):
        name, unit, icon = FEATURE_INFO[f]
        lo, hi = feature_bounds(f)
        label = f"{icon} {name} ({unit})" if unit else f"{icon} {name}"
        st.slider(label, round(lo, 1), round(hi, 1), step=0.1, key=f"in_{f}")

    st.divider()
    st.caption("Tip: pick a crop in the quick start box to see how a typical field for that crop looks.")


input_df = pd.DataFrame([[st.session_state[f"in_{f}"] for f in FEATURES]], columns=FEATURES)

# Prediction for the current input
proba = model.predict_proba(input_df)[0]
order = np.argsort(proba)[::-1]
pred_crop = classes[order[0]]
pred_index = int(order[0])
confidence = float(proba[order[0]])


# ============================================================
# 7. TABS
# ============================================================

tab_rec, tab_why, tab_whatif, tab_report, tab_library = st.tabs(
    ["🌱 Recommendation", "🔍 Why this crop", "🔀 What-if lab", "📊 Model report", "📚 Crop library"]
)


# ------------------------------------------------------------
# TAB 1 - RECOMMENDATION
# ------------------------------------------------------------
with tab_rec:
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="label">Best match for your field</div>
                <div class="crop">{pretty(pred_crop)}</div>
                <div class="conf">{confidence:.1%} model confidence</div>
                <div class="bar-bg"><div class="bar-fg" style="width:{confidence*100:.1f}%"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if confidence < 0.60:
            st.warning(
                "The model is not very sure here. Your conditions sit between several crops, "
                "so compare the runner-ups on the right."
            )

        st.markdown("##### Your field vs a typical " + pred_crop.capitalize() + " field")
        rows = []
        for f in FEATURES:
            name, unit, _ = FEATURE_INFO[f]
            val = float(input_df[f].iloc[0])
            lo, hi = float(crop_min.loc[pred_crop, f]), float(crop_max.loc[pred_crop, f])
            if val < lo:
                status = "Below usual range"
            elif val > hi:
                status = "Above usual range"
            else:
                status = "Within range"
            rows.append({
                "Parameter": f"{name} {f'({unit})' if unit else ''}".strip(),
                "Your value": round(val, 1),
                "Typical": round(float(crop_profile.loc[pred_crop, f]), 1),
                "Seen in data": f"{lo:.1f} to {hi:.1f}",
                "Status": status,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    with right:
        st.markdown("##### Top 5 candidate crops")
        top = order[:5]
        names = [classes[i].capitalize() for i in top][::-1]
        probs = [proba[i] * 100 for i in top][::-1]
        colors = [C_LIGHT] * 4 + [C_MID]

        fig, ax = plt.subplots(figsize=(6, 3.6))
        bars = ax.barh(names, probs, color=colors, height=0.62)
        for bar, p in zip(bars, probs):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{p:.1f}%", va="center", fontsize=9, color="#42544b")
        ax.set_xlim(0, 112)
        ax.set_xlabel("Probability (%)")
        style_axis(ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        runner_ups = [classes[i] for i in order[1:4] if proba[i] >= 0.01]
        if runner_ups:
            st.markdown("**Also worth a look:**")
            st.markdown(
                "".join(f'<span class="chip alt">{pretty(c)} ({proba[classes.index(c)]:.1%})</span>'
                        for c in runner_ups),
                unsafe_allow_html=True,
            )
        else:
            st.caption("No other crop gets more than 1% probability, so this is a clear-cut case.")


# ------------------------------------------------------------
# TAB 2 - EXPLAINABILITY (SHAP)
# ------------------------------------------------------------
with tab_why:
    st.markdown(
        f"""
        <div class="card">
        <h4>How the model reached {pretty(pred_crop)}</h4>
        <div class="muted">SHAP splits the prediction into one contribution per feature.
        Green bars pushed the model toward this crop, red bars pushed it away.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("##### This prediction (local)")
        try:
            sv = bundle["explainer"].shap_values(input_df)
            contrib = pd.Series(local_shap_for_class(sv, pred_index), index=FEATURES)
            contrib = contrib.reindex(contrib.abs().sort_values().index)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(
                [f"{FEATURE_INFO[f][0]} = {input_df[f].iloc[0]:.1f}" for f in contrib.index],
                contrib.values,
                color=[C_MID if v >= 0 else C_RED for v in contrib.values],
                height=0.62,
            )
            ax.axvline(0, color="#8a9a91", lw=0.8)
            ax.set_xlabel("SHAP value (push toward / away from the crop)")
            style_axis(ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            strongest = contrib.abs().idxmax()
            direction = "supports" if contrib[strongest] >= 0 else "argues against"
            st.caption(
                f"The biggest influence is {FEATURE_INFO[strongest][0].lower()}, "
                f"which {direction} {pred_crop.capitalize()} for your current inputs."
            )
        except Exception as err:
            st.warning(f"Local SHAP explanation could not be generated: {err}")

    with c2:
        st.markdown("##### Across the whole dataset (global)")
        try:
            g = global_shap_importance(bundle["explainer"], bundle["X_test"])
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh([FEATURE_INFO[f][0] for f in g.index], g.values, color=C_LIGHT, height=0.62)
            ax.barh([FEATURE_INFO[g.index[-1]][0]], [g.values[-1]], color=C_MID, height=0.62)
            ax.set_xlabel("Mean |SHAP value|")
            style_axis(ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                f"Overall, {FEATURE_INFO[g.index[-1]][0].lower()} matters most for telling crops apart, "
                f"and {FEATURE_INFO[g.index[0]][0].lower()} matters least."
            )
        except Exception as err:
            st.warning(f"Global SHAP importance could not be generated: {err}")


# ------------------------------------------------------------
# TAB 3 - WHAT-IF LAB
# ------------------------------------------------------------
with tab_whatif:
    st.markdown(
        """
        <div class="card">
        <h4>What-if lab</h4>
        <div class="muted">Pick one factor and slide it across its full range while everything else stays
        as you set it. This shows how sensitive the model is, it does not prove that changing one real
        field condition on its own will change what grows best.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    feature_choice = st.selectbox(
        "Factor to vary",
        FEATURES,
        index=FEATURES.index("rainfall"),
        format_func=lambda f: f"{FEATURE_INFO[f][2]} {FEATURE_INFO[f][0]}",
    )

    lo, hi = feature_bounds(feature_choice)
    grid = np.linspace(lo, hi, 60)
    sweep = pd.concat([input_df.astype(float)] * len(grid), ignore_index=True)
    sweep[feature_choice] = grid

    sweep_proba = model.predict_proba(sweep)
    sweep_pred = [classes[i] for i in sweep_proba.argmax(axis=1)]
    current_value = float(input_df[feature_choice].iloc[0])
    unit = FEATURE_INFO[feature_choice][1]

    # Group consecutive identical predictions into ranges
    segments = []
    start = 0
    for i in range(1, len(grid) + 1):
        if i == len(grid) or sweep_pred[i] != sweep_pred[start]:
            segments.append((grid[start], grid[i - 1], sweep_pred[start]))
            start = i

    w1, w2 = st.columns([1.2, 1], gap="large")

    with w1:
        st.markdown("##### How confidence in each crop changes")
        fig, ax = plt.subplots(figsize=(7, 4))
        shown = {pred_crop}
        shown.update(sweep_pred)
        palette = plt.cm.tab20(np.linspace(0, 1, 20))
        for n, crop in enumerate(sorted(shown)):
            ci = classes.index(crop)
            is_main = crop == pred_crop
            ax.plot(
                grid, sweep_proba[:, ci] * 100,
                lw=3 if is_main else 1.6,
                color=C_MID if is_main else palette[(n * 3) % 20],
                label=pretty(crop).split(" ", 1)[1],
                zorder=3 if is_main else 2,
            )
        ax.axvline(current_value, color=C_AMBER, ls="--", lw=1.6)
        ax.text(current_value, 103, " you are here", color=C_AMBER, fontsize=8, va="bottom")
        ax.set_ylim(0, 112)
        ax.set_xlabel(f"{FEATURE_INFO[feature_choice][0]} {f'({unit})' if unit else ''}")
        ax.set_ylabel("Probability (%)")
        ax.legend(fontsize=8, ncol=2, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.2))
        style_axis(ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with w2:
        st.markdown("##### Recommended crop along the range")
        for seg_lo, seg_hi, crop in segments:
            is_now = seg_lo <= current_value <= seg_hi
            tag = " &nbsp;(your current setting)" if is_now else ""
            st.markdown(
                f"""
                <div class="seg {'now' if is_now else ''}">
                    <b>{pretty(crop)}</b>{tag}<br>
                    <span class="muted">{seg_lo:.1f} to {seg_hi:.1f} {unit}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        changed = sorted({c for _, _, c in segments if c != pred_crop})
        if changed:
            st.markdown(
                "Alternatives reached by changing this factor alone: "
                + "".join(f'<span class="chip">{pretty(c)}</span>' for c in changed),
                unsafe_allow_html=True,
            )
        else:
            st.info("Changing this factor alone never changes the recommendation, so it is stable here.")


# ------------------------------------------------------------
# TAB 4 - MODEL REPORT
# ------------------------------------------------------------
with tab_report:
    st.markdown("##### Final model scorecard (held-out 30% test set)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Test accuracy", f"{metrics['test_acc']:.2%}")
    m2.metric("Macro precision", f"{metrics['precision']:.2%}")
    m3.metric("Macro recall", f"{metrics['recall']:.2%}")
    m4.metric("Macro F1", f"{metrics['f1']:.2%}")
    m5.metric("Macro ROC-AUC", f"{metrics['roc_auc']:.4f}")
    st.caption(
        f"Training accuracy is {metrics['train_acc']:.2%}, so the train/test gap is "
        f"{(metrics['train_acc'] - metrics['test_acc']):.2%}."
    )

    st.markdown("##### Model comparison")
    comp = bundle["comparison"]
    st.dataframe(
        comp.style
        .format({c: "{:.2%}" for c in ["Accuracy", "Precision", "Recall", "F1-Score"]})
        .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score"],
                       color=C_SOFT, props=f"background-color:{C_SOFT}; color:{C_DARK}; font-weight:700"),
        hide_index=True,
    )
    st.caption(
        "LightGBM is the model used for every prediction in this app because it gives well calibrated "
        "class probabilities and works directly with SHAP explanations."
    )

    r1, r2 = st.columns(2, gap="large")

    with r1:
        st.markdown("##### Confusion matrix")
        cm = confusion_matrix(bundle["y_test"], bundle["y_pred"], labels=classes)
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        ax.imshow(cm, cmap="Greens")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j] > 0:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                            color="white" if cm[i, j] > cm.max() * 0.5 else "#1c2b24")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with r2:
        st.markdown("##### Stability checks")
        st.caption("These run on demand because they retrain or re-score the model many times.")

        if st.button("Run 5-fold cross-validation"):
            scores = run_cross_validation(bundle["X"], bundle["y"])
            st.session_state["cv_scores"] = scores
        if "cv_scores" in st.session_state:
            s = st.session_state["cv_scores"]
            st.success(f"Mean CV accuracy **{s.mean():.2%}** (std {s.std():.2%})")
            st.caption("Fold scores: " + ", ".join(f"{v:.2%}" for v in s))

        if st.button("Run noise robustness test"):
            st.session_state["noise_df"] = run_noise_test(
                model, bundle["X_test"], bundle["y_test"], bundle["X_train"]
            )
        if "noise_df" in st.session_state:
            nd = st.session_state["noise_df"]
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.errorbar(nd["Noise (σ)"], nd["Mean accuracy"] * 100, yerr=nd["Std"] * 100,
                        marker="o", color=C_MID, capsize=4, lw=2)
            ax.set_xlabel("Relative Gaussian noise (σ of each feature)")
            ax.set_ylabel("Accuracy (%)")
            style_axis(ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Accuracy drops gradually as the inputs get noisier, which is what we want to see.")


# ------------------------------------------------------------
# TAB 5 - CROP LIBRARY
# ------------------------------------------------------------
with tab_library:
    st.markdown(
        """
        <div class="card"><h4>Crop library</h4>
        <div class="muted">Browse the average growing conditions of each crop in the dataset.</div></div>
        """,
        unsafe_allow_html=True,
    )

    pick = st.selectbox(
        "Choose a crop", sorted(classes),
        index=sorted(classes).index(pred_crop),
        format_func=pretty,
    )

    st.markdown(f"#### {pretty(pick)}")
    cols = st.columns(len(FEATURES))
    for col, f in zip(cols, FEATURES):
        name, unit, _ = FEATURE_INFO[f]
        col.metric(f, f"{crop_profile.loc[pick, f]:.1f}")

    # Crop value relative to the full dataset range, so all 7 features share one axis
    rel_mean, rel_lo, rel_hi = [], [], []
    for f in FEATURES:
        a, b = float(df[f].min()), float(df[f].max())
        span = (b - a) or 1.0
        rel_mean.append((crop_profile.loc[pick, f] - a) / span * 100)
        rel_lo.append((crop_min.loc[pick, f] - a) / span * 100)
        rel_hi.append((crop_max.loc[pick, f] - a) / span * 100)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    y_pos = np.arange(len(FEATURES))
    ax.barh(y_pos, np.array(rel_hi) - np.array(rel_lo), left=rel_lo, color=C_LIGHT, height=0.5,
            label="Range seen for this crop")
    ax.scatter(rel_mean, y_pos, color=C_DARK, zorder=3, s=45, label="Crop average")
    current_rel = [
        (float(input_df[f].iloc[0]) - float(df[f].min())) / ((float(df[f].max()) - float(df[f].min())) or 1) * 100
        for f in FEATURES
    ]
    ax.scatter(current_rel, y_pos, color=C_AMBER, marker="D", zorder=4, s=40, label="Your field")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([FEATURE_INFO[f][0] for f in FEATURES])
    ax.invert_yaxis()
    ax.set_xlim(-3, 103)
    ax.set_xlabel("Position within the full dataset range (0% = lowest seen, 100% = highest seen)")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    style_axis(ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("See average profile of all crops"):
        st.dataframe(crop_profile.round(1).rename_axis("crop").reset_index())


# ============================================================
# 8. FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="footer">
        {APP_NAME} is a decision-support demo built on a public crop dataset. Use it as a starting point
        and confirm with local agronomy advice before planting.
    </div>
    """,
    unsafe_allow_html=True,
)
