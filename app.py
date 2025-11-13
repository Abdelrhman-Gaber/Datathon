
import os
import json
import inspect
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# sklearn imports
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ---- Config ----
st.set_page_config(layout="wide", page_title="Stroke Risk Dashboard")

# Color palette & dark theme CSS
color_palette = [
    "#4A6369",
    "#AB5852",
    "#D49C47",
    "#838469",
    "#D2C38B"
]

st.markdown("""
<style>
body { background-color: #121212; color: #f0f0f0; }
h1, h2, h3, h4, h5, h6 { color: #708090; }
.metric-card {
    padding: 10px; border-radius: 10px; border: 1px solid #333; text-align: center;
}
thead, tbody, tfoot, tr, td, th { color: #708090; }
</style>
""", unsafe_allow_html=True)

# ---- Paths ----
DEFAULT_DATA_PATH = "DataSet_datathon_13112025_CLEAN(in).csv"
MODEL_PATH = "model.joblib"
META_PATH = "model_meta.json"

# ---- Helpers & Caching ----
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def compute_missingness(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    out = missing.to_frame('missing')
    out['missing_percent'] = (out['missing'] / len(df) * 100).round(2)
    return out

def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_preprocessor(X: pd.DataFrame):
    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pre = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])
    # OneHotEncoder compatibility across sklearn versions
    ohe_kwargs = {}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        ohe_kwargs["sparse_output"] = True
    else:
        ohe_kwargs["sparse"] = True
    categorical_pre = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", **ohe_kwargs))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pre, numeric_features),
            ("cat", categorical_pre, categorical_features),
        ]
    )
    return pre, numeric_features, categorical_features

def pick_pos_label(y: pd.Series):
    classes = np.array(sorted(y.dropna().unique()))
    return 1 if 1 in classes else (y.value_counts().idxmin() if len(classes)>0 else 1)

def get_scores(est, X_test, pos_label, y_test=None):
    proba = None
    if hasattr(est, "predict_proba"):
        # pick index of pos label
        classes = list(est.classes_) if hasattr(est, "classes_") else [0,1]
        if pos_label in classes:
            idx = classes.index(pos_label)
        else:
            idx = 1 if len(classes)>1 else 0
        proba = est.predict_proba(X_test)[:, idx]
    elif hasattr(est, "decision_function"):
        from sklearn.preprocessing import MinMaxScaler
        proba = MinMaxScaler().fit_transform(est.decision_function(X_test).reshape(-1,1)).ravel()
    return proba

def feature_names_from_pre(pre: ColumnTransformer, num, cat):
    names = list(num)
    try:
        ohe = pre.named_transformers_["cat"].named_steps["onehot"]
        cats = ohe.categories_
        for col, cats_for_col in zip(cat, cats):
            names.extend([f"{col}={c}" for c in cats_for_col])
    except Exception:
        names.extend(list(cat))
    return names

def save_artifacts(model, meta: dict):
    import joblib
    joblib.dump(model, MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

# ---- Load Data ----
df = load_csv(DEFAULT_DATA_PATH)

st.markdown("<h1>Stroke Risk Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Interactive dashboard exploring stroke risk factors, with notebook-style EDA & modeling — plus a trained classifier for predictions.")

# ---- Sidebar: upload + filters ----
st.sidebar.header("Data")
up = st.sidebar.file_uploader("Upload CSV (optional). Columns like: gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, stroke", type=["csv"])
if up is not None:
    try:
        df = pd.read_csv(up)
    except Exception as e:
        st.error(f"Upload failed: {e}")

if df.empty:
    st.warning("No data found. Please upload a CSV or ensure the default dataset exists at the configured path.")
    st.stop()

df = ensure_numeric(df, ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease", "stroke"])

# Sidebar Filters
st.sidebar.header("Filters")
def selbox(label, column):
    if column not in df.columns:
        return None, None
    options = ["All"] + sorted([x for x in df[column].dropna().unique().tolist()])
    val = st.sidebar.selectbox(label, options)
    return column, val

cat_filters = []
for label, col in [
    ("Gender", "gender"),
    ("Ever Married", "ever_married"),
    ("Work Type", "work_type"),
    ("Residence Type", "Residence_type"),
    ("Smoking Status", "smoking_status")
]:
    cat_filters.append(selbox(label, col))

bin_filters = []
for label, col in [
    ("Hypertension", "hypertension"),
    ("Heart Disease", "heart_disease"),
    ("Stroke (label)", "stroke")
]:
    if col in df.columns:
        opts = ["All", 0, 1]
        val = st.sidebar.selectbox(label, opts)
        bin_filters.append((col, val))

def numeric_slider(label, column, step=1, as_int=False):
    if column not in df.columns or df[column].dropna().empty:
        return None, None, None
    vmin = float(np.nanmin(df[column]))
    vmax = float(np.nanmax(df[column]))
    if as_int:
        vmin_i, vmax_i = int(np.floor(vmin)), int(np.ceil(vmax))
        val = st.sidebar.slider(label, min_value=vmin_i, max_value=vmax_i, value=(vmin_i, vmax_i), step=step)
    else:
        val = st.sidebar.slider(label, min_value=float(vmin), max_value=float(vmax), value=(float(vmin), float(vmax)))
    return column, val[0], val[1]

age_range = numeric_slider("Age Range", "age", step=1, as_int=True)
glc_range = numeric_slider("Avg Glucose Level", "avg_glucose_level")
bmi_range = numeric_slider("BMI", "bmi")

filtered_df = df.copy()
for col, val in cat_filters:
    if col is not None and val is not None and val != "All":
        filtered_df = filtered_df[filtered_df[col] == val]
for col, val in bin_filters:
    if col is not None and val != "All":
        filtered_df = filtered_df[filtered_df[col] == val]
for rng in [age_range, glc_range, bmi_range]:
    if rng[0] is not None:
        col, lo, hi = rng
        filtered_df = filtered_df[(filtered_df[col] >= lo) & (filtered_df[col] <= hi)]

# ---- KPIs ----
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"<div class='metric-card'><h5>Total Records</h5><h2>{len(filtered_df):,}</h2></div>", unsafe_allow_html=True)
with c2:
    mean_age = filtered_df["age"].mean() if "age" in filtered_df.columns else np.nan
    st.markdown(f"<div class='metric-card'><h5>Avg Age</h5><h2>{(mean_age if pd.notna(mean_age) else 0):.1f}</h2></div>", unsafe_allow_html=True)
with c3:
    mean_glc = filtered_df["avg_glucose_level"].mean() if "avg_glucose_level" in filtered_df.columns else np.nan
    st.markdown(f"<div class='metric-card'><h5>Avg Glucose</h5><h2>{(mean_glc if pd.notna(mean_glc) else 0):.1f}</h2></div>", unsafe_allow_html=True)
with c4:
    mean_bmi = filtered_df["bmi"].mean() if "bmi" in filtered_df.columns else np.nan
    st.markdown(f"<div class='metric-card'><h5>Avg BMI</h5><h2>{(mean_bmi if pd.notna(mean_bmi) else 0):.1f}</h2></div>", unsafe_allow_html=True)
with c5:
    if "stroke" in filtered_df.columns and len(filtered_df) > 0:
        rate = 100 * filtered_df["stroke"].mean()
        st.markdown(f"<div class='metric-card'><h5>Stroke Rate</h5><h2>{rate:.2f}%</h2></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='metric-card'><h5>Stroke Rate</h5><h2>—</h2></div>", unsafe_allow_html=True)

# ---- Tabs ----
tabs = st.tabs([
    "Demographics",
    "Medical",
    "Lifestyle & Work",
    "Residence & Geography",
    "Model & Predictions",
    # Notebook-style additions:
    "Overview & Missingness",
    "Distributions & Boxplots",
    "Correlation",
    "Model Training (LR vs RF)",
    "PR/ROC & Threshold",
    "Cost-based Threshold",
    "Feature Importance",
    "Advanced (SMOTE/Tuning/Calibration)",
    "Artifacts"
])

# ---------------- Existing Thematic Tabs ----------------

# DEMOGRAPHICS TAB
with tabs[0]:
    st.header("Demographics")
    demo_cols = [c for c in ["gender", "ever_married", "age"] if c in filtered_df.columns]
    if demo_cols:
        st.subheader(f"Summary Statistics for: {', '.join(demo_cols)}")
        st.dataframe(filtered_df[demo_cols].describe(include='all').transpose(), use_container_width=True)
    cols = st.columns(2)
    with cols[0]:
        if "age" in filtered_df.columns and "stroke" in filtered_df.columns:
            st.subheader("Age Distribution by Stroke")
            st.plotly_chart(px.histogram(filtered_df, x="age", color="stroke", nbins=30, barmode="group",
                                         color_discrete_sequence=color_palette), use_container_width=True)
        if "gender" in filtered_df.columns:
            st.subheader("Gender Distribution")
            st.plotly_chart(px.pie(filtered_df, names="gender", hole=0.3, color_discrete_sequence=color_palette),
                            use_container_width=True)
    with cols[1]:
        if set(["ever_married","stroke"]).issubset(filtered_df.columns):
            st.subheader("Stroke by Marital Status")
            st.plotly_chart(px.bar(filtered_df, x="ever_married", color="stroke", barmode="group",
                                   color_discrete_sequence=color_palette), use_container_width=True)
        if set(["age","stroke"]).issubset(filtered_df.columns):
            st.subheader("Age by Stroke (Box)")
            st.plotly_chart(px.box(filtered_df, x="stroke", y="age", color="stroke",
                                   color_discrete_sequence=color_palette), use_container_width=True)

# MEDICAL TAB
with tabs[1]:
    st.header("Medical")
    med_cols = [c for c in ["hypertension","heart_disease","avg_glucose_level","bmi","stroke"] if c in filtered_df.columns]
    if med_cols:
        st.subheader(f"Summary Statistics for: {', '.join(med_cols)}")
        st.dataframe(filtered_df[med_cols].describe(include='all').transpose(), use_container_width=True)
    cols = st.columns(2)
    with cols[0]:
        if set(["hypertension","stroke"]).issubset(filtered_df.columns):
            st.subheader("Hypertension by Stroke")
            st.plotly_chart(px.bar(filtered_df, x="hypertension", color="stroke", barmode="group",
                                   color_discrete_sequence=color_palette), use_container_width=True)
        if set(["avg_glucose_level","stroke"]).issubset(filtered_df.columns):
            st.subheader("Glucose Distribution by Stroke")
            st.plotly_chart(px.histogram(filtered_df, x="avg_glucose_level", color="stroke", nbins=40, barmode="group",
                                         color_discrete_sequence=color_palette), use_container_width=True)
    with cols[1]:
        if set(["heart_disease","stroke"]).issubset(filtered_df.columns):
            st.subheader("Heart Disease by Stroke")
            st.plotly_chart(px.bar(filtered_df, x="heart_disease", color="stroke", barmode="group",
                                   color_discrete_sequence=color_palette), use_container_width=True)
        if set(["bmi","stroke"]).issubset(filtered_df.columns):
            st.subheader("BMI by Stroke (Box)")
            st.plotly_chart(px.box(filtered_df, x="stroke", y="bmi", color="stroke",
                                   color_discrete_sequence=color_palette), use_container_width=True)

# LIFESTYLE & WORK TAB
with tabs[2]:
    st.header("Lifestyle & Work")
    lw_cols = [c for c in ["work_type","smoking_status","stroke"] if c in filtered_df.columns]
    if lw_cols:
        st.subheader(f"Summary Statistics for: {', '.join(lw_cols)}")
        st.dataframe(filtered_df[lw_cols].describe(include='all').transpose(), use_container_width=True)
    cols = st.columns(2)
    with cols[0]:
        if "work_type" in filtered_df.columns:
            st.subheader("Work Type Distribution")
            ct = filtered_df["work_type"].value_counts().reset_index()
            ct.columns = ["work_type","count"]
            st.plotly_chart(px.bar(ct, x="work_type", y="count", color="work_type",
                                   color_discrete_sequence=color_palette), use_container_width=True)
        if set(["smoking_status","stroke"]).issubset(filtered_df.columns):
            st.subheader("Smoking Status by Stroke")
            st.plotly_chart(px.bar(filtered_df, x="smoking_status", color="stroke", barmode="group",
                                   color_discrete_sequence=color_palette), use_container_width=True)
    with cols[1]:
        if set(["smoking_status"]).issubset(filtered_df.columns):
            st.subheader("Smoking Status Distribution")
            st.plotly_chart(px.pie(filtered_df, names="smoking_status", hole=0.3,
                                   color_discrete_sequence=color_palette), use_container_width=True)

# RESIDENCE & GEOGRAPHY TAB
with tabs[3]:
    st.header("Residence & Geography")
    rg_cols = [c for c in ["Residence_type","age","avg_glucose_level","bmi","stroke"] if c in filtered_df.columns]
    if rg_cols:
        st.subheader(f"Summary Statistics for: {', '.join(rg_cols)}")
        st.dataframe(filtered_df[rg_cols].describe(include='all').transpose(), use_container_width=True)
    cols = st.columns(2)
    with cols[0]:
        if "Residence_type" in filtered_df.columns:
            st.subheader("Residence Type Distribution")
            st.plotly_chart(px.pie(filtered_df, names="Residence_type", hole=0.3,
                                   color_discrete_sequence=color_palette), use_container_width=True)
        if set(["Residence_type","stroke"]).issubset(filtered_df.columns):
            st.subheader("Stroke by Residence Type")
            st.plotly_chart(px.bar(filtered_df, x="Residence_type", color="stroke", barmode="group",
                                   color_discrete_sequence=color_palette), use_container_width=True)
    with cols[1]:
        numeric_cols = [c for c in ["age","avg_glucose_level","bmi"] if c in filtered_df.columns]
        if len(numeric_cols) >= 2 and "stroke" in filtered_df.columns:
            st.subheader("Scatter: Age vs Glucose (color=Stroke)")
            st.plotly_chart(px.scatter(filtered_df, x=numeric_cols[0], y="avg_glucose_level",
                                       color="stroke", opacity=0.6,
                                       color_discrete_sequence=color_palette), use_container_width=True)

# MODEL & PREDICTIONS TAB (quick use)
with tabs[4]:
    st.header("Model & Predictions")
    model, meta = None, None
    default_threshold = 0.5
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        try:
            import joblib
            model = joblib.load(MODEL_PATH)
            with open(META_PATH, "r") as f:
                meta = json.load(f)
            st.success(f"Loaded model: {meta.get('best_model')}")
            if isinstance(meta.get("default_threshold"), (float, int)):
                default_threshold = float(meta["default_threshold"])
        except Exception as e:
            st.error(f"Could not load model: {e}")
    else:
        st.info("No saved model found yet. Use the 'Model Training' tab to create one.")

    if model is not None:
        proba_available = hasattr(model, "predict_proba") or hasattr(model, "decision_function")
        threshold = default_threshold
        if proba_available:
            threshold = st.slider("Decision threshold for positive class", min_value=0.05, max_value=0.95, value=float(default_threshold), step=0.05)

        use_df = filtered_df.copy()
        target_col = meta.get("target_col") if meta else "stroke"
        y_true = None
        if target_col in use_df.columns:
            y_true = use_df.pop(target_col)

        try:
            proba = get_scores(model, use_df, pos_label=1, y_test=None) if proba_available else None
            if proba is not None:
                preds = (proba >= threshold).astype(int)
            else:
                preds = model.predict(use_df)

            out = use_df.copy()
            out["prediction"] = preds
            if proba is not None:
                out["proba_1"] = proba
            st.subheader("Prediction Preview")
            st.dataframe(out.head(30), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

            if y_true is not None:
                from sklearn.metrics import classification_report, confusion_matrix
                st.subheader("Evaluation on filtered subset (if ground truth present)")
                try:
                    report = classification_report(y_true, preds, zero_division=0)
                    st.text(report)
                except Exception as e:
                    st.info(f"Could not compute classification report: {e}")

                try:
                    cm = confusion_matrix(y_true, preds)
                    fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", labels=dict(x="Pred", y="True"),
                                    color_continuous_scale="Viridis")
                    fig.update_xaxes(side="top")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"Could not plot confusion matrix: {e}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------- Notebook-style Tabs ----------------

# OVERVIEW & MISSINGNESS
with tabs[5]:
    st.header("Overview & Missingness")
    st.subheader("Describe (include='all')")
    st.dataframe(df.describe(include='all').transpose(), use_container_width=True)
    miss = compute_missingness(df)
    if not miss.empty:
        st.subheader("Missing values")
        st.dataframe(miss.head(50), use_container_width=True)

# DISTRIBUTIONS & BOXPLOTS
with tabs[6]:
    st.header("Distributions & Boxplots")
    if "stroke" in df.columns:
        st.subheader("Class distribution — stroke")
        vc = df["stroke"].value_counts().reset_index()
        vc.columns = ["class","count"]
        st.plotly_chart(px.bar(vc, x="class", y="count", color="class", color_discrete_sequence=color_palette), use_container_width=True)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "stroke"]
    st.subheader("Numeric histograms (up to 8)")
    cols = st.columns(2)
    for i, c in enumerate(numeric_cols[:8]):
        with cols[i % 2]:
            st.plotly_chart(px.histogram(df, x=c, nbins=50, title=f"Histogram — {c}"), use_container_width=True)

    st.subheader("Box plots by class (up to 6)")
    cols2 = st.columns(2)
    if "stroke" in df.columns:
        for i, c in enumerate(numeric_cols[:6]):
            with cols2[i % 2]:
                st.plotly_chart(px.box(df, x="stroke", y=c, title=f"Boxplot by stroke — {c}", color="stroke",
                                       color_discrete_sequence=color_palette), use_container_width=True)

# CORRELATION
with tabs[7]:
    st.header("Correlation (numeric)")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "stroke"]
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        st.plotly_chart(px.imshow(corr, title="Correlation heatmap (numeric)"), use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation.")

# MODEL TRAINING (LR vs RF)
with tabs[8]:
    st.header("Model Training — Logistic Regression vs RandomForest")
    if "stroke" not in df.columns:
        st.warning("No 'stroke' column found. Please provide a label to train.")
    else:
        # Train/Load controls
        use_filters_for_train = st.checkbox("Train on filtered subset", value=False, help="If unchecked, trains on full dataset.")
        train_df = filtered_df if use_filters_for_train else df
        target_col = "stroke"
        X = train_df.drop(columns=[target_col])
        y = train_df[target_col].astype(int)

        pre, num_feats, cat_feats = build_preprocessor(X)

        log_reg = Pipeline(steps=[("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
        rf = Pipeline(steps=[("pre", pre), ("clf", RandomForestClassifier(n_estimators=200, random_state=42))])

        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        random_state = 42
        stratify = y if y.nunique() <= 20 else None

        if st.button("Train LR & RF"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
            log_reg.fit(X_train, y_train)
            rf.fit(X_train, y_train)

            def eval_model(pipe, name):
                yp = pipe.predict(X_test)
                rep = classification_report(y_test, yp, output_dict=True, zero_division=0)
                f1_macro = rep["macro avg"]["f1-score"]
                return f1_macro, yp, rep

            f1_lr, ypred_lr, rep_lr = eval_model(log_reg, "LogReg")
            f1_rf, ypred_rf, rep_rf = eval_model(rf, "RandomForest")

            best = rf if f1_rf >= f1_lr else log_reg
            best_name = "RandomForest" if best is rf else "LogReg"

            st.session_state["trained"] = {
                "best_name": best_name,
                "best_model": best,
                "X_test": X_test, "y_test": y_test,
                "X_train": X_train, "y_train": y_train,
                "pre": pre, "num_feats": num_feats, "cat_feats": cat_feats,
                "rf": rf, "log_reg": log_reg
            }
            st.success(f"Selected best: {best_name} (by macro-F1)")

            st.subheader("Classification report (best)")
            st.text(classification_report(y_test, best.predict(X_test), zero_division=0))

            cm = confusion_matrix(y_test, best.predict(X_test))
            fig = px.imshow(cm, text_auto=True, title=f"Confusion Matrix — {best_name}", labels=dict(x="Pred", y="True"))
            fig.update_xaxes(side="top")
            st.plotly_chart(fig, use_container_width=True)

            # Save artifacts
            meta = {
                "best_model": best_name,
                "target_col": target_col,
                "numeric_features": num_feats,
                "categorical_features": cat_feats,
            }
            save_artifacts(best, meta)
            st.info("Artifacts saved to disk (model.joblib, model_meta.json).")

        elif "trained" in st.session_state:
            st.info("Using model from previous training step in this session.")
        else:
            st.info("Click 'Train LR & RF' to train a model here, or use the 'Model & Predictions' tab to load an existing one.")

# PR/ROC & THRESHOLD
with tabs[9]:
    st.header("PR/ROC & Threshold")
    state = st.session_state.get("trained")
    model = None
    y_test = None
    X_test = None
    if state is not None:
        model = state["best_model"]
        X_test = state["X_test"]
        y_test = state["y_test"]
    elif os.path.exists(MODEL_PATH):
        import joblib
        model = joblib.load(MODEL_PATH)
        # fallback: rebuild test split from full df
        if "stroke" in df.columns:
            target_col = "stroke"
            X = df.drop(columns=[target_col])
            y = df[target_col].astype(int)
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        st.info("No model available. Train or load a model first.")

    if model is not None and y_test is not None and len(np.unique(y_test))==2:
        pos_label = pick_pos_label(y_test)
        y_scores = get_scores(model, X_test, pos_label, y_test)
        if y_scores is None:
            st.warning("Model does not expose probability/decision scores.")
        else:
            ap = average_precision_score((y_test==pos_label).astype(int), y_scores)
            auc = roc_auc_score((y_test==pos_label).astype(int), y_scores)
            prec, rec, thr = precision_recall_curve((y_test==pos_label).astype(int), y_scores)
            f1s = (2*prec*rec)/(prec+rec+1e-12)
            best_idx = int(np.nanargmax(f1s[:-1]))
            best_thr = float(thr[best_idx])
            best_f1 = float(f1s[best_idx])

            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"PR (AP={ap:.3f})"))
                fig.update_layout(title="Precision–Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.line(x=thr, y=f1s[:-1], labels={"x":"Threshold","y":"F1"}, title="F1 vs Threshold")
                fig2.add_vline(x=best_thr, line_dash="dash")
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown(f"**Best-F1 threshold:** `{best_thr:.2f}` (F1={best_f1:.3f}) — AP={ap:.3f} | ROC AUC={auc:.3f}")
            st.session_state["best_thr"] = best_thr

# COST-BASED THRESHOLD
with tabs[10]:
    st.header("Cost-based Threshold")
    state = st.session_state.get("trained")
    model = None
    y_test = None
    X_test = None
    if state is not None:
        model = state["best_model"]; X_test = state["X_test"]; y_test = state["y_test"]
    elif os.path.exists(MODEL_PATH):
        import joblib
        model = joblib.load(MODEL_PATH)
        if "stroke" in df.columns:
            X = df.drop(columns=["stroke"]); y = df["stroke"].astype(int)
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if model is None or y_test is None:
        st.info("No model available. Train or load a model first.")
    else:
        pos_label = pick_pos_label(y_test)
        y_scores = get_scores(model, X_test, pos_label, y_test)
        if y_scores is None:
            st.warning("Model does not expose probability/decision scores.")
        else:
            COST_FP = st.number_input("Cost of False Positive (FP)", min_value=0.0, value=1.0, step=0.5)
            COST_FN = st.number_input("Cost of False Negative (FN)", min_value=0.0, value=5.0, step=0.5)
            grid = np.linspace(0.05, 0.95, 19)
            from sklearn.metrics import confusion_matrix
            rows = []
            y_true_bin = (y_test==pos_label).astype(int)
            for t in grid:
                yp = (y_scores >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true_bin, yp).ravel()
                rows.append({"threshold": t, "FP": fp, "FN": fn, "cost": COST_FP*fp + COST_FN*fn})
            df_cost = pd.DataFrame(rows).sort_values("threshold")
            st.dataframe(df_cost, use_container_width=True)
            best_row = df_cost.loc[df_cost["cost"].idxmin()]
            st.markdown(f"**Min-cost threshold:** `{best_row['threshold']:.2f}`  | Cost={int(best_row['cost'])} (FP={int(best_row['FP'])}, FN={int(best_row['FN'])})")
            st.plotly_chart(px.line(df_cost, x="threshold", y="cost", title="Cost vs Threshold"), use_container_width=True)
            st.session_state["best_thr_cost"] = float(best_row["threshold"])

# FEATURE IMPORTANCE
with tabs[11]:
    st.header("Feature Importance")
    state = st.session_state.get("trained")
    model = None; pre = None; num_feats=[]; cat_feats=[]
    X_test=None; y_test=None
    if state is not None:
        model = state["best_model"]; pre = state["pre"]; num_feats = state["num_feats"]; cat_feats = state["cat_feats"]
        X_test=state["X_test"]; y_test=state["y_test"]
    elif os.path.exists(MODEL_PATH):
        import joblib
        model = joblib.load(MODEL_PATH)
        # rebuild pre by fitting on full df
        if "stroke" in df.columns:
            X = df.drop(columns=["stroke"]); y = df["stroke"].astype(int)
            pre, num_feats, cat_feats = build_preprocessor(X)
            pre.fit(X, y)  # for OHE categories
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if model is None:
        st.info("No model available.")
    else:
        feat_names = feature_names_from_pre(model.named_steps.get("pre", pre), num_feats, cat_feats)

        # Native importances
        est = model.named_steps.get("clf", model)
        imp_df = None
        if hasattr(est, "coef_"):
            coef = est.coef_
            if coef.ndim > 1:
                idx = 1 if coef.shape[0] > 1 else 0
                coef = coef[idx]
            imp_df = pd.DataFrame({"feature": feat_names, "importance": coef})
            imp_df["abs"] = imp_df["importance"].abs()
            imp_df = imp_df.sort_values("abs", ascending=False).head(20).drop(columns="abs")
            st.plotly_chart(px.bar(imp_df, x="importance", y="feature", orientation="h",
                                   title="Linear coefficients (top 20)"), use_container_width=True)
        elif hasattr(est, "feature_importances_"):
            imp_df = pd.DataFrame({"feature": feat_names, "importance": est.feature_importances_}) \
                        .sort_values("importance", ascending=False).head(20)
            st.plotly_chart(px.bar(imp_df, x="importance", y="feature", orientation="h",
                                   title="Tree feature importances (top 20)"), use_container_width=True)
        else:
            st.info("Model does not expose native importances.")

        # Permutation importance (optional)
        if X_test is not None and y_test is not None:
            from sklearn.inspection import permutation_importance
            try:
                perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
                imp_perm = pd.DataFrame({"feature": feat_names, "importance": perm.importances_mean}) \
                            .sort_values("importance", ascending=False).head(20)
                st.plotly_chart(px.bar(imp_perm, x="importance", y="feature", orientation="h",
                                       title="Permutation importance (top 20)"), use_container_width=True)
            except Exception as e:
                st.info(f"Permutation importance failed: {e}")

# ADVANCED: SMOTE / TUNING / CALIBRATION
with tabs[12]:
    st.header("Advanced: SMOTE / Hyperparameter Tuning / Calibration")
    if "stroke" not in df.columns:
        st.info("Requires 'stroke' column.")
    else:
        X = df.drop(columns=["stroke"]); y = df["stroke"].astype(int)
        pre, num_feats, cat_feats = build_preprocessor(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Class-weighted Logistic Regression"):
                pipe_bal = Pipeline(steps=[("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
                pipe_bal.fit(X_train, y_train)
                st.text(classification_report(y_test, pipe_bal.predict(X_test), zero_division=0))
        with c2:
            if st.button("SMOTE + Logistic Regression (requires imbalanced-learn)"):
                try:
                    from imblearn.pipeline import Pipeline as ImbPipeline
                    from imblearn.over_sampling import SMOTE
                    smote_pipe = ImbPipeline(steps=[
                        ("pre", pre), ("smote", SMOTE(random_state=42)), ("clf", LogisticRegression(max_iter=1000))
                    ])
                    smote_pipe.fit(X_train, y_train)
                    st.text(classification_report(y_test, smote_pipe.predict(X_test), zero_division=0))
                except Exception as e:
                    st.error("Install `imbalanced-learn` to enable SMOTE.")
        with c3:
            if st.button("Tune RandomForest"):
                rf = Pipeline(steps=[("pre", pre), ("clf", RandomForestClassifier(random_state=42))])
                grid = {"clf__n_estimators":[200,400], "clf__max_depth":[None,8,16], "clf__min_samples_leaf":[1,2,5]}
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                gs = GridSearchCV(rf, grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
                gs.fit(X_train, y_train)
                st.write("Best params:", gs.best_params_)
                st.text(classification_report(y_test, gs.best_estimator_.predict(X_test), zero_division=0))

        # Calibration
        if st.button("Calibrate best session model (isotonic)"):
            state = st.session_state.get("trained")
            if state is None:
                st.info("Train a model first in the 'Model Training' tab.")
            else:
                try:
                    from sklearn.calibration import CalibratedClassifierCV
                    base = state["best_model"]
                    cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
                    cal.fit(state["X_train"], state["y_train"])
                    y_scores_cal = get_scores(cal, state["X_test"], pick_pos_label(state["y_test"]), state["y_test"])
                    ap_cal = average_precision_score((state["y_test"]==pick_pos_label(state["y_test"])).astype(int), y_scores_cal)
                    st.success(f"Calibrated model AP: {ap_cal:.3f}")
                    st.session_state["trained"]["best_model"] = cal
                    st.session_state["trained"]["best_name"] = "Calibrated " + st.session_state["trained"]["best_name"]
                except Exception as e:
                    st.error(f"Calibration failed: {e}")

# ARTIFACTS
with tabs[13]:
    st.header("Artifacts")
    default_thr = st.session_state.get("best_thr_cost") or st.session_state.get("best_thr") or 0.5
    st.write("Set a default decision threshold to store in `model_meta.json` (used by 'Model & Predictions' tab).")
    thr_val = st.slider("Default threshold", 0.05, 0.95, float(default_thr), 0.05)
    if st.button("Save artifacts now"):
        state = st.session_state.get("trained")
        if state is None and os.path.exists(MODEL_PATH):
            import joblib
            mdl = joblib.load(MODEL_PATH)
            with open(META_PATH, "r") as f:
                meta = json.load(f)
        elif state is not None:
            mdl = state["best_model"]
            meta = {
                "best_model": state["best_name"],
                "target_col": "stroke",
                "numeric_features": state["num_feats"],
                "categorical_features": state["cat_feats"],
            }
        else:
            mdl, meta = None, None

        if mdl is None or meta is None:
            st.error("No model to save. Train or load one first.")
        else:
            meta["default_threshold"] = float(thr_val)
            meta["pos_label"] = 1
            save_artifacts(mdl, meta)
            st.success("Artifacts saved.")

    st.write("Current files:")
    st.write(f"- Model: `{MODEL_PATH}` exists? **{os.path.exists(MODEL_PATH)}**")
    st.write(f"- Meta: `{META_PATH}` exists? **{os.path.exists(META_PATH)}**")

