import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import api_client

from train_model import build_preprocessor, evaluate, load_and_clean, TARGET
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


st.set_page_config(
    page_title="HR Analytics Dashboard",
    layout="wide"
)

MODEL_PATH = "attrition_model.joblib"
METADATA_PATH = "attrition_model_metadata.json"


@st.cache_resource
def load_default_model():
    try:
        model = joblib.load(MODEL_PATH)

        with open(METADATA_PATH) as f:
            metadata = json.load(f)

        return model, metadata

    except FileNotFoundError:
        return None, None


def train_new_model(df: pd.DataFrame):

    df = df.copy()

    df = df.drop(
        columns=[
            c
            for c in [
                "EmployeeCount",
                "EmployeeNumber",
                "Over18",
                "StandardHours"
            ]
            if c in df.columns
        ]
    )

    df = df.drop_duplicates()

    df[TARGET] = (
        df[TARGET].map({"Yes": 1, "No": 0})
        if df[TARGET].dtype == object
        else df[TARGET]
    )

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor, cat_cols, num_cols = build_preprocessor(df)

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        ),

        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        ),
    }

    results = {}
    pipelines = {}

    for name, clf in candidates.items():

        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", clf)
            ]
        )

        pipe.fit(X_train, y_train)

        metrics, _ = evaluate(
            pipe,
            X_test,
            y_test
        )

        results[name] = metrics
        pipelines[name] = pipe

    best_name = max(
        results,
        key=lambda n: results[n]["roc_auc"]
    )

    best_pipeline = pipelines[best_name]

    metadata = {
        "best_model": best_name,
        "all_results": results,
        "feature_columns": X.columns.tolist(),
        "categorical_columns": cat_cols,
        "numeric_columns": num_cols,
        "target": TARGET,
    }

    return best_pipeline, metadata, results


def predict_df(df, model, metadata):

    feature_columns = metadata["feature_columns"]

    missing = [
        c
        for c in feature_columns
        if c not in df.columns
    ]

    if missing:
        st.error(
            f"Uploaded file is missing required columns: {missing}"
        )
        return None

    aligned = df[feature_columns].copy()

    proba = model.predict_proba(aligned)[:, 1]
    pred = model.predict(aligned)

    out = df.copy()

    out["Attrition_Prediction"] = [
        "Yes" if p == 1 else "No"
        for p in pred
    ]

    out["Attrition_Probability"] = proba.round(4)

    out["Risk_Level"] = pd.cut(
        proba,
        bins=[-0.01, 0.3, 0.6, 1.01],
        labels=["Low", "Medium", "High"]
    )

    return out


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "HR Analytics Dashboard — Employee Attrition Prediction"
)

st.caption(
    "Upload employee data, explore trends, train a model, "
    "and predict attrition risk."
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Dashboard / EDA",
        "🤖 Train Model",
        "🔮 Predict Attrition",
        "📜 My History",
        "ℹ️ About"
    ]
)


# ============================================================
# SESSION STATE
# ============================================================

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None


if "model" not in st.session_state:

    m, md = load_default_model()

    st.session_state.model = m
    st.session_state.metadata = md


if "auth_token" not in st.session_state:
    st.session_state.auth_token = None


if "current_user" not in st.session_state:
    st.session_state.current_user = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Account")

    if not api_client.is_backend_up():

        st.error(
            "Backend API is not reachable. Start it with "
            "`python backend/app.py` "
            "(see backend/README or the main README)."
        )

    elif st.session_state.auth_token is None:

        auth_mode = st.radio(
            "",
            ["Login", "Sign up"],
            horizontal=True,
            label_visibility="collapsed"
        )

        # ----------------------------------------------------
        # LOGIN
        # ----------------------------------------------------

        if auth_mode == "Login":

            with st.form("login_form"):

                email = st.text_input("Email")

                password = st.text_input(
                    "Password",
                    type="password"
                )

                submitted = st.form_submit_button(
                    "Log in"
                )

            if submitted:

                resp = api_client.login(
                    email,
                    password
                )

                if resp.get("success"):

                    st.session_state.auth_token = (
                        resp["data"]["token"]
                    )

                    st.session_state.current_user = (
                        resp["data"]["user"]
                    )

                    st.success("Logged in.")

                    st.rerun()

                else:

                    st.error(
                        resp.get(
                            "message",
                            "Login failed"
                        )
                    )

        # ----------------------------------------------------
        # SIGN UP
        # ----------------------------------------------------

        else:

            with st.form("signup_form"):

                name = st.text_input(
                    "Full name"
                )

                email = st.text_input(
                    "Email"
                )

                password = st.text_input(
                    "Password (min 6 characters)",
                    type="password"
                )

                submitted = st.form_submit_button(
                    "Create account"
                )

            if submitted:

                resp = api_client.register(
                    name,
                    email,
                    password
                )

                if resp.get("success"):

                    st.success(
                        "Account created. Please log in."
                    )

                else:

                    msg = resp.get(
                        "message",
                        "Registration failed"
                    )

                    errs = resp.get("errors")

                    st.error(
                        f"{msg}"
                        + (f" — {errs}" if errs else "")
                    )

    # --------------------------------------------------------
    # LOGGED IN
    # --------------------------------------------------------

    else:

        user = (
            st.session_state.current_user
            or {}
        )

        st.success(
            f"Logged in as **{user.get('name', 'User')}**"
        )

        st.caption(
            user.get("email", "")
        )

        if st.button("Log out"):

            api_client.logout(
                st.session_state.auth_token
            )

            st.session_state.auth_token = None
            st.session_state.current_user = None

            st.rerun()

    st.divider()

    # ========================================================
    # UPLOAD DATASET
    # ========================================================

    st.header("1. Upload Employee Dataset")

    uploaded = st.file_uploader(
        "CSV file (e.g. IBM HR Attrition dataset)",
        type=["csv"]
    )

    if uploaded is not None:

        st.session_state.raw_df = pd.read_csv(
            uploaded
        )

        st.success(
            f"Loaded {len(st.session_state.raw_df)} rows"
        )

    if st.session_state.model is not None:

        st.info(
            f"Active model: "
            f"**{st.session_state.metadata['best_model']}**"
        )

    else:

        st.warning(
            "No trained model loaded yet. "
            "Go to 'Train Model' tab."
        )


# ============================================================
# TAB 1: DASHBOARD / EDA
# ============================================================

with tab1:

    df = st.session_state.raw_df

    if df is None:

        st.info(
            "Upload a dataset from the sidebar "
            "to see HR analytics here."
        )

    else:

        if TARGET not in df.columns:

            st.warning(
                f"No '{TARGET}' column found — "
                "showing general stats only."
            )

        else:

            col1, col2, col3, col4 = st.columns(4)

            rate = (
                (df[TARGET] == "Yes").mean()
                if df[TARGET].dtype == object
                else df[TARGET].mean()
            )

            col1.metric(
                "Total Employees",
                len(df)
            )

            col2.metric(
                "Attrition Rate",
                f"{rate:.1%}"
            )

            col3.metric(
                "Avg Monthly Income",
                (
                    f"{df['MonthlyIncome'].mean():,.0f}"
                    if "MonthlyIncome" in df
                    else "N/A"
                )
            )

            col4.metric(
                "Avg Years at Company",
                (
                    f"{df['YearsAtCompany'].mean():.1f}"
                    if "YearsAtCompany" in df
                    else "N/A"
                )
            )

            # ------------------------------------------------
            # DEPARTMENT / JOB ROLE
            # ------------------------------------------------

            c1, c2 = st.columns(2)

            if "Department" in df.columns:

                dep = (
                    df.groupby("Department")[TARGET]
                    .apply(
                        lambda s: (
                            s == "Yes"
                        ).mean()
                    )
                    .reset_index(
                        name="AttritionRate"
                    )
                )

                fig = px.bar(
                    dep,
                    x="Department",
                    y="AttritionRate",
                    title="Attrition Rate by Department"
                )

                c1.plotly_chart(
                    fig,
                    use_container_width=True
                )

            if "JobRole" in df.columns:

                role = (
                    df.groupby("JobRole")[TARGET]
                    .apply(
                        lambda s: (
                            s == "Yes"
                        ).mean()
                    )
                    .reset_index(
                        name="AttritionRate"
                    )
                )

                fig = px.bar(
                    role,
                    x="JobRole",
                    y="AttritionRate",
                    title="Attrition Rate by Job Role"
                )

                fig.update_xaxes(
                    tickangle=45
                )

                c2.plotly_chart(
                    fig,
                    use_container_width=True
                )

            # ------------------------------------------------
            # OVERTIME / AGE
            # ------------------------------------------------

            c3, c4 = st.columns(2)

            if "OverTime" in df.columns:

                ot = (
                    df.groupby("OverTime")[TARGET]
                    .apply(
                        lambda s: (
                            s == "Yes"
                        ).mean()
                    )
                    .reset_index(
                        name="AttritionRate"
                    )
                )

                fig = px.bar(
                    ot,
                    x="OverTime",
                    y="AttritionRate",
                    title="Attrition Rate by OverTime"
                )

                c3.plotly_chart(
                    fig,
                    use_container_width=True
                )

            if "Age" in df.columns:

                fig = px.histogram(
                    df,
                    x="Age",
                    color=TARGET,
                    barmode="overlay",
                    title="Age Distribution by Attrition"
                )

                c4.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # ----------------------------------------------------
        # RAW DATA
        # ----------------------------------------------------

        st.subheader("Raw Data Preview")

        st.dataframe(
            df.head(50),
            use_container_width=True
        )


# ============================================================
# TAB 2: TRAIN MODEL
# ============================================================

with tab2:

    st.subheader(
        "Train / Retrain the Attrition Prediction Model"
    )

    df = st.session_state.raw_df

    if df is None:

        st.info(
            "Upload a labeled dataset "
            "(with an 'Attrition' column) "
            "from the sidebar first."
        )

    elif TARGET not in df.columns:

        st.error(
            f"Training requires a '{TARGET}' column "
            "(Yes/No) in the uploaded data."
        )

    else:

        if st.button(
            "🚀 Train Models "
            "(Logistic Regression, Random Forest, "
            "Gradient Boosting)"
        ):

            with st.spinner(
                "Training and evaluating models..."
            ):

                model, metadata, results = (
                    train_new_model(df)
                )

                st.session_state.model = model
                st.session_state.metadata = metadata

                joblib.dump(
                    model,
                    MODEL_PATH
                )

                with open(
                    METADATA_PATH,
                    "w"
                ) as f:

                    json.dump(
                        metadata,
                        f,
                        indent=2
                    )

            st.success(
                f"Best model: "
                f"**{metadata['best_model']}** "
                f"(saved to {MODEL_PATH})"
            )

            results_df = pd.DataFrame(
                results
            ).T

            st.dataframe(
                results_df,
                use_container_width=True
            )

            fig = px.bar(
                results_df.reset_index(),
                x="index",
                y="roc_auc",
                title="Model Comparison (ROC-AUC)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# TAB 3: PREDICT
# ============================================================

with tab3:

    st.subheader(
        "Predict Employee Attrition"
    )

    model = st.session_state.model
    metadata = st.session_state.metadata

    if model is None:

        st.warning(
            "Train a model first (see 'Train Model' tab), "
            "or make sure attrition_model.joblib exists."
        )

    else:

        mode = st.radio(
            "Prediction mode",
            [
                "Batch (CSV upload)",
                "Single employee (manual entry)"
            ],
            horizontal=True
        )

        # ====================================================
        # BATCH PREDICTION
        # ====================================================

        if mode == "Batch (CSV upload)":

            pred_file = st.file_uploader(
                "Upload employee records to score "
                "(no Attrition column needed)",
                type=["csv"],
                key="pred_upload"
            )

            if pred_file is not None:

                new_df = pd.read_csv(
                    pred_file
                )

                result = predict_df(
                    new_df,
                    model,
                    metadata
                )

                if result is not None:

                    st.dataframe(
                        result,
                        use_container_width=True
                    )

                    high_risk = (
                        result["Risk_Level"] == "High"
                    ).sum()

                    st.metric(
                        "High-risk employees flagged",
                        high_risk
                    )

                    csv = (
                        result
                        .to_csv(index=False)
                        .encode("utf-8")
                    )

                    st.download_button(
                        "⬇️ Download predictions as CSV",
                        csv,
                        "attrition_predictions.csv",
                        "text/csv"
                    )

        # ====================================================
        # SINGLE EMPLOYEE
        # ====================================================

        else:

            if st.session_state.auth_token is None:

                st.warning(
                    "🔒 Please log in from the sidebar "
                    "to run a single-employee prediction. "
                    "This calls the backend API and saves "
                    "the result to your personal history."
                )

            else:

                st.write(
                    "Enter employee details:"
                )

                cols = st.columns(3)

                inputs = {}

                feature_columns = metadata[
                    "feature_columns"
                ]

                cat_cols = metadata.get(
                    "categorical_columns",
                    []
                )

                base_df = st.session_state.raw_df

                for i, col in enumerate(
                    feature_columns
                ):

                    target_col = cols[i % 3]

                    if (
                        col in cat_cols
                        and base_df is not None
                        and col in base_df.columns
                    ):

                        options = sorted(
                            base_df[col]
                            .dropna()
                            .unique()
                            .tolist()
                        )

                        inputs[col] = (
                            target_col.selectbox(
                                col,
                                options
                            )
                        )

                    else:

                        if (
                            base_df is not None
                            and col in base_df.columns
                        ):

                            default_val = float(
                                base_df[col].mean()
                            )

                        else:

                            default_val = 0.0

                        inputs[col] = (
                            target_col.number_input(
                                col,
                                value=default_val
                            )
                        )

                if st.button("Predict"):

                    # Goes through the backend API
                    # so the result is validated server-side
                    # and saved to this user's prediction history.

                    resp = api_client.predict(
                        st.session_state.auth_token,
                        inputs
                    )

                    if resp.get("success"):

                        data = resp["data"]

                        st.metric(
                            "Prediction",
                            data["prediction"]
                        )

                        st.metric(
                            "Probability of Leaving",
                            f"{data['probability']:.1%}"
                        )

                        st.metric(
                            "Risk Level",
                            data["risk_level"]
                        )

                        st.caption(
                            "Saved to your history at "
                            f"{data['timestamp']}"
                        )

                    elif resp.get("_status") == 401:

                        st.error(
                            "Your session has expired. "
                            "Please log in again from the sidebar."
                        )

                        st.session_state.auth_token = None
                        st.session_state.current_user = None

                    else:

                        st.error(
                            resp.get(
                                "message",
                                "Prediction failed"
                            )
                        )


# ============================================================
# TAB 4: MY HISTORY
# ============================================================

with tab4:

    st.subheader(
        "My Prediction History"
    )

    if st.session_state.auth_token is None:

        st.warning(
            "🔒 Please log in from the sidebar "
            "to view your prediction history."
        )

    else:

        if st.button(
            "🔄 Refresh history"
        ):

            st.rerun()

        resp = api_client.get_history(
            st.session_state.auth_token
        )

        if resp.get("success"):

            records = resp["data"]["history"]

            if not records:

                st.info(
                    "No predictions yet. "
                    "Run one from the "
                    "'Predict Attrition' tab."
                )

            else:

                hist_df = pd.DataFrame(
                    [
                        {
                            "Date": r["createdAt"],
                            "Prediction": r["prediction"],
                            "Probability": r["probability"],
                            "Risk Level": r["riskLevel"],
                            "id": r["id"],
                        }
                        for r in records
                    ]
                )

                st.dataframe(
                    hist_df.drop(
                        columns=["id"]
                    ),
                    use_container_width=True
                )

                st.markdown(
                    "##### Delete a record"
                )

                del_id = st.selectbox(
                    "Select a record ID to delete",
                    hist_df["id"].tolist()
                )

                if st.button(
                    "🗑️ Delete selected record"
                ):

                    del_resp = (
                        api_client.delete_prediction(
                            st.session_state.auth_token,
                            del_id
                        )
                    )

                    if del_resp.get("success"):

                        st.success(
                            "Deleted."
                        )

                        st.rerun()

                    else:

                        st.error(
                            del_resp.get(
                                "message",
                                "Delete failed"
                            )
                        )

        elif resp.get("_status") == 401:

            st.error(
                "Your session has expired. "
                "Please log in again from the sidebar."
            )

            st.session_state.auth_token = None
            st.session_state.current_user = None

        else:

            st.error(
                resp.get(
                    "message",
                    "Could not load history"
                )
            )


# ============================================================
# TAB 5: ABOUT
# ============================================================

with tab5:

    st.markdown(
        """
### HR Analytics Dashboard with Employee Attrition Prediction

This end-to-end system:

- Cleans and preprocesses employee HR data

- Trains and compares Logistic Regression, Random Forest,
  and Gradient Boosting classifiers

- Evaluates models with Accuracy, Precision, Recall,
  F1-score, and ROC-AUC

- Predicts attrition risk for individual employees
  or full batches

- Visualizes attrition trends by department, job role,
  overtime, and age

- Requires login for single-employee predictions
  and keeps a private, per-user history of every
  prediction made (backed by a FastAPI + JWT +
  database backend in `backend/`)

Built to satisfy the functional requirements (FR-1 to FR-8)
defined in the project SRS.
"""
    )