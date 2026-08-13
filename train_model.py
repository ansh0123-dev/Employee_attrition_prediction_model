"""
train_model.py
----------------
HR Analytics - Employee Attrition Prediction
End-to-end training pipeline:
  1. Load raw employee data (CSV)
  2. Clean / preprocess (drop useless columns, encode categoricals, scale numerics)
  3. Train multiple ML models (Logistic Regression, Random Forest, Gradient Boosting)
  4. Evaluate each model (Accuracy, Precision, Recall, F1, ROC-AUC)
  5. Pick the best model (by ROC-AUC) and save it, along with the preprocessing
     pipeline and feature list, to disk so predict.py / app.py can reuse it.

Run:
    python train_model.py --data employee_data.csv
"""

import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

TARGET = "Attrition"

# Columns that carry no predictive signal (constant or pure ID columns)
DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.drop_duplicates()
    # Encode target: Yes -> 1, No -> 0
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    df = df.dropna(subset=[TARGET])
    return df


def build_preprocessor(df: pd.DataFrame):
    features = df.drop(columns=[TARGET])
    cat_cols = features.select_dtypes(include=["object"]).columns.tolist()
    num_cols = features.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return preprocessor, cat_cols, num_cols


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }, y_pred


def main(data_path: str, output_prefix: str = "attrition_model"):
    print(f"Loading data from: {data_path}")
    df = load_and_clean(data_path)
    print(f"Rows after cleaning: {len(df)} | Attrition rate: {df[TARGET].mean():.2%}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor, cat_cols, num_cols = build_preprocessor(df)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }

    results = {}
    fitted_pipelines = {}

    for name, clf in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        metrics, y_pred = evaluate(pipe, X_test, y_test)
        results[name] = metrics
        fitted_pipelines[name] = pipe
        print(f"\n--- {name} ---")
        for k, v in metrics.items():
            print(f"  {k:10s}: {v}")
        print(classification_report(y_test, y_pred, target_names=["Stayed", "Left"]))

    # Pick best model by ROC-AUC (robust to class imbalance)
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\n>>> Best model: {best_name} (ROC-AUC={results[best_name]['roc_auc']})")

    # Save artifacts
    model_path = f"{output_prefix}.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"Saved trained pipeline to: {model_path}")

    metadata = {
        "best_model": best_name,
        "all_results": results,
        "feature_columns": X.columns.tolist(),
        "categorical_columns": cat_cols,
        "numeric_columns": num_cols,
        "target": TARGET,
        "target_mapping": {"No": 0, "Yes": 1},
    }
    with open(f"{output_prefix}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {output_prefix}_metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HR attrition prediction model")
    parser.add_argument("--data", default="employee_data.csv", help="Path to training CSV")
    parser.add_argument("--out", default="attrition_model", help="Output filename prefix")
    args = parser.parse_args()
    main(args.data, args.out)
