"""
predict.py
----------
Load the trained attrition model and generate predictions for new employee
data (single record or a CSV of many records).

Usage:
    # Predict for a whole CSV of employees (no Attrition column needed):
    python predict.py --input new_employees.csv --output predictions.csv

    # Quick single-employee prediction via a JSON string:
    python predict.py --json '{"Age":34, "BusinessTravel":"Travel_Rarely", ...}'
"""

import argparse
import json

import joblib
import pandas as pd

MODEL_PATH = "attrition_model.joblib"
METADATA_PATH = "attrition_model_metadata.json"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return model, metadata


def align_columns(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Make sure incoming data has exactly the columns the model expects."""
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input data is missing required columns: {missing}\n"
            f"Expected columns: {feature_columns}"
        )
    extra = [c for c in df.columns if c not in feature_columns]
    return df[feature_columns].copy(), extra


def predict_dataframe(df: pd.DataFrame, model, metadata: dict) -> pd.DataFrame:
    feature_columns = metadata["feature_columns"]
    aligned, extra = align_columns(df, feature_columns)
    if extra:
        print(f"Note: ignoring extra columns not used by the model: {extra}")

    probabilities = model.predict_proba(aligned)[:, 1]
    predictions = model.predict(aligned)

    result = df.copy()
    result["Attrition_Prediction"] = ["Yes" if p == 1 else "No" for p in predictions]
    result["Attrition_Probability"] = probabilities.round(4)
    result["Risk_Level"] = pd.cut(
        probabilities,
        bins=[-0.01, 0.3, 0.6, 1.01],
        labels=["Low", "Medium", "High"],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict employee attrition")
    parser.add_argument("--input", help="Path to input CSV with employee records")
    parser.add_argument("--output", default="predictions.csv", help="Where to save results")
    parser.add_argument("--json", help="Single employee record as a JSON string")
    args = parser.parse_args()

    model, metadata = load_artifacts()

    if args.json:
        record = json.loads(args.json)
        df = pd.DataFrame([record])
        result = predict_dataframe(df, model, metadata)
        row = result.iloc[0]
        print(f"\nPrediction: {row['Attrition_Prediction']}")
        print(f"Probability of leaving: {row['Attrition_Probability']:.2%}")
        print(f"Risk level: {row['Risk_Level']}")
        return

    if args.input:
        df = pd.read_csv(args.input)
        result = predict_dataframe(df, model, metadata)
        result.to_csv(args.output, index=False)
        print(f"Predictions saved to: {args.output}")
        print(result[["Attrition_Prediction", "Attrition_Probability", "Risk_Level"]].head(10))
        return

    parser.error("Provide either --input <csv> or --json '<record>'")


if __name__ == "__main__":
    main()
