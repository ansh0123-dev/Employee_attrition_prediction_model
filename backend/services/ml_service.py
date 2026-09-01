"""
services/ml_service.py
-----------------------
Thin wrapper around the project's EXISTING predict.py — this module does not
reimplement prediction logic. It imports predict.py's own load_artifacts()
and predict_dataframe() functions directly, so the FastAPI and the
command-line tool always run the exact same code path against the exact
same trained model (attrition_model.joblib).
"""

import os
import sys

import pandas as pd

# predict.py lives one level up from backend/ (the project root).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from predict import load_artifacts, predict_dataframe  # noqa: E402  (existing project code, reused as-is)


class MLService:
    def __init__(self):
        # Loads the SAME model + metadata that predict.py's CLI uses.
        self.model, self.metadata = load_artifacts()

    @property
    def feature_columns(self):
        return self.metadata["feature_columns"]

    def validate_input(self, payload: dict):
        """Returns (is_valid, errors_dict_or_None)."""
        missing = [c for c in self.feature_columns if c not in payload]
        if missing:
            return False, {"missing_fields": missing}
        return True, None

    def predict(self, payload: dict) -> dict:
        """Runs predict.py's own predict_dataframe() on a single employee
        record and returns a structured result for the API response."""
        df = pd.DataFrame([{col: payload[col] for col in self.feature_columns}])
        result_df = predict_dataframe(df, self.model, self.metadata)
        row = result_df.iloc[0]

        return {
            "prediction": row["Attrition_Prediction"],
            "probability": round(float(row["Attrition_Probability"]), 4),
            "risk_level": str(row["Risk_Level"]),
        }


# Loaded once at process startup and reused for every request.
ml_service = MLService()
