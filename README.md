# HR Analytics Dashboard — Employee Attrition Prediction (End-to-End)

A working, runnable version of the system described in your SRS / business
research document. It takes employee data in, trains a machine-learning
model, and gives you attrition predictions and a dashboard out.

## What's included

| File | Purpose |
|---|---|
| `employee_data.csv` | The IBM HR Attrition dataset you uploaded (used to train the default model) |
| `train_model.py` | Cleans data, trains 3 models (Logistic Regression, Random Forest, Gradient Boosting), evaluates them, and saves the best one |
| `predict.py` | Command-line script — feed it new employee data (CSV or single JSON record), get predictions back |
| `app.py` | Interactive **Streamlit dashboard**: upload data, view HR analytics charts, retrain the model, predict attrition (batch or single employee), download results |
| `attrition_model.joblib` | The already-trained model (pipeline: preprocessing + classifier) |
| `attrition_model_metadata.json` | Model metadata: which model won, its scores, and the exact feature columns it expects |
| `requirements.txt` | Python packages needed |

This maps directly onto your SRS functional requirements:
FR-1 Upload → FR-2 Clean/preprocess → FR-3 HR analysis → FR-4/5 Train &
evaluate models → FR-6 Predict attrition → FR-7 Dashboard/charts → FR-8
Downloadable report.

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Train the model (already done once, but you can rerun anytime)

```bash
python train_model.py --data employee_data.csv
```

This prints Accuracy / Precision / Recall / F1 / ROC-AUC for all three
models, picks the best by ROC-AUC, and saves `attrition_model.joblib` +
`attrition_model_metadata.json`.

Current results on this dataset:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.752 | 0.349 | 0.638 | 0.451 | 0.803 |
| Random Forest | 0.844 | 0.522 | 0.255 | 0.343 | 0.789 |
| **Gradient Boosting (selected)** | **0.857** | **0.667** | 0.213 | 0.323 | **0.810** |

Note: attrition datasets are naturally imbalanced (~16% leave), so recall on
the "Left" class is the number to watch if the priority is catching every
at-risk employee — Logistic Regression trades some accuracy for much higher
recall (63.8%) and may be preferable for that use case. You can swap the
model-selection metric in `train_model.py` (`max(results, key=...)`) if you'd
rather optimize for recall than ROC-AUC.

## 3. Get predictions for new employees

**Batch, from a CSV:**
```bash
python predict.py --input new_employees.csv --output predictions.csv
```

**Single employee, quick check:**
```bash
python predict.py --json '{"Age":34,"BusinessTravel":"Travel_Rarely","DailyRate":1000,"Department":"Sales","DistanceFromHome":5,"Education":3,"EducationField":"Marketing","EnvironmentSatisfaction":3,"Gender":"Male","HourlyRate":60,"JobInvolvement":3,"JobLevel":2,"JobRole":"Sales Executive","JobSatisfaction":3,"MaritalStatus":"Married","MonthlyIncome":6000,"MonthlyRate":15000,"NumCompaniesWorked":2,"OverTime":"Yes","PercentSalaryHike":13,"PerformanceRating":3,"RelationshipSatisfaction":3,"StockOptionLevel":0,"TotalWorkingYears":8,"TrainingTimesLastYear":2,"WorkLifeBalance":3,"YearsAtCompany":5,"YearsInCurrentRole":3,"YearsSinceLastPromotion":1,"YearsWithCurrManager":3}'
```

The input file/record does **not** need an `Attrition` column — that's what
gets predicted. Extra columns (like `EmployeeNumber`) are ignored automatically.

Output columns added:
- `Attrition_Prediction` — Yes / No
- `Attrition_Probability` — model's confidence the employee will leave (0–1)
- `Risk_Level` — Low / Medium / High, bucketed from the probability

## 4. Run the full interactive dashboard

```bash
streamlit run app.py
```

This opens in your browser with four tabs:
1. **Dashboard / EDA** — attrition rate, key metrics, charts by department, job role, overtime, age
2. **Train Model** — upload any labeled CSV and retrain from the UI
3. **Predict Attrition** — upload a batch of employees to score, or fill in a form for one employee, and download results
4. **About** — how it maps to your SRS

## Notes on productionizing further

- Swap `employee_data.csv` for your organization's real HRMS export — the
  pipeline only assumes the same column names as the IBM dataset; adjust
  `DROP_COLS` in `train_model.py` if your columns differ.
- To deploy the dashboard for your team, host `app.py` on Streamlit
  Community Cloud, or containerize it (Docker) and deploy behind your
  internal auth — the current SRS lists "Secure employee data" as a
  non-functional requirement, so add authentication before exposing it
  beyond local use.
- For the "Automated model retraining" future enhancement in your SRS, you
  could schedule `train_model.py` to run periodically (e.g. via cron or an
  Airflow DAG) as new HR data arrives.
