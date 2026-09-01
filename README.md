# HR Analytics Dashboard — Employee Attrition Prediction (End-to-End)

A working, runnable version of the system described in your SRS / business
research document. It takes employee data in, trains a machine-learning
model, and gives you attrition predictions and a dashboard out — now with a
full backend: **login, sessions, a prediction API, and per-user history**.

## What's included

| File / folder | Purpose |
|---|---|
| `employee_data.csv` | The IBM HR Attrition dataset you uploaded (used to train the default model) |
| `train_model.py` | Cleans data, trains 3 models (Logistic Regression, Random Forest, Gradient Boosting), evaluates them, and saves the best one |
| `predict.py` | Command-line script — feed it new employee data (CSV or single JSON record), get predictions back |
| `app.py` | Interactive **Streamlit dashboard**: login/signup, upload data, view HR analytics charts, retrain the model, predict attrition (batch or single employee), view your prediction history |
| `api_client.py` | Thin HTTP client the Streamlit app uses to call the backend |
| `attrition_model.joblib` | The already-trained model (pipeline: preprocessing + classifier) |
| `attrition_model_metadata.json` | Model metadata: which model won, its scores, and the exact feature columns it expects |
| `requirements.txt` | Python packages needed for the frontend |
| `backend/` | **New**: FastAPI API — auth (register/login/logout), JWT sessions, `/api/predict`, prediction history, database. See `backend/README.md` for full API docs. |

## Architecture

```
Streamlit (app.py)  <--HTTP + JWT-->  FastAPI API (backend/app.py)  -->  SQLite/Postgres
                                              |
                                              v
                                  attrition_model.joblib (your trained model)
```

The Streamlit app is still the same dashboard as before (EDA, training,
batch prediction all work exactly as they did). Login and single-employee
prediction now go through the new backend, which validates the request,
runs your **actual trained model** (by importing `predict.py`'s own
functions — not a reimplementation), and saves the result to that user's
private history in a database.

## API list

| Method | Endpoint | Auth required | Purpose |
|---|---|---|---|
| POST | `/api/auth/register` | No | Create a new user account |
| POST | `/api/auth/login` | No | Log in, returns a JWT |
| POST | `/api/auth/logout` | Yes | Invalidate the current token (server-side) |
| GET | `/api/auth/me` | Yes | Get the current logged-in user |
| POST | `/api/predict` | Yes | Run the real ML model, save to history |
| GET | `/api/predictions/history` | Yes | Get your own prediction history |
| GET/DELETE | `/api/predictions/<id>` | Yes | Get/delete one of your own records |
| GET | `/api/health` | No | Liveness check |

Full request/response examples: `backend/README.md`.

This maps directly onto your SRS functional requirements:
FR-1 Upload → FR-2 Clean/preprocess → FR-3 HR analysis → FR-4/5 Train &
evaluate models → FR-6 Predict attrition → FR-7 Dashboard/charts → FR-8
Downloadable report.

## 1. Setup

**Backend (do this first):**
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"   # paste into .env as JWT_SECRET_KEY
python app.py
```
Leave this running — it serves the API on `http://localhost:8000`.
Full details, all endpoints, and a curl walkthrough: see `backend/README.md`.

**Frontend (in a second terminal):**
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

Current results on this dataset (retrained with scikit-learn 1.9.0 — see note below):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.752 | 0.349 | 0.638 | 0.451 | 0.803 |
| Random Forest | 0.823 | 0.449 | 0.468 | 0.458 | 0.793 |
| **Gradient Boosting (selected)** | **0.857** | **0.667** | 0.213 | 0.323 | **0.810** |

> **Notes on this copy:** two things were fixed here before packaging.
> (1) `attrition_model.joblib` was pickled with an older scikit-learn version
> and threw `ModuleNotFoundError: No module named '_loss'` on load with a
> newer one — it's been retrained and re-pickled with scikit-learn 1.9.0, and
> both `requirements.txt` files now pin `scikit-learn==1.9.0` to match. If you
> ever see that error again, rerun `python train_model.py --data
> employee_data.csv` to regenerate the model against whatever scikit-learn
> version you have installed. (2) `backend/app.py` was missing its
> `if __name__ == "__main__":` block, so `python app.py` defined the FastAPI
> app and exited immediately instead of serving anything — that block has
> been added so it now starts uvicorn on `HOST`/`PORT` from `.env` as this
> README describes.

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

Make sure the backend (step 1) is already running, then in the frontend terminal:

```bash
streamlit run app.py
```

This opens in your browser with five tabs:
1. **Dashboard / EDA** — attrition rate, key metrics, charts by department, job role, overtime, age
2. **Train Model** — upload any labeled CSV and retrain from the UI
3. **Predict Attrition** — upload a batch of employees to score (no login needed, unchanged from before), or **log in** and fill in a form to predict one employee through the backend API (saved to your history)
4. **My History** — every prediction you've made, fetched from your account; delete records you don't need
5. **About** — how it maps to your SRS

**Sign up first**: use the "Sign up" option in the sidebar to create an
account, then log in. Single-employee prediction and history are gated
behind login; the Dashboard, Train Model, and batch-CSV prediction tabs work
without logging in, exactly as before.

If `BACKEND_URL` isn't set, the frontend assumes the backend is at
`http://localhost:8000`. To point it elsewhere: `export BACKEND_URL=http://your-host:5000`
before running `streamlit run app.py`.

## 5. Demonstrating the full flow in a meeting

This exact sequence is what to click through live:

1. **Signup screen** — sidebar → "Sign up" → fill name/email/password → "Create account"
2. **Login screen** — sidebar → "Login" → same email/password → "Log in". Sidebar now shows **"Logged in as ..."** with your email — this is the logged-in user info + confirms the JWT/session was created.
3. **Predict Attrition tab** → "Single employee (manual entry)" → the form is the **prediction input** → fill it in (or leave defaults) → click **Predict**.
4. **Prediction result** appears immediately (Prediction / Probability / Risk Level) plus a caption showing the exact timestamp it was saved.
5. **My History tab** → click "Refresh history" → shows every prediction you've made, each with its timestamp, prediction, probability, and risk level — proves it's reading from SQLite via `GET /api/predictions/history`, not just local session state.
6. **Logout** — sidebar → "Log out" button.
7. **Prove logout actually invalidated the session**: try opening "My History" again — it re-fetches from the backend and will show "session has expired" once the stored token is cleared, or you can show it more explicitly with curl (see `backend/README.md` — send the same token after logout, backend returns `401`).

For extra credibility, open two terminals side by side: one running `python backend/app.py` so the class can see live request logs, and one running `streamlit run app.py`.

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
