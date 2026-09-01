# Backend — FastAPI Auth + Prediction API + History

This backend is the FastAPI version of the original project. The Streamlit frontend, trained ML model, prediction logic, and API contract are kept compatible with the original project.

## Architecture

```text
Streamlit frontend
       |
       | HTTP + JSON
       v
FastAPI REST API
       |
       +-- JWT authentication
       +-- SQLAlchemy + SQLite
       +-- Existing ML model (attrition_model.joblib)
```

## Project structure

```text
backend/
  app.py                    FastAPI application
  auth.py                   JWT creation + authentication dependency
  config.py                 Environment configuration
  extensions.py             SQLAlchemy engine/session/Base
  models.py                 User, PredictionHistory, TokenBlocklist
  schemas.py                Pydantic request validation
  routes/
    auth_routes.py          /api/auth/*
    predict_routes.py       /api/predict, /api/predictions/*
  services/
    ml_service.py           Reuses the existing predict.py model code
  utils/
    responses.py            Consistent JSON response helpers
  requirements.txt
  .env.example
```

## Setup

From the project root:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Generate a secret:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Put the generated value into `JWT_SECRET_KEY` in `.env`.

## Run FastAPI

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Backend:

```text
http://localhost:8000
```

Interactive Swagger/OpenAPI documentation:

```text
http://localhost:8000/docs
```

Alternative ReDoc documentation:

```text
http://localhost:8000/redoc
```

Health check:

```text
GET http://localhost:8000/api/health
```

## API reference

| Method | Endpoint | Auth | Purpose |
|---|---|---|---|
| POST | `/api/auth/register` | No | Create account |
| POST | `/api/auth/login` | No | Log in and receive JWT |
| POST | `/api/auth/logout` | Yes | Blocklist current JWT |
| GET | `/api/auth/me` | Yes | Current user |
| POST | `/api/predict` | Yes | Run existing ML model and save history |
| GET | `/api/predictions/history` | Yes | Current user's history |
| GET | `/api/predictions/{id}` | Yes | One owned prediction |
| DELETE | `/api/predictions/{id}` | Yes | Delete one owned prediction |
| GET | `/api/health` | No | Liveness check |

Protected endpoints use:

```text
Authorization: Bearer <token>
```

## Streamlit frontend

The frontend remains Streamlit. `api_client.py` now defaults to:

```text
http://localhost:8000
```

Run it in a second terminal from the project root:

```bash
streamlit run app.py
```

## Important

The backend does **not** retrain or replace the ML model. `services/ml_service.py` continues to import `load_artifacts()` and `predict_dataframe()` from the existing root `predict.py`, so the same `attrition_model.joblib` and metadata are used.

Pydantic is used for FastAPI request validation, SQLAlchemy is used directly instead of SQLAlchemy, and JWT authentication is implemented with `PyJWT` plus a database-backed token blocklist for logout.
