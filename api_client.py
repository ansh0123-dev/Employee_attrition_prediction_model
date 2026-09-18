"""
api_client.py
-------------
Thin wrapper around the FastAPI backend's REST API, used by the Streamlit
frontend (app.py). Keeps all HTTP/auth-header plumbing in one place.
"""

import os
import requests

BACKEND_URL = os.environ.get(
    "BACKEND_URL",
    "http://localhost:8000"
).rstrip("/")

TIMEOUT = 10


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _handle(resp: requests.Response) -> dict:
    try:
        body = resp.json()
    except ValueError:
        body = {
            "success": False,
            "message": f"Unexpected server response ({resp.status_code})"
        }

    body["_status"] = resp.status_code
    return body


def is_backend_up() -> bool:
    try:
        r = requests.get(
            f"{BACKEND_URL}/api/health",
            timeout=3
        )
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def register(name: str, email: str, password: str) -> dict:
    try:
        r = requests.post(
            f"{BACKEND_URL}/api/auth/register",
            json={
                "name": name,
                "email": email,
                "password": password
            },
            timeout=TIMEOUT
        )
        return _handle(r)

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": f"Could not reach backend: {e}",
            "_status": 0
        }


def login(email: str, password: str) -> dict:
    try:
        r = requests.post(
            f"{BACKEND_URL}/api/auth/login",
            json={
                "email": email,
                "password": password
            },
            timeout=TIMEOUT
        )
        return _handle(r)

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": f"Could not reach backend: {e}",
            "_status": 0
        }


def logout(token: str) -> dict:
    try:
        r = requests.post(
            f"{BACKEND_URL}/api/auth/logout",
            headers=_auth_headers(token),
            timeout=TIMEOUT
        )
        return _handle(r)

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": f"Could not reach backend: {e}",
            "_status": 0
        }


def get_me(token: str) -> dict:
    try:
        r = requests.get(
            f"{BACKEND_URL}/api/auth/me",
            headers=_auth_headers(token),
            timeout=TIMEOUT
        )
        return _handle(r)

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": f"Could not reach backend: {e}",
            "_status": 0
        }


def predict(token: str, payload: dict) -> dict:
    try:
        r = requests.post(
            f"{BACKEND_URL}/api/predict",
            json=payload,
            headers=_auth_headers(token),
            timeout=TIMEOUT
        )
        return _handle(r)

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": f"Could not reach backend: {e}",
            "_status": 0
        }


def get_history(token: str) -> dict:
    try:
        r = requests.get(
            f"{BACKEND_URL}/api/predictions/history",
            headers=_auth_headers(token),
            timeout=TIMEOUT
        )
        return _handle(r)

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": f"Could not reach backend: {e}",
            "_status": 0
        }


def delete_prediction(token: str, record_id: str) -> dict:
    try:
        r = requests.delete(
            f"{BACKEND_URL}/api/predictions/{record_id}",
            headers=_auth_headers(token),
            timeout=TIMEOUT
        )
        return _handle(r)

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": f"Could not reach backend: {e}",
            "_status": 0
        }