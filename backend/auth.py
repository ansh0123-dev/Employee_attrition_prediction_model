"""JWT helpers and FastAPI authentication dependency."""

from datetime import datetime, timezone
import uuid

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from jwt import InvalidTokenError
from sqlalchemy.orm import Session

from config import Config
from extensions import get_db
from models import TokenBlocklist, User

security = HTTPBearer(auto_error=False)
ALGORITHM = "HS256"


def create_access_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    expire = now + Config.JWT_ACCESS_TOKEN_EXPIRES
    payload = {
        "sub": user_id,
        "jti": str(uuid.uuid4()),
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm=ALGORITHM)


def get_token_payload(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: Session = Depends(get_db),
):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")

    try:
        payload = jwt.decode(
            credentials.credentials,
            Config.JWT_SECRET_KEY,
            algorithms=[ALGORITHM],
        )
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired authentication token.")

    jti = payload.get("jti")
    user_id = payload.get("sub")
    if not jti or not user_id:
        raise HTTPException(status_code=401, detail="Invalid authentication token.")

    revoked = db.query(TokenBlocklist.id).filter(TokenBlocklist.jti == jti).first()
    if revoked:
        raise HTTPException(status_code=401, detail="Session has been logged out. Please log in again.")

    return payload


def get_current_user(payload=Depends(get_token_payload), db: Session = Depends(get_db)) -> User:
    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    return user
