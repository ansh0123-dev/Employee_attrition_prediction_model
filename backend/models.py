"""SQLAlchemy models: User, PredictionHistory, TokenBlocklist."""

import base64
import hashlib
import hmac
import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship
from extensions import Base


def _uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    predictions = relationship(
        "PredictionHistory", back_populates="user", cascade="all, delete-orphan"
    )

    def set_password(self, password: str):
        # PBKDF2 password hashing using Python's standard library.
        salt = os.urandom(16)
        iterations = 310000
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations
        )
        self.password_hash = (
            f"pbkdf2:sha256:{iterations}$"
            f"{base64.urlsafe_b64encode(salt).decode()}$"
            f"{base64.urlsafe_b64encode(digest).decode()}"
        )

    def check_password(self, password: str) -> bool:
        try:
            method, salt_b64, digest_b64 = self.password_hash.split("$", 2)
            _, algorithm, iterations_text = method.split(":", 2)
            iterations = int(iterations_text)
            salt = base64.urlsafe_b64decode(salt_b64.encode())
            expected = base64.urlsafe_b64decode(digest_b64.encode())
            actual = hashlib.pbkdf2_hmac(
                algorithm, password.encode("utf-8"), salt, iterations
            )
            return hmac.compare_digest(actual, expected)
        except (ValueError, TypeError):
            return False

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
        }


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    input_data = Column(JSON, nullable=False)
    prediction = Column(String(10), nullable=False)
    probability = Column(Float, nullable=False)
    risk_level = Column(String(10), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="predictions")

    def to_dict(self):
        return {
            "id": self.id,
            "userId": self.user_id,
            "inputData": self.input_data,
            "prediction": self.prediction,
            "probability": self.probability,
            "riskLevel": self.risk_level,
            "createdAt": self.created_at.isoformat(),
        }


class TokenBlocklist(Base):
    __tablename__ = "token_blocklist"

    id = Column(Integer, primary_key=True)
    jti = Column(String(36), nullable=False, index=True, unique=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
