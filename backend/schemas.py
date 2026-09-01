"""Pydantic request/response schemas used by FastAPI."""

from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=6)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class PredictionRequest(BaseModel):
    # The ML model's feature list is read from attrition_model_metadata.json.
    # Extra fields are allowed so the API remains compatible with that
    # authoritative model metadata without duplicating the feature list here.
    model_config = ConfigDict(extra="allow")

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=False)
