"""Authentication endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from auth import create_access_token, get_current_user, get_token_payload
from extensions import get_db
from models import TokenBlocklist, User
from schemas import LoginRequest, RegisterRequest
from utils.responses import success

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    name = request.name.strip()
    email = str(request.email).strip().lower()

    if not name:
        raise HTTPException(status_code=400, detail="Name is required")

    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    user = User(name=name, email=email)
    user.set_password(request.password)
    db.add(user)
    db.commit()
    db.refresh(user)

    return success("User registered successfully", {"user": user.to_dict()})


@router.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    email = str(request.email).strip().lower()
    user = db.query(User).filter(User.email == email).first()

    if not user or not user.check_password(request.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_access_token(user.id)
    return success(
        "Login successful",
        {"token": access_token, "user": user.to_dict()},
    )


@router.post("/logout")
def logout(
    payload=Depends(get_token_payload),
    db: Session = Depends(get_db),
):
    jti = payload["jti"]
    if not db.query(TokenBlocklist.id).filter(TokenBlocklist.jti == jti).first():
        db.add(TokenBlocklist(jti=jti))
        db.commit()
    return success("Logged out successfully")


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return success("Current user", {"user": user.to_dict()})
