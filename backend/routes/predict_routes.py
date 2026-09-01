"""Prediction and per-user prediction-history endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from auth import get_current_user
from extensions import get_db
from models import PredictionHistory, User
from schemas import PredictionRequest
from services.ml_service import ml_service
from utils.responses import success

router = APIRouter(prefix="/api", tags=["Predictions"])


@router.post("/predict", status_code=status.HTTP_201_CREATED)
def predict(
    payload: PredictionRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    data = payload.as_dict()
    is_valid, errors = ml_service.validate_input(data)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail={"message": "Invalid input for prediction", "errors": errors},
        )

    try:
        result = ml_service.predict(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model error while generating prediction: {exc}")

    record = PredictionHistory(
        user_id=user.id,
        input_data=data,
        prediction=result["prediction"],
        probability=result["probability"],
        risk_level=result["risk_level"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return success(
        "Prediction successful",
        {
            "id": record.id,
            "prediction": result["prediction"],
            "probability": result["probability"],
            "risk_level": result["risk_level"],
            "timestamp": record.created_at.isoformat(),
        },
    )


@router.get("/predictions/history")
def history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    records = (
        db.query(PredictionHistory)
        .filter(PredictionHistory.user_id == user.id)
        .order_by(PredictionHistory.created_at.desc())
        .all()
    )
    return success("Prediction history", {"history": [r.to_dict() for r in records]})


@router.get("/predictions/{record_id}")
def get_one(
    record_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    record = db.query(PredictionHistory).filter(PredictionHistory.id == record_id).first()
    if not record or record.user_id != user.id:
        raise HTTPException(status_code=404, detail="Prediction record not found")
    return success("Prediction record", record.to_dict())


@router.delete("/predictions/{record_id}")
def delete_one(
    record_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    record = db.query(PredictionHistory).filter(PredictionHistory.id == record_id).first()
    if not record or record.user_id != user.id:
        raise HTTPException(status_code=404, detail="Prediction record not found")
    db.delete(record)
    db.commit()
    return success("Prediction record deleted")
