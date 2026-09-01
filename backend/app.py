"""FastAPI application for authentication, prediction, and history."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import Config
from extensions import Base, engine
import models  # noqa: F401 - registers SQLAlchemy models before create_all
from routes.auth_routes import router as auth_router
from routes.predict_routes import router as predict_router
from utils.responses import error


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="HR Attrition Prediction API",
    description="REST API for authentication, employee attrition prediction, and prediction history.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(predict_router)


@app.get("/api/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content=error("Validation failed", {"validation": exc.errors()}),
    )


@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(status_code=404, content=error("Endpoint not found"))


@app.exception_handler(500)
async def server_error(request: Request, exc):
    return JSONResponse(status_code=500, content=error("Internal server error"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
