"""Experimental FastAPI service for the fitted credit risk scoring pipeline.

The trained .joblib must be supplied separately by the operator.
Never load a model artifact obtained from an untrusted source.
"""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.scoring import predict_credit_risk, validate_model

logger = logging.getLogger(__name__)


class CreditFeatures(BaseModel):
    """18 pre-credit features, without customer identifiers or outcome labels."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    ingresos_medios_6m_eur: float = Field(ge=0)
    gastos_ordinarios_medios_6m_eur: float = Field(ge=0)
    variabilidad_ingresos_6m: float = Field(ge=0)
    saldo_total_ultimo_mes_eur: float
    limite_credito_total_ultimo_mes_eur: float = Field(ge=0)
    credito_utilizado_ultimo_mes_eur: float = Field(ge=0)
    utilizacion_credito_previa: float = Field(ge=0)
    cuota_deuda_externa_eur: float = Field(ge=0)
    mora_previa_max_dias: int = Field(ge=0)
    dti_con_nueva_cuota: float = Field(ge=0)
    margen_libre_tras_prestamo_eur: float
    accesos_digitales_6m: int = Field(ge=0)
    antiguedad_laboral_meses: int = Field(ge=0)
    situacion_laboral: Literal[
        "ASALARIADO", "AUTONOMO", "PENSIONISTA", "TEMPORAL"
    ]
    importe_solicitado_eur: float = Field(gt=0)
    plazo_meses: int = Field(gt=0)
    cuota_propuesta_eur: float = Field(ge=0)
    tasa_nominal_anual_simulada: float = Field(ge=0)


class Prediction(BaseModel):
    probabilidad_incumplimiento: float = Field(ge=0, le=1)
    score_crediticio: int = Field(ge=300, le=850)
    modelo_version: str


def load_trusted_model() -> object | None:
    """Load once at startup; absent/incompatible artifacts disable predictions."""
    model_path = Path(
        os.environ.get("MODEL_PATH", "models/modelo_scoring_crediticio_v1.joblib")
    )
    if not model_path.is_file():
        logger.warning("No trained model artifact supplied. Inference disabled.")
        return None
    try:
        model = joblib.load(model_path)
        validate_model(model)
    except Exception as exc:
        logger.error(
            "Model unavailable or incompatible (%s). Inference disabled.",
            type(exc).__name__,
        )
        return None
    return model


def create_app(model: object | None = None) -> FastAPI:
    """Optional model injection is intended solely for isolated unit tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = model if model is not None else load_trusted_model()
        if app.state.model is not None:
            validate_model(app.state.model)
        yield

    app = FastAPI(
        title="Bank Risk Scoring API",
        description=(
            "Academic prototype using synthetic bank data. Not suitable for "
            "automated lending decisions or production credit scoring."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health/live")
    def liveness() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    def readiness(request: Request) -> dict[str, str]:
        if request.app.state.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model artifact is unavailable; inference is disabled.",
            )
        return {"status": "ready"}

    @app.post("/predict", response_model=Prediction)
    def predict(payload: CreditFeatures, request: Request) -> Prediction:
        if request.app.state.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model artifact is unavailable; inference is disabled.",
            )
        try:
            result = predict_credit_risk(
                request.app.state.model,
                payload.model_dump(),
            )
            return Prediction(**result)
        except Exception as exc:
            logger.error("Inference failed (%s).", type(exc).__name__)
            raise HTTPException(
                status_code=500,
                detail="Prediction could not be completed.",
            ) from exc

    return app


app = create_app()
