from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os
from app.schemas import CreditRequest, CreditResponse

app = FastAPI(
    title="CrediPulse - API de Scoring Crediticio",
    description="API REST para evaluación de riesgo crediticio",
    version="1.2.0"
)

# Ruta del modelo
MODEL_PATH = os.path.join("models", "pipeline_model.pkl")

try:
    model_pipeline = joblib.load(MODEL_PATH)
except Exception:
    model_pipeline = None

@app.get("/health", tags=["Estado"])
def health_check():
    return {"status": "ok", "model_loaded": model_pipeline is not None}

@app.post("/predict", response_model=CreditResponse, tags=["Inferencia"])
def predict_credit_risk(data: CreditRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="El modelo no está disponible.")
    
    input_data = pd.DataFrame([data.dict()])
    prediction = int(model_pipeline.predict(input_data)[0])
    prob_default = float(model_pipeline.predict_proba(input_data)[0][1])
    
    riesgo = "Bajo" if prob_default < 0.30 else ("Medio" if prob_default < 0.70 else "Alto")
    
    return CreditResponse(
        prediccion=prediction,
        probabilidad_default=round(prob_default, 4),
        nivel_riesgo=riesgo
    )