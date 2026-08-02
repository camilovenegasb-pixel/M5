import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import CreditRequest, CreditResponse

app = FastAPI(
    title="CrediPulse - API de Scoring Crediticio",
    description="API REST para evaluación de riesgo crediticio",
    version="1.2.0"
)

# 🌐 Configuración de CORS para permitir peticiones desde Streamlit/Swagger
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🎯 Ruta absoluta al modelo
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

# Carga inicial del modelo
model_pipeline = None
try:
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"✅ Modelo cargado exitosamente desde: {MODEL_PATH}")
    else:
        print(f"⚠️ El archivo no existe en la ruta: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")

@app.get("/health", tags=["Estado"])
def health_check():
    return {"status": "ok", "model_loaded": model_pipeline is not None}

@app.post("/predict", response_model=CreditResponse, tags=["Inferencia"])
def predict_credit_risk(data: CreditRequest):
    if model_pipeline is None:
        raise HTTPException(
            status_code=500, 
            detail=f"El modelo no está cargado. Revisa la ruta: {MODEL_PATH}"
        )
    
    try:
        # Compatibilidad Pydantic v1 / v2
        data_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        input_data = pd.DataFrame([data_dict])
        
        prediction = int(model_pipeline.predict(input_data)[0])
        prob_default = float(model_pipeline.predict_proba(input_data)[0][1])
        
        riesgo = "Bajo" if prob_default < 0.30 else ("Medio" if prob_default < 0.70 else "Alto")
        
        return CreditResponse(
            prediccion=prediction,
            probabilidad_default=round(prob_default, 4),
            nivel_riesgo=riesgo
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la predicción: {str(e)}")