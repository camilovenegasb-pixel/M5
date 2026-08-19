import os
import io
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(
    title="API de Despliegue de Modelo Crediticio - PI M5",
    description="Endpoint para predicciones individuales y por lotes (batch) de riesgo crediticio.",
    version="1.0.0"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_credito.pkl")

def load_or_create_model():
    """
    Intenta cargar el modelo guardado. Si no existe, entrena un modelo de demostración.
    """
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error cargando modelo desde {MODEL_PATH}: {e}")
            
    # Modelo de contingencia/demo basado en variables principales
    print("⚠️ No se encontró modelo previo. Entrenando modelo de respaldo demostrativo...")
    X_demo = np.random.rand(100, 5)
    y_demo = np.random.choice([0, 1], size=100)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_demo, y_demo)
    return model

model = load_or_create_model()

# --- Esquemas Pydantic ---
class CreditInput(BaseModel):
    capital_prestado: float = Field(..., example=3500000.0)
    salario_cliente: float = Field(..., example=4000000.0)
    cuota_pactada: float = Field(..., example=300000.0)
    edad_cliente: int = Field(..., example=35)
    puntaje_datacredito: float = Field(..., example=720.0)

class BatchCreditInput(BaseModel):
    records: List[CreditInput]

# --- Endpoints ---

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "API de Predicción Crediticia - MLOps",
        "documentation": "/docs"
    }

@app.post("/predict")
def predict_single(data: CreditInput):
    """
    Predicción individual desde un objeto JSON.
    """
    try:
        features = np.array([[
            data.capital_prestado,
            data.salario_cliente,
            data.cuota_pactada,
            data.edad_cliente,
            data.puntaje_datacredito
        ]])
        
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0].tolist() if hasattr(model, "predict_proba") else [0.5, 0.5]
        
        return {
            "prediction": prediction,
            "label": "Aprobado / Al Día" if prediction == 1 else "Riesgo / Incumplimiento",
            "probability_score": round(probabilities[1], 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/predict_batch")
def predict_batch(batch: BatchCreditInput):
    """
    Predicción por lotes recibiendo una lista JSON de registros.
    """
    try:
        features = np.array([
            [r.capital_prestado, r.salario_cliente, r.cuota_pactada, r.edad_cliente, r.puntaje_datacredito]
            for r in batch.records
        ])
        
        predictions = model.predict(features).tolist()
        
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "record_index": i,
                "prediction": int(pred),
                "label": "Aprobado / Al Día" if pred == 1 else "Riesgo / Incumplimiento"
            })
            
        return {
            "total_records": len(results),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en procesar lote: {str(e)}")

@app.post("/predict_batch_csv")
async def predict_batch_csv(file: UploadFile = File(...)):
    """
    Soporte para carga de archivos CSV masivos.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV válido.")
        
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    required_cols = ['capital_prestado', 'salario_cliente', 'cuota_pactada', 'edad_cliente', 'puntaje_datacredito']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        raise HTTPException(status_code=400, detail=f"Columnas faltantes en el CSV: {missing}")
        
    features = df[required_cols].values
    preds = model.predict(features)
    
    df['prediccion'] = preds
    df['resultado_label'] = np.where(df['prediccion'] == 1, "Aprobado / Al Día", "Riesgo / Incumplimiento")
    
    return {
        "filename": file.filename,
        "total_filas_procesadas": len(df),
        "resumen_predicciones": df['resultado_label'].value_counts().to_dict(),
        "preview": df[['capital_prestado', 'puntaje_datacredito', 'prediccion', 'resultado_label']].head(10).to_dict(orient="records")
    }