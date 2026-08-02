from pydantic import BaseModel, Field

# Esquema de datos de entrada que la API esperará recibir
class CreditRequest(BaseModel):
    ingresos_mensuales: float = Field(..., gt=0, example=3500.0)
    monto_solicitado: float = Field(..., gt=0, example=10000.0)
    edad: int = Field(..., ge=18, le=100, example=32)
    estado_civil: str = Field(..., example="Soltero")
    historial_crediticio: str = Field(..., example="Bueno")

# Esquema de la respuesta que devolverá la API
class CreditResponse(BaseModel):
    prediccion: int
    probabilidad_default: float
    nivel_riesgo: str