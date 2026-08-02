from pydantic import BaseModel, Field

class CreditRequest(BaseModel):
    monthly_income: float = Field(..., gt=0, example=3500.0)
    loan_amount: float = Field(..., gt=0, example=10000.0)
    age: int = Field(..., ge=18, le=100, example=32)
    employment_status: str = Field(..., example="Employed")
    education_level: str = Field(..., example="Bachelor")
    loan_purpose: str = Field(..., example="Personal")
    region: str = Field(..., example="Urban")

class CreditResponse(BaseModel):
    prediccion: int
    probabilidad_default: float
    nivel_riesgo: str