import streamlit as st
import requests

st.title("CrediPulse 💳 - Evaluación de Riesgo Crediticio")

# Entradas en la interfaz
monthly_income = st.number_input("Ingresos Mensuales ($)", min_value=100.0, value=3500.0)
loan_amount = st.number_input("Monto del Crédito ($)", min_value=500.0, value=10000.0)
age = st.slider("Edad", 18, 100, 32)
employment_status = st.selectbox("Estado Laboral", ["Employed", "Unemployed", "Self-Employed"])
education_level = st.selectbox("Nivel Educativo", ["Bachelor", "Master", "High School", "PhD"])
loan_purpose = st.selectbox("Propósito del Préstamo", ["Personal", "Car", "Home", "Education"])
region = st.selectbox("Región", ["Urban", "Rural", "Suburban"])

if st.button("Evaluar Riesgo"):
    payload = {
        "monthly_income": monthly_income,
        "loan_amount": loan_amount,
        "age": age,
        "employment_status": employment_status,
        "education_level": education_level,
        "loan_purpose": loan_purpose,
        "region": region
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if response.status_code == 200:
            data = response.json()
            st.success(f"Nivel de Riesgo: **{data['nivel_riesgo']}**")
            st.metric("Probabilidad de Default", f"{data['probabilidad_default'] * 100:.1f}%")
        else:
            st.error(f"Error en la API: {response.text}")
    except Exception as e:
        st.error(f"No se pudo conectar con la API: {e}")