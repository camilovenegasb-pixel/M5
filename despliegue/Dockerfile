# Imagen base ligera de Python
FROM python:3.10-slim

# Directorio de trabajo en el contenedor
WORKDIR /app

# Copiar e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente de la aplicación
COPY model_deploy.py .

# Exponer el puerto de Uvicorn/FastAPI
EXPOSE 8000

# Comando para iniciar la API
CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]