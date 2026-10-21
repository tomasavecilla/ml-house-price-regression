# Se construye desde la raíz del repo, porque necesita archivos de api/
# y de docker/:  docker build -f api/Dockerfile -t precio-casas-api .
FROM python:3.14-slim

WORKDIR /app

# Dependencias primero, para aprovechar el cache de capas
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# El código de la API y el mismo binario que usa la inferencia por lotes
COPY api/main.py .
COPY docker/modelo.joblib .
# main.py lee la ruta del modelo de esta variable
ENV MODELO=/app/modelo.joblib

# 0.0.0.0: escuchar en todas las interfaces, si no el puerto no se ve
# desde afuera del contenedor
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
