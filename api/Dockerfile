# Se construye desde la raíz del repo, porque necesita archivos de api/
# y de docker/:  docker build -f api/Dockerfile -t precio-casas-api .
FROM python:3.14-slim

WORKDIR /app

# Dependencias primero, para aprovechar el cache de capas
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Misma estructura que el repo (api/ y docker/ lado a lado): main.py
# importa docker/inferencia.py, el mismo código y el mismo binario que
# usa la inferencia por lotes
COPY api/main.py api/
COPY docker/inferencia.py docker/modelo.joblib docker/

# 0.0.0.0: escuchar en todas las interfaces, si no el puerto no se ve
# desde afuera del contenedor
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
