# La interfaz del tasador:  docker build -t precio-casas-app app/
FROM python:3.14-slim

WORKDIR /app

# Dependencias primero, para aprovechar el cache de capas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# El código y el tema. Acá no viaja ningún modelo: la app solo habla
# con la API
COPY app.py .
COPY .streamlit .streamlit
# Dónde está la API; docker-compose.yml la sobreescribe con el nombre
# del servicio
ENV API_URL=http://localhost:8000

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", \
     "--server.address=0.0.0.0"]
