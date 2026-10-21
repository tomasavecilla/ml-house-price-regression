# Misma versión de Python que la del entrenamiento; slim trae lo justo
FROM python:3.14-slim

# Todo lo que sigue ocurre dentro de /app
WORKDIR /app

# Primero las dependencias: cambian poco, así que Docker reutiliza esta
# capa (cache) aunque después se modifique inferencia.py
COPY requirements.txt .
# --no-cache-dir: pip no guarda los paquetes descargados, imagen más
# chica
RUN pip install --no-cache-dir -r requirements.txt

# Recién ahora el código y el modelo entrenado
COPY inferencia.py modelo.joblib ./

# El contenedor es el script: lo que se pase a docker run son sus
# argumentos (casas.csv precios.csv)
ENTRYPOINT ["python", "inferencia.py"]
