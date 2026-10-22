# La imagen oficial de MLflow no trae dos cosas que necesitamos: boto3, para
# guardar los artefactos en S3, y Flask-WTF, que pide la autenticación
# (el extra mlflow[auth]). Se agrega eso y nada más, con versiones fijas.
# La versión de MLflow tiene que ser la misma que fija requirements.txt en
# los clientes.
FROM ghcr.io/mlflow/mlflow:v3.17.1

# Un usuario común en vez de root: si alguien rompe el servidor, no queda
# con permisos de administrador dentro del contenedor. /datos es donde el
# servidor guarda sus bases sqlite (en producción es una carpeta del host)
RUN pip install --no-cache-dir "mlflow[auth]==3.17.1" Flask-WTF==1.3.0 boto3==1.43.117 \
    && useradd --uid 1000 --create-home mlflow \
    && mkdir /datos \
    && chown mlflow /datos
USER mlflow
WORKDIR /datos
