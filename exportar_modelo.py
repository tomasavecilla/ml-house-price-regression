"""Exporta la versión con alias @champion a docker/modelo.joblib.

El registro de MLflow decide qué versión va a producción; este binario
es el que viaja en el contenedor.

Uso:
    python exportar_modelo.py
"""

import os

import joblib
import mlflow

NOMBRE_MODELO = "precio-casas"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
DESTINO = "docker/modelo.joblib"


def main() -> None:
    """Descarga el modelo @champion del registro y lo guarda en DESTINO.

    Raises:
        mlflow.exceptions.MlflowException: si ninguna versión del
            modelo tiene el alias champion.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    cliente = mlflow.MlflowClient()
    version = cliente.get_model_version_by_alias(NOMBRE_MODELO, "champion")
    pipeline = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@champion")
    joblib.dump(pipeline, DESTINO)
    print(f"Versión {version.version} (@champion) exportada a {DESTINO}")


if __name__ == "__main__":
    main()
