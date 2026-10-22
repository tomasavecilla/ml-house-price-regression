"""Exporta una versión del modelo registrado a docker/modelo.joblib.

El registro de MLflow decide qué versión va a producción; este binario
es el que viaja en el contenedor. Por defecto se exporta la versión con
alias @champion; con --version, la que se indique (así se exporta una
versión antes de promoverla).

Junto al binario se escribe docker/modelo.json, con de qué versión del
registro salió: la API lo muestra, porque dos versiones pueden ser el
mismo tipo de modelo y no distinguirse de otra forma.

Uso:
    python -m entrenamiento.exportar_modelo
    python -m entrenamiento.exportar_modelo --version 21
    python -m entrenamiento.exportar_modelo --version 20 \\
        --solo-metadatos
"""

import argparse
import json
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
from mlflow.entities.model_registry import ModelVersion
from sklearn.pipeline import Pipeline

from entrenamiento import entrenar

NOMBRE_MODELO = entrenar.NOMBRE_MODELO
TRACKING_URI = entrenar.TRACKING_URI
DESTINO = entrenar.MODELO
METADATOS = DESTINO.with_suffix(".json")


def version_valida(texto: str) -> int:
    """Convierte el argumento --version en un número de versión.

    Args:
        texto: lo que se escribió en la línea de comandos.

    Returns:
        El número de versión, entero y mayor que cero.

    Raises:
        argparse.ArgumentTypeError: si no es un entero positivo.
    """
    if not texto.isdecimal() or int(texto) < 1:
        raise argparse.ArgumentTypeError(
            f"'{texto}' no es un número de versión (entero positivo)")
    return int(texto)


def es_el_mismo_modelo(nuevo: Pipeline, actual: Pipeline) -> bool:
    """Dice si dos pipelines predicen exactamente igual.

    Comparar los bytes de los archivos no sirve: el mismo modelo,
    guardado con otra versión de Python o de joblib, da bytes
    distintos. Se compara lo que importa, las predicciones, sobre las
    casas del dataset (con sus valores faltantes, para que también
    cuente la imputación).

    Args:
        nuevo: el pipeline recién descargado del registro.
        actual: el pipeline que ya está en DESTINO.

    Returns:
        True si las dos predicciones son idénticas.
    """
    casas = entrenar.leer_dataset().drop(columns=[entrenar.OBJETIVO])
    try:
        return bool(np.array_equal(nuevo.predict(casas),
                                   actual.predict(casas)))
    except ValueError:
        # Columnas distintas: no pueden ser el mismo modelo
        return False


def escribir_metadatos(cliente: mlflow.MlflowClient,
                       version: ModelVersion) -> None:
    """Guarda en METADATOS de qué versión del registro es el binario.

    No incluye la fecha de la exportación sino la del registro de la
    versión: así exportar dos veces la misma versión da el mismo
    archivo y el repo no ve ningún cambio.

    Args:
        cliente: cliente de MLflow ya conectado al registro.
        version: la versión que se exportó.
    """
    metricas = cliente.get_run(version.run_id).data.metrics
    registrada = datetime.fromtimestamp(
        version.creation_timestamp / 1000, tz=timezone.utc)
    metadatos = {
        "version": int(version.version),
        "run_id": version.run_id,
        # Una versión que no salió de nuestros scripts puede no tenerlo
        "rmse_test": (round(metricas["rmse_test"], 4)
                      if "rmse_test" in metricas else None),
        "registrada": registrada.strftime("%Y-%m-%d"),
    }
    METADATOS.write_text(json.dumps(metadatos, indent=2) + "\n")


def main() -> None:
    """Descarga una versión del registro y la guarda en DESTINO.

    Raises:
        mlflow.exceptions.MlflowException: si la versión pedida no
            existe, o si ninguna versión tiene el alias champion.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", type=version_valida,
                        help="versión a exportar (por defecto, @champion)")
    parser.add_argument("--solo-metadatos", action="store_true",
                        help="escribe modelo.json sin tocar el binario")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    cliente = mlflow.MlflowClient()
    if args.version is None:
        version = cliente.get_model_version_by_alias(NOMBRE_MODELO,
                                                     "champion")
        origen = f"models:/{NOMBRE_MODELO}@champion"
        detalle = f"{version.version} (@champion)"
    else:
        version = cliente.get_model_version(NOMBRE_MODELO,
                                            str(args.version))
        origen = f"models:/{NOMBRE_MODELO}/{args.version}"
        detalle = str(args.version)
    if not args.solo_metadatos:
        pipeline = mlflow.sklearn.load_model(origen)
        # Si el modelo ya está exportado, se deja el archivo como está:
        # así el repo no ve un cambio donde no lo hay
        if DESTINO.exists() and es_el_mismo_modelo(
                pipeline, joblib.load(DESTINO)):
            print(f"Versión {detalle}: es el mismo modelo que ya está "
                  f"en {DESTINO}, el archivo no cambia")
        else:
            joblib.dump(pipeline, DESTINO)
            print(f"Versión {detalle} exportada a {DESTINO}")
    escribir_metadatos(cliente, version)
    print(f"Metadatos de la versión {detalle} en {METADATOS}")


if __name__ == "__main__":
    main()
