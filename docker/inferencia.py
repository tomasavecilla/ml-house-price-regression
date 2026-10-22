"""Estima el precio de un conjunto de casas.

Lee un CSV con las características de las casas y escribe otro con una
columna más, precio_estimado, en miles de dólares (la unidad de MEDV).

Uso:
    python inferencia.py casas.csv precios.csv
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

MODELO = Path(__file__).parent / "modelo.joblib"


def cargar_modelo(ruta: Path = MODELO) -> Pipeline:
    """Carga el pipeline completo tal como quedó al entrenar.

    Args:
        ruta: dónde está el binario. Por defecto, el modelo.joblib que
            está al lado de este script.

    Returns:
        El pipeline: imputación, escalado y modelo.
    """
    return joblib.load(ruta)


def predecir(pipeline: Pipeline, casas: pd.DataFrame) -> pd.Series:
    """Estima el precio de cada casa, en miles de dólares.

    Es el único lugar donde se predice: el script por lotes y la API
    llaman a esta función, así que dan siempre el mismo precio.

    Args:
        pipeline: el pipeline que devolvió cargar_modelo.
        casas: una fila por casa. Tiene que tener las columnas con las
            que se entrenó el modelo; las que sobran se ignoran.

    Returns:
        El precio de cada casa, redondeado a un decimal y con el mismo
        índice que casas.

    Raises:
        ValueError: si a casas le falta alguna columna.
    """
    # El pipeline recuerda con qué columnas se entrenó
    columnas = list(pipeline.feature_names_in_)
    faltantes = [c for c in columnas if c not in casas.columns]
    if faltantes:
        raise ValueError(
            f"le faltan las columnas {', '.join(faltantes)}")

    # predict pasa las casas por todo el pipeline: escala con la media
    # y el desvío de train, imputa con lo aprendido en train y recién
    # después aplica el modelo. Nunca fit: eso ya pasó al entrenar, y lo
    # aprendido viaja en modelo.joblib.
    # casas[columnas] además ordena las columnas y deja afuera las que
    # sobran.
    precios = pipeline.predict(casas[columnas])
    return pd.Series(precios, index=casas.index).round(1)


def main(entrada: str, salida: str) -> None:
    """Predice el precio de cada casa de entrada y lo guarda en salida.

    Args:
        entrada: ruta al CSV con las casas. Tiene que tener las columnas
            con las que se entrenó el modelo; las que sobran se ignoran
            y se copian tal cual a la salida.
        salida: ruta del CSV a escribir, con las columnas de entrada
            más precio_estimado.

    Raises:
        SystemExit: si a la entrada le falta alguna columna.
    """
    pipeline = cargar_modelo()
    casas = pd.read_csv(entrada)
    try:
        casas["precio_estimado"] = predecir(pipeline, casas)
    except ValueError as error:
        sys.exit(f"Error: a {entrada} {error}")
    casas.to_csv(salida, index=False)
    print(f"{len(casas)} casas tasadas en {salida}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Uso: python inferencia.py <casas.csv> <precios.csv>")
    main(sys.argv[1], sys.argv[2])
