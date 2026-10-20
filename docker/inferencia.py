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

MODELO = Path(__file__).parent / "modelo.joblib"


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
    # El pipeline completo: imputación, escalado y modelo, tal como
    # quedaron al entrenar
    pipeline = joblib.load(MODELO)
    # El pipeline recuerda con qué columnas se entrenó
    columnas = list(pipeline.feature_names_in_)

    casas = pd.read_csv(entrada)
    faltantes = [c for c in columnas if c not in casas.columns]
    if faltantes:
        sys.exit(f"Error: a {entrada} le faltan las columnas "
                 f"{', '.join(faltantes)}")

    # predict pasa las casas por todo el pipeline: escala con la media
    # y el desvío de train, imputa con lo aprendido en train y recién
    # después aplica el modelo. Nunca fit: eso ya pasó al entrenar, y lo
    # aprendido viaja en modelo.joblib.
    # casas[columnas] además ordena las columnas y deja afuera las que
    # sobran.
    casas["precio_estimado"] = pipeline.predict(casas[columnas]).round(1)
    casas.to_csv(salida, index=False)
    print(f"{len(casas)} casas tasadas en {salida}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Uso: python inferencia.py <casas.csv> <precios.csv>")
    main(sys.argv[1], sys.argv[2])
