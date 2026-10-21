"""Muestra qué hay adentro de docker/modelo.joblib.

El archivo no guarda código: guarda los pasos del pipeline y lo que
cada uno aprendió en train.

Uso:
    python inspeccionar_modelo.py
"""

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

MODELO = "docker/modelo.joblib"


def mostrar_aprendido(pipeline: Pipeline) -> None:
    """Imprime los pasos del pipeline y los valores aprendidos en train.

    Args:
        pipeline: el pipeline cargado desde el .joblib.
    """
    preprocesamiento = pipeline.named_steps["preprocesamiento"]
    continuas = preprocesamiento.named_transformers_["continuas"]
    escalar = continuas.named_steps["escalar"]
    imputar = continuas.named_steps["imputar"]
    modelo = pipeline.named_steps["modelo"]

    print("Columnas de entrada:", ", ".join(pipeline.feature_names_in_))
    print("\nStandardScaler, media de cada variable continua:")
    for columna, media in zip(escalar.feature_names_in_, escalar.mean_):
        print(f"  {columna:8} {media:8.2f}")
    if hasattr(imputar, "_fit_X"):
        # KNN necesita las casas de train para buscar vecinos
        print(f"\nKNNImputer guarda {imputar._fit_X.shape[0]} casas de "
              "train para buscar vecinos")
    print(f"\n{type(modelo).__name__}, coeficientes:",
          np.round(modelo.coef_, 2))
    print(f"{type(modelo).__name__}, intercepto: {modelo.intercept_:.2f}")


if __name__ == "__main__":
    pipeline = joblib.load(MODELO)
    print(pipeline, "\n")
    mostrar_aprendido(pipeline)
