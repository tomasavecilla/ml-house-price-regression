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
    # El preprocesamiento reordena las columnas (primero las continuas,
    # después las discretas): los nombres hay que pedírselos a él, no
    # tomarlos de la entrada. Vienen como "continuas__RM"
    nombres = [n.split("__")[-1]
               for n in preprocesamiento.get_feature_names_out()]
    print(f"\n{type(modelo).__name__}, coeficientes ordenados por peso.")
    print("Como las variables están escaladas, cada uno dice cuántos miles"
          "\nde USD cambia el precio si la variable sube un desvío:")
    for indice in np.argsort(-np.abs(modelo.coef_)):
        print(f"  {nombres[indice]:8} {modelo.coef_[indice]:+6.2f}")
    print(f"Intercepto (precio de una casa promedio): "
          f"{modelo.intercept_:.2f}")


if __name__ == "__main__":
    pipeline = joblib.load(MODELO)
    print(pipeline, "\n")
    mostrar_aprendido(pipeline)
