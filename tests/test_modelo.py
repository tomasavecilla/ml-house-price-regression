"""Tests del binario exportado (docker/modelo.joblib).

No prueban código sino el artefacto: que el modelo que va a producción
sigue siendo bueno y se puede cargar en cualquier lado.
"""

from pathlib import Path

import joblib
import pytest
from sklearn.pipeline import Pipeline

from entrenamiento import entrenar

MODELO = Path(__file__).resolve().parent.parent / "docker" / "modelo.joblib"
# El champion actual (ElasticNet, alpha 0.1) tiene RMSE 4.9 en test. Si
# alguien exporta un modelo peor que esto, el CI lo frena antes de que
# llegue a un contenedor
RMSE_MAXIMO = 5.5


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    """Carga el pipeline exportado una vez para todo el módulo."""
    return joblib.load(MODELO)


def test_rmse_no_empeoro(pipeline: Pipeline) -> None:
    """El RMSE sobre el mismo split de test del entrenamiento está bajo
    el umbral."""
    _, X_test, _, y_test = entrenar.cargar_datos()
    metricas = entrenar.evaluar(pipeline, X_test, y_test)
    assert metricas["rmse_test"] < RMSE_MAXIMO


def test_solo_usa_clases_de_sklearn(pipeline: Pipeline) -> None:
    """Ningún paso depende de código del proyecto.

    Si el pipeline tuviera una clase propia, joblib.load fallaría en el
    contenedor, donde solo está instalado scikit-learn.
    """
    # get_params(deep=True) recorre también lo que hay adentro del
    # ColumnTransformer y de sus sub-pipelines
    pasos = [p for p in pipeline.get_params(deep=True).values()
             if hasattr(p, "fit")]
    assert pasos
    for paso in pasos:
        assert type(paso).__module__.startswith("sklearn"), paso
