"""Entrena el pipeline de precio de casas y lo registra en MLflow.

Cada ejecución queda como un run y como una versión nueva del modelo
en el registro.

Uso:
    python entrenar.py                         # Ridge, imputador KNN
    python entrenar.py --modelo lasso --alpha 0.1
    python entrenar.py --imputador mediana
    python entrenar.py --lotes datos/lotes/*.csv   # suma casas nuevas
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import os
from pathlib import Path
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# El dataset original y la carpeta donde llegan las casas nuevas
DATOS = "house-prices-tp.csv"
LOTES = "datos/lotes"
OBJETIVO = "MEDV"
# B se descarta: se calcula a partir de la proporción de población
# negra de cada ciudad
DESCARTADAS = ["B"]
# Variables que solo toman ciertos valores: se imputan con la moda
DISCRETAS = ["CHAS", "RAD"]
NOMBRE_MODELO = "precio-casas"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def separar(datos: pd.DataFrame
            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa un conjunto de casas en train y test.

    Descarta las filas sin precio y usa siempre la misma semilla, así
    todas las versiones del modelo se evalúan sobre las mismas casas.

    Args:
        datos: casas con sus características y la columna MEDV.

    Returns:
        X_train, X_test, y_train, y_test, en ese orden.
    """
    # Sin precio no hay nada que aprender: esas filas no sirven para
    # entrenar ni para evaluar
    datos = datos.dropna(subset=[OBJETIVO])
    X = datos.drop(columns=[OBJETIVO])
    y = datos[OBJETIVO]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def lotes_disponibles() -> list[str]:
    """Lista los lotes de casas nuevas que llegaron a LOTES.

    Returns:
        Las rutas de los CSV, ordenadas por nombre.
    """
    return sorted(str(p) for p in Path(LOTES).glob("*.csv"))


def cargar_datos(lotes: list[str] | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame,
                            pd.Series, pd.Series]:
    """Carga el dataset y, si se piden, los lotes nuevos.

    El dataset original y cada lote se separan por su cuenta y después
    se juntan. Así el test también tiene casas de los lotes, y un
    modelo se evalúa sobre los datos que encuentra hoy en producción.

    Args:
        lotes: rutas a CSV con casas nuevas (mismas columnas que el
            dataset, sin B). None o vacío: solo el dataset original.

    Returns:
        X_train, X_test, y_train, y_test, en ese orden. Las X tienen
        una columna por característica; las y tienen el precio (MEDV)
        en miles de dólares.
    """
    conjuntos = [pd.read_csv(DATOS).drop(columns=DESCARTADAS)]
    conjuntos += [pd.read_csv(lote) for lote in lotes or []]
    partes = [separar(datos) for datos in conjuntos]
    # partes es una lista de (X_train, X_test, y_train, y_test): se
    # concatena cada una de las cuatro posiciones por separado
    return tuple(pd.concat(parte) for parte in zip(*partes))


def construir_modelo(modelo: Literal["lineal", "lasso", "ridge",
                                     "elasticnet"],
                     alpha: float) -> RegressorMixin:
    """Crea el regresor sin entrenar.

    Args:
        modelo: uno de los modelos que compara el TP.
        alpha: fuerza de la regularización. No se usa en "lineal".

    Returns:
        Un regresor de scikit-learn.
    """
    if modelo == "lineal":
        return LinearRegression()
    if modelo == "lasso":
        return Lasso(alpha=alpha)
    if modelo == "elasticnet":
        return ElasticNet(alpha=alpha, l1_ratio=0.5)
    return Ridge(alpha=alpha)


def construir_pipeline(imputador: Literal["knn", "mediana"],
                       regresor: RegressorMixin,
                       columnas: list[str]) -> Pipeline:
    """Arma el pipeline de preprocesamiento y modelo, sin entrenar.

    Las variables continuas se escalan y después se imputan. Las
    discretas (CHAS y RAD) se imputan con el valor más frecuente, para
    que el valor completado sea uno que existe, y después se escalan.

    Args:
        imputador: para las continuas. "knn" usa los 5 vecinos más
            cercanos; "mediana" completa con la mediana de cada columna.
        regresor: el modelo que va al final del pipeline.
        columnas: nombres de las características de entrada.

    Returns:
        Un Pipeline de scikit-learn listo para hacer fit.
    """
    continuas = [c for c in columnas if c not in DISCRETAS]
    if imputador == "knn":
        imputer = KNNImputer(n_neighbors=5)
    else:
        imputer = SimpleImputer(strategy="median")
    continuas_pipeline = Pipeline([
        # Se escala antes de imputar para que KNN mida distancias con
        # todas las variables en la misma escala
        ("escalar", StandardScaler()),
        ("imputar", imputer),
    ])
    discretas_pipeline = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("escalar", StandardScaler()),
    ])
    preprocesamiento = ColumnTransformer([
        ("continuas", continuas_pipeline, continuas),
        ("discretas", discretas_pipeline, DISCRETAS),
    ])
    return Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("modelo", regresor),
    ])


def evaluar(pipeline: Pipeline, X: pd.DataFrame,
            y: pd.Series) -> dict[str, float]:
    """Calcula las métricas del pipeline sobre un conjunto de datos.

    Args:
        pipeline: pipeline ya entrenado.
        X: características de las casas.
        y: precio real de esas casas, en miles de dólares.

    Returns:
        Diccionario con rmse_test, mae_test y r2_test.
    """
    prediccion = pipeline.predict(X)
    return {
        "rmse_test": float(np.sqrt(mean_squared_error(y, prediccion))),
        "mae_test": float(mean_absolute_error(y, prediccion)),
        "r2_test": float(r2_score(y, prediccion)),
    }


def main() -> None:
    """Entrena, evalúa y registra una versión del modelo en MLflow."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--modelo", default="ridge",
                        choices=["lineal", "lasso", "ridge", "elasticnet"])
    parser.add_argument("--imputador", choices=["knn", "mediana"],
                        default="knn")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lotes", nargs="*", default=[],
                        help="CSV con casas nuevas para sumar")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = cargar_datos(args.lotes)
    regresor = construir_modelo(args.modelo, args.alpha)
    pipeline = construir_pipeline(args.imputador, regresor,
                                  list(X_train.columns))

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(NOMBRE_MODELO)
    with mlflow.start_run():
        mlflow.log_params({
            "modelo": args.modelo,
            "imputador": args.imputador,
            # La regresión lineal no tiene alpha: se registra vacío
            # para no mostrar un valor que no se usó
            "alpha": "" if args.modelo == "lineal" else args.alpha,
            "filas_train": len(X_train),
            "lotes": ", ".join(args.lotes),
        })
        pipeline.fit(X_train, y_train)
        metricas = evaluar(pipeline, X_test, y_test)
        mlflow.log_metrics(metricas)
        info = mlflow.sklearn.log_model(
            pipeline,
            name="modelo",
            registered_model_name=NOMBRE_MODELO,
            signature=infer_signature(X_train, pipeline.predict(X_train)),
            input_example=X_train.head(3),
            # MLflow guarda con skops, un formato que no ejecuta código
            # al cargar; hay que declarar qué tipos puede reconstruir
            skops_trusted_types=["numpy.dtype"],
        )

    print(f"RMSE test: {metricas['rmse_test']:.2f}  |  "
          f"versión registrada: {info.registered_model_version}")


if __name__ == "__main__":
    main()
