"""Entrena el pipeline de precio de casas y lo registra en MLflow.

Cada ejecución queda como un run y como una versión nueva del modelo
en el registro.

Uso:
    python -m entrenamiento.entrenar              # Ridge, imputador KNN
    python -m entrenamiento.entrenar --modelo lasso --alpha 0.1
    python -m entrenamiento.entrenar --imputador mediana
    python -m entrenamiento.entrenar --lotes datos/lotes/*.csv
                                                  # suma casas nuevas
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import getpass
import os
import warnings
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
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# El dataset original y la carpeta donde llegan las casas nuevas
# Las rutas se arman desde la raíz del repo, así los scripts andan
# desde cualquier directorio
RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "datos" / "house-prices-tp.csv"
LOTES = RAIZ / "datos" / "lotes"
MODELO = RAIZ / "docker" / "modelo.joblib"
OBJETIVO = "MEDV"
# B se descarta: se calcula a partir de la proporción de población
# negra de cada ciudad
DESCARTADAS = ["B"]
# Variables que solo toman ciertos valores: se imputan con la moda
DISCRETAS = ["CHAS", "RAD"]
NOMBRE_MODELO = "precio-casas"
TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}")
# MLflow 3.17 registra dos veces el mismo tipo de origen de datos y
# avisa en cada log_input que el origen "se puede interpretar de dos
# formas". Las dos son la misma, así que el aviso no aporta nada
warnings.filterwarnings(
    "ignore", message="The specified dataset source can be interpreted")


def enlace_mlflow(ruta: str) -> str | None:
    """Arma el link a una página de la UI del MLflow publicado.

    Args:
        ruta: lo que va después de "#/", por ejemplo
            "models/precio-casas/versions/21".

    Returns:
        El link, o None si MLflow es local (un archivo sqlite no tiene
        UI a la que apuntar).
    """
    if not TRACKING_URI.startswith("http"):
        return None
    return f"{TRACKING_URI.rstrip('/')}/#/{ruta}"


def leer_dataset() -> pd.DataFrame:
    """Lee el dataset original y descarta las filas que no son válidas.

    El dataset del TP agrega a Boston Housing filas con ruido, con
    valores imposibles como RAD = 10.55 (un índice entero). Se las
    reconoce con una regla de validez, no por cómo afectan al modelo:
    Boston registra cada valor con 5 decimales como máximo, y esas
    filas tienen más.

    Returns:
        Las casas válidas, sin las columnas de DESCARTADAS.
    """
    datos = pd.read_csv(DATOS)
    con_ruido = ((datos.round(5) != datos) & datos.notna()).any(axis=1)
    return datos[~con_ruido].drop(columns=DESCARTADAS)


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
    conjuntos = [leer_dataset()]
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


def validar_cruzado(pipeline: Pipeline, X: pd.DataFrame,
                    y: pd.Series) -> dict[str, float]:
    """Estima las métricas con validación cruzada de 5 particiones.

    Es la métrica para elegir hiperparámetros: usa solo train, así el
    test queda sin tocar hasta evaluar el modelo elegido. En cada
    partición se reentrena el pipeline entero, imputación y escalado
    incluidos, para que nada aprendido se filtre a la validación.

    Args:
        pipeline: pipeline sin entrenar.
        X: características de train.
        y: precios de train, en miles de dólares.

    Returns:
        Diccionario con rmse_cv, mae_cv y r2_cv (promedio de las 5
        particiones) y rmse_cv_desvio, cuánto varía el RMSE entre ellas.
    """
    particiones = KFold(n_splits=5, shuffle=True, random_state=42)
    # scikit-learn maximiza los scores: los errores vienen con signo
    # negativo y se les cambia el signo
    resultado = cross_validate(pipeline, X, y, cv=particiones, scoring={
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    })
    return {
        "rmse_cv": float(-resultado["test_rmse"].mean()),
        "rmse_cv_desvio": float(resultado["test_rmse"].std()),
        "mae_cv": float(-resultado["test_mae"].mean()),
        "r2_cv": float(resultado["test_r2"].mean()),
    }


def entrenar_y_registrar(modelo: Literal["lineal", "lasso", "ridge",
                                         "elasticnet"],
                         imputador: Literal["knn", "mediana"],
                         alpha: float,
                         lotes: list[str] | None = None,
                         autor: str | None = None) -> dict:
    """Entrena una configuración y la registra como versión nueva.

    Args:
        modelo: el regresor, uno de los que compara el TP.
        imputador: cómo se completan las variables continuas.
        alpha: fuerza de la regularización. No se usa en "lineal".
        lotes: CSV con casas nuevas para sumar a los datos.
        autor: quién entrena. Por defecto, el usuario del sistema.

    Returns:
        La versión registrada y las métricas de validación cruzada y de
        test.
    """
    lotes = lotes or []
    X_train, X_test, y_train, y_test = cargar_datos(lotes)
    pipeline = construir_pipeline(imputador, construir_modelo(modelo, alpha),
                                  list(X_train.columns))

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(NOMBRE_MODELO)
    configuracion = {
        "modelo": modelo,
        "imputador": imputador,
        # La regresión lineal no tiene alpha: se registra vacío para no
        # mostrar un valor que no se usó
        "alpha": "" if modelo == "lineal" else alpha,
    }
    etiquetas_run = {
        # MLflow local no tiene login: el autor es una etiqueta. Con un
        # servidor compartido saldría del usuario autenticado
        "mlflow.user": autor or getpass.getuser(),
        # Como tags, sirven para filtrar en la UI: tags.modelo = 'ridge'
        "modelo": modelo,
        "imputador": imputador,
    }
    with mlflow.start_run(tags=etiquetas_run):
        mlflow.log_params({**configuracion, "filas_train": len(X_train),
                           "lotes": ", ".join(lotes)})
        # Qué datos se usaron: MLflow guarda una huella (digest) de cada
        # conjunto, y si los datos cambian, cambia la huella
        for contexto, X, y in [("train", X_train, y_train),
                               ("test", X_test, y_test)]:
            conjunto = mlflow.data.from_pandas(
                X.assign(**{OBJETIVO: y}), source=str(DATOS),
                name=f"casas-{contexto}", targets=OBJETIVO)
            mlflow.log_input(conjunto, context=contexto)
        # cv para comparar configuraciones; test para el modelo elegido
        metricas_cv = validar_cruzado(pipeline, X_train, y_train)
        pipeline.fit(X_train, y_train)
        metricas = evaluar(pipeline, X_test, y_test)
        mlflow.log_metrics({**metricas_cv, **metricas})
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
        version = info.registered_model_version
        cliente = mlflow.MlflowClient()
        # La configuración también queda en la versión, para verla desde
        # el registro sin entrar al run
        etiquetas = {**configuracion,
                     "rmse_cv": f"{metricas_cv['rmse_cv']:.2f}"}
        for clave, valor in etiquetas.items():
            cliente.set_model_version_tag(NOMBRE_MODELO, version, clave,
                                          str(valor))
    return {"version": version, **metricas_cv, **metricas}


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
    parser.add_argument("--autor", help="quién entrena (por defecto, el "
                        "usuario del sistema)")
    args = parser.parse_args()

    resultado = entrenar_y_registrar(args.modelo, args.imputador,
                                     args.alpha, args.lotes, args.autor)
    print(f"RMSE cv: {resultado['rmse_cv']:.2f}  |  "
          f"RMSE test: {resultado['rmse_test']:.2f}  |  "
          f"versión registrada: {resultado['version']}")


if __name__ == "__main__":
    main()
