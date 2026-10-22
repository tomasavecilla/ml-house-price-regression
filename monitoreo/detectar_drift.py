"""Compara lotes de casas nuevas contra los datos de entrenamiento.

Mira dos cosas:

- Data drift: si las características cambiaron de distribución. Lo
  mide Evidently, variable por variable (test de Kolmogorov-Smirnov en
  las continuas, Z-test en CHAS). No necesita precios.
- Degradación: si el lote trae precios reales (llegan cuando las casas
  se venden), si el modelo de producción se equivoca más que en test.

El reporte de Evidently y las métricas quedan en el experimento
"monitoreo" de MLflow, en la misma UI de siempre.

Uso (desde la raíz del repo):
    python -m monitoreo.detectar_drift datos/lotes/zona-nueva.csv
    python -m monitoreo.detectar_drift datos/lotes/*.csv --resumen r.md
"""

import argparse
import os
from pathlib import Path

import joblib
import mlflow
from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd
from sklearn.pipeline import Pipeline

from entrenamiento import entrenar

MODELO = entrenar.MODELO
REPORTES = entrenar.RAIZ / "monitoreo" / "reportes"
# Una variable sola puede salir distinta por azar; si cambian 3 de las
# 12 (25%) ya no es casualidad
PROPORCION_DRIFT = 0.25
# Se tolera hasta un 20% más de error que en test antes de alertar
TOLERANCIA_RMSE = 1.2


def medir_drift(referencia: pd.DataFrame, lote: pd.DataFrame,
                destino_html: Path) -> tuple[list[str], float]:
    """Corre Evidently y guarda el reporte HTML.

    Args:
        referencia: características con las que se entrenó el modelo.
        lote: características de las casas nuevas, mismas columnas.
        destino_html: dónde guardar el reporte.

    Returns:
        Las columnas con drift y la proporción de columnas con drift.
    """
    reporte = Report([DataDriftPreset(drift_share=PROPORCION_DRIFT)])
    resultado = reporte.run(current_data=lote, reference_data=referencia)
    resultado.save_html(str(destino_html))

    con_drift = []
    proporcion = 0.0
    for metrica in resultado.dict()["metrics"]:
        configuracion = metrica["config"]
        if configuracion["type"].endswith("DriftedColumnsCount"):
            proporcion = metrica["value"]["share"]
        elif configuracion["type"].endswith("ValueDrift"):
            # El valor es el p-valor del test: por debajo del umbral, la
            # distribución del lote es distinta a la de referencia
            if metrica["value"] < configuracion["threshold"]:
                con_drift.append(configuracion["column"])
    return con_drift, proporcion


def medir_rmse(pipeline: Pipeline, X: pd.DataFrame,
               y: pd.Series) -> float:
    """Calcula el RMSE del modelo de producción.

    Args:
        pipeline: el modelo de producción.
        X: características de las casas.
        y: precios reales.

    Returns:
        El RMSE, en miles de dólares.
    """
    return entrenar.evaluar(pipeline, X, y)["rmse_test"]


def revisar_lote(ruta: str, pipeline: Pipeline, referencia: pd.DataFrame,
                 rmse_referencia: float) -> dict:
    """Mide drift y degradación de un lote y lo registra en MLflow.

    Args:
        ruta: CSV del lote.
        pipeline: el modelo de producción.
        referencia: características de train.
        rmse_referencia: RMSE del modelo en el test original.

    Returns:
        Un diccionario con el resultado del lote: nombre, columnas con
        drift, hay_drift, rmse_lote (None si no trae precios) y
        hay_degradacion.
    """
    lote = pd.read_csv(ruta)
    columnas = list(pipeline.feature_names_in_)
    nombre = Path(ruta).stem
    html = REPORTES / f"{nombre}.html"
    con_drift, proporcion = medir_drift(referencia[columnas],
                                        lote[columnas], html)

    rmse_lote = None
    if entrenar.OBJETIVO in lote and lote[entrenar.OBJETIVO].notna().any():
        con_precio = lote.dropna(subset=[entrenar.OBJETIVO])
        rmse_lote = medir_rmse(pipeline, con_precio[columnas],
                               con_precio[entrenar.OBJETIVO])
    resultado = {
        "lote": nombre,
        "columnas": con_drift,
        "hay_drift": proporcion >= PROPORCION_DRIFT,
        "rmse_lote": rmse_lote,
        "hay_degradacion": (rmse_lote is not None
                            and rmse_lote > rmse_referencia
                            * TOLERANCIA_RMSE),
    }

    with mlflow.start_run(run_name=nombre):
        mlflow.log_params({"lote": ruta, "filas": len(lote)})
        mlflow.log_metrics({"proporcion_columnas_drift": proporcion,
                            "columnas_drift": len(con_drift),
                            "rmse_referencia": rmse_referencia})
        if rmse_lote is not None:
            mlflow.log_metric("rmse_lote", rmse_lote)
        mlflow.set_tags({"hay_drift": resultado["hay_drift"],
                         "hay_degradacion": resultado["hay_degradacion"]})
        mlflow.log_artifact(str(html))
    return resultado


def resumir(resultados: list[dict], rmse_referencia: float) -> str:
    """Arma un resumen en Markdown, para la consola y para el issue.

    Args:
        resultados: lo que devolvió revisar_lote para cada lote.
        rmse_referencia: RMSE del modelo en el test original.

    Returns:
        El resumen en Markdown.
    """
    lineas = [f"RMSE del modelo en el test original: "
              f"{rmse_referencia:.2f}", ""]
    for r in resultados:
        drift = ", ".join(r["columnas"]) or "ninguna"
        lineas.append(f"### Lote `{r['lote']}`")
        lineas.append(f"- Variables con drift: {drift}")
        lineas.append(f"- Data drift: {'sí' if r['hay_drift'] else 'no'}")
        if r["rmse_lote"] is None:
            lineas.append("- Sin precios reales: no se puede medir el "
                          "error")
        else:
            lineas.append(f"- RMSE en el lote: {r['rmse_lote']:.2f} "
                          f"(degradación: "
                          f"{'sí' if r['hay_degradacion'] else 'no'})")
        lineas.append("")
    return "\n".join(lineas)


def main() -> None:
    """Revisa los lotes, imprime el resumen y avisa al workflow."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("lotes", nargs="+", help="CSV de casas nuevas")
    parser.add_argument("--resumen", help="dónde guardar el resumen .md")
    args = parser.parse_args()

    pipeline = joblib.load(MODELO)
    X_train, X_test, _, y_test = entrenar.cargar_datos()
    rmse_referencia = medir_rmse(pipeline, X_test, y_test)
    REPORTES.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(entrenar.TRACKING_URI)
    mlflow.set_experiment("monitoreo")
    resultados = [revisar_lote(ruta, pipeline, X_train, rmse_referencia)
                  for ruta in args.lotes]

    resumen = resumir(resultados, rmse_referencia)
    print(resumen)
    if args.resumen:
        Path(args.resumen).write_text(resumen)

    alerta = any(r["hay_drift"] or r["hay_degradacion"]
                 for r in resultados)
    # En GitHub Actions, GITHUB_OUTPUT es un archivo donde el paso deja
    # valores que los pasos siguientes pueden leer
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as salida:
            salida.write(f"alerta={str(alerta).lower()}\n")


if __name__ == "__main__":
    main()
