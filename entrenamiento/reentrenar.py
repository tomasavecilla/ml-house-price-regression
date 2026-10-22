"""Reentrena el modelo con los lotes nuevos y lo compara con producción.

El challenger (modelo nuevo) usa la misma configuración que el
champion (docker/modelo.joblib), pero se entrena con el dataset
original más todos los lotes de datos/lotes/. Los dos se evalúan sobre
el mismo test, que incluye casas de los lotes.

El challenger se registra siempre como versión nueva de precio-casas en
MLflow, con las métricas de los dos y el reporte de Evidently, pero no
se promueve: el alias @champion no cambia. Una persona decide mirando
la comparación y, si corresponde, corre el workflow Promover. El
script recomienda promover solo si el challenger se equivoca al menos
un 1% menos.

Uso:
    python -m entrenamiento.reentrenar
    python -m entrenamiento.reentrenar --resumen resumen.md
"""

import argparse
import os
from pathlib import Path

import joblib
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.base import clone

from entrenamiento import entrenar
from monitoreo.detectar_drift import REPORTES

MODELO = entrenar.MODELO
# El challenger tiene que equivocarse al menos un 1% menos: una mejora
# más chica puede ser ruido, y cambiar de modelo tiene un costo
MEJORA_MINIMA = 0.01


def armar_comparacion(lotes: list[str], filas: tuple[int, int],
                      champion: dict[str, float],
                      challenger: dict[str, float], mejora: float,
                      version: str, version_champion: str | None,
                      enlace_version: str | None,
                      enlace_run: str | None) -> str:
    """Arma la comparación champion vs challenger, en Markdown.

    Es el texto que se comenta en el issue de alerta.

    Args:
        lotes: rutas de los lotes usados para entrenar.
        filas: cantidad de casas en train y en test.
        champion: rmse_test, mae_test y r2_test del champion.
        challenger: las mismas métricas, del challenger.
        mejora: cuánto menos se equivoca el challenger en RMSE, como
            fracción del RMSE del champion (0.03 es un 3% menos).
        version: número de versión del challenger en el registro.
        version_champion: número de versión del alias @champion, o None
            si no se pudo averiguar.
        enlace_version: link a la versión en MLflow, si hay.
        enlace_run: link al run en MLflow, si hay.

    Returns:
        El resumen en Markdown.
    """
    etiqueta_champion = "Champion (producción"
    if version_champion:
        etiqueta_champion += f", versión {version_champion}"
    etiqueta_champion += ")"
    lineas = [
        "## Reentrenamiento con los lotes nuevos",
        "",
        f"Lotes usados: {', '.join(Path(lote).name for lote in lotes)}",
        f"Casas en train: {filas[0]}, en test: {filas[1]}",
        "",
        "| Modelo | RMSE | MAE | R² |",
        "| --- | --- | --- | --- |",
        f"| {etiqueta_champion} | {champion['rmse_test']:.2f} "
        f"| {champion['mae_test']:.2f} | {champion['r2_test']:.3f} |",
        f"| Challenger (versión {version}) "
        f"| {challenger['rmse_test']:.2f} | {challenger['mae_test']:.2f} "
        f"| {challenger['r2_test']:.3f} |",
        "",
        "Los dos se evaluaron sobre el mismo test. El RMSE del "
        f"challenger es un {abs(mejora):.1%} "
        f"{'menor' if mejora >= 0 else 'mayor'}.",
        "",
    ]
    if mejora >= MEJORA_MINIMA:
        lineas.append(f"**Recomendación: promover.** Mejora al menos "
                      f"un {MEJORA_MINIMA:.0%}.")
    else:
        lineas.append(f"**Recomendación: no promover.** No mejora al "
                      f"menos un {MEJORA_MINIMA:.0%}; la versión queda "
                      f"registrada por si se quiere revisar.")
    lineas.append("")
    if enlace_version:
        lineas.append(f"- Versión {version} en MLflow: {enlace_version}")
    if enlace_run:
        lineas.append(f"- Run con el reporte de Evidently: {enlace_run}")
    condicion = ("Para promoverla" if mejora >= MEJORA_MINIMA
                 else "Si igual se quiere promover")
    lineas += [
        f"- {condicion}, correr el workflow Promover con la versión "
        f"{version} (pestaña Actions, o `gh workflow run promover.yml "
        f"-f version={version}`).",
        "- El alias `@champion` no cambió: producción sigue igual.",
    ]
    return "\n".join(lineas)


def main() -> None:
    """Entrena el challenger y lo registra como versión nueva."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--resumen", help="dónde guardar el resumen .md")
    args = parser.parse_args()

    lotes = entrenar.lotes_disponibles()
    X_train, X_test, y_train, y_test = entrenar.cargar_datos(lotes)
    champion = joblib.load(MODELO)
    # clone copia la configuración (pasos e hiperparámetros) pero no lo
    # aprendido: es el mismo modelo, listo para entrenar de cero
    challenger = clone(champion).fit(X_train, y_train)

    metricas_champion = entrenar.evaluar(champion, X_test, y_test)
    metricas = entrenar.evaluar(challenger, X_test, y_test)
    rmse_champion = metricas_champion["rmse_test"]
    mejora = 1 - metricas["rmse_test"] / rmse_champion

    mlflow.set_tracking_uri(entrenar.TRACKING_URI)
    cliente = mlflow.MlflowClient()
    try:
        alias = cliente.get_model_version_by_alias(entrenar.NOMBRE_MODELO,
                                                   "champion")
        version_champion = str(alias.version)
    except MlflowException:
        # Un registro sin alias (por ejemplo, uno local recién creado)
        version_champion = None

    mlflow.set_experiment(entrenar.NOMBRE_MODELO)
    # En Actions el usuario del sistema es "runner": se deja claro que
    # este modelo lo entrenó el proceso automático
    with mlflow.start_run(run_name="reentrenamiento",
                          tags={"mlflow.user": "reentrenamiento-automatico"}
                          ) as run:
        mlflow.log_params({"lotes": ", ".join(lotes),
                           "filas_train": len(X_train)})
        mlflow.log_metrics({
            **metricas,
            "rmse_champion": rmse_champion,
            "mae_champion": metricas_champion["mae_test"],
            "r2_champion": metricas_champion["r2_test"],
            "mejora_rmse": mejora,
        })
        # El reporte de Evidently de cada lote, para ver por qué se
        # reentrenó sin salir de MLflow
        for lote in lotes:
            reporte = REPORTES / f"{Path(lote).stem}.html"
            if reporte.exists():
                mlflow.log_artifact(str(reporte), "reportes")
        info = mlflow.sklearn.log_model(
            challenger,
            name="modelo",
            registered_model_name=entrenar.NOMBRE_MODELO,
            signature=infer_signature(X_train, challenger.predict(X_train)),
            skops_trusted_types=["numpy.dtype"],
        )
    version = str(info.registered_model_version)

    # Las etiquetas de la versión cuentan de dónde salió y cómo le fue,
    # para decidir desde el registro sin abrir el run
    etiquetas = {
        "origen": "reentrenamiento",
        "lotes": ", ".join(Path(lote).name for lote in lotes),
        "rmse": f"{metricas['rmse_test']:.2f}",
        "rmse_champion": f"{rmse_champion:.2f}",
        "mejora": f"{mejora:.1%}",
        "recomendacion": ("promover" if mejora >= MEJORA_MINIMA
                          else "no promover"),
    }
    if version_champion:
        etiquetas["champion"] = version_champion
    if "GITHUB_RUN_ID" in os.environ:
        etiquetas["workflow"] = (
            f"{os.environ['GITHUB_SERVER_URL']}/"
            f"{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
            f"{os.environ['GITHUB_RUN_ID']}")
    for clave, valor in etiquetas.items():
        cliente.set_model_version_tag(entrenar.NOMBRE_MODELO, version,
                                      clave, valor)

    resumen = armar_comparacion(
        lotes, (len(X_train), len(X_test)), metricas_champion, metricas,
        mejora, version, version_champion,
        entrenar.enlace_mlflow(
            f"models/{entrenar.NOMBRE_MODELO}/versions/{version}"),
        entrenar.enlace_mlflow(
            f"experiments/{run.info.experiment_id}/runs/{run.info.run_id}"))
    print(resumen)
    if args.resumen:
        Path(args.resumen).write_text(resumen)

    # En GitHub Actions, GITHUB_OUTPUT es un archivo donde el paso deja
    # valores que los pasos siguientes pueden leer
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as salida:
            salida.write(f"version={version}\n")


if __name__ == "__main__":
    main()
