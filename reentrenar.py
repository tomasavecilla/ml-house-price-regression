"""Reentrena el modelo con los lotes nuevos y lo compara con producción.

El challenger (modelo nuevo) usa la misma configuración que el
champion (docker/modelo.joblib), pero se entrena con el dataset
original más todos los lotes de datos/lotes/. Los dos se evalúan sobre
el mismo test, que incluye casas de los lotes. Solo si el challenger
se equivoca al menos un 1% menos reemplaza al champion; igual hace
falta que una persona apruebe el cambio (el PR que abre el workflow).

Uso:
    python reentrenar.py
    python reentrenar.py --resumen resumen.md
"""

import argparse
import os
from pathlib import Path

import joblib
import mlflow
from mlflow.models import infer_signature
from sklearn.base import clone

import entrenar

MODELO = "docker/modelo.joblib"
# El challenger tiene que equivocarse al menos un 1% menos: una mejora
# más chica puede ser ruido, y cambiar de modelo tiene un costo
MEJORA_MINIMA = 0.01


def main() -> None:
    """Entrena el challenger y, si es mejor, reemplaza al champion."""
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

    rmse_champion = entrenar.evaluar(champion, X_test, y_test)["rmse_test"]
    metricas = entrenar.evaluar(challenger, X_test, y_test)
    rmse_challenger = metricas["rmse_test"]
    mejoro = rmse_challenger < rmse_champion * (1 - MEJORA_MINIMA)

    mlflow.set_tracking_uri(entrenar.TRACKING_URI)
    mlflow.set_experiment(entrenar.NOMBRE_MODELO)
    # En Actions el usuario del sistema es "runner": se deja claro que
    # este modelo lo entrenó el proceso automático
    with mlflow.start_run(run_name="reentrenamiento",
                          tags={"mlflow.user": "reentrenamiento-automatico"}):
        mlflow.log_params({"lotes": ", ".join(lotes),
                           "filas_train": len(X_train)})
        mlflow.log_metrics({**metricas, "rmse_champion": rmse_champion})
        mlflow.sklearn.log_model(
            challenger,
            name="modelo",
            registered_model_name=entrenar.NOMBRE_MODELO,
            signature=infer_signature(X_train, challenger.predict(X_train)),
            skops_trusted_types=["numpy.dtype"],
        )

    resumen = "\n".join([
        f"Lotes usados: {', '.join(Path(lote).name for lote in lotes)}",
        f"Casas en train: {len(X_train)}, en test: {len(X_test)}",
        "",
        "| Modelo | RMSE en test |",
        "| --- | --- |",
        f"| Champion (producción) | {rmse_champion:.2f} |",
        f"| Challenger (reentrenado) | {rmse_challenger:.2f} |",
        "",
        "El challenger es mejor: reemplaza a docker/modelo.joblib."
        if mejoro else
        "El challenger no mejora al menos un "
        f"{MEJORA_MINIMA:.0%}: producción no cambia.",
    ])
    print(resumen)
    if args.resumen:
        Path(args.resumen).write_text(resumen)

    if mejoro:
        joblib.dump(challenger, MODELO)
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as salida:
            salida.write(f"mejoro={str(mejoro).lower()}\n")


if __name__ == "__main__":
    main()
