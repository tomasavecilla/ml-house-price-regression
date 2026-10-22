"""Asigna el alias @champion a una versión y resume la promoción.

Es el último paso del workflow Promover: antes de llegar acá, la
versión ya se exportó a docker/modelo.joblib, pasó los tests y el
binario está commiteado en main. Recién entonces el registro de MLflow
apunta a esa versión, así el alias y el repo nunca quedan desacoplados.

Uso:
    python -m entrenamiento.promover --version 21 --asignar
    python -m entrenamiento.promover --version 21 --resumen resumen.md
"""

import argparse
from pathlib import Path

import mlflow

from entrenamiento import entrenar
from entrenamiento.exportar_modelo import version_valida

# Las métricas que se muestran, si el run las tiene
METRICAS = ["rmse_test", "mae_test", "r2_test", "rmse_cv"]


def resumir(cliente: mlflow.MlflowClient, version: int,
            estado: str) -> str:
    """Arma el resumen de una versión del registro, en Markdown.

    Args:
        cliente: cliente de MLflow ya conectado al registro.
        version: número de versión.
        estado: qué pasó con la promoción, en una frase.

    Returns:
        El resumen, con las métricas del run y las etiquetas de la
        versión.
    """
    modelo = cliente.get_model_version(entrenar.NOMBRE_MODELO,
                                       str(version))
    metricas = cliente.get_run(modelo.run_id).data.metrics
    lineas = [f"### Promoción de la versión {version}", "", estado, ""]
    if any(m in metricas for m in METRICAS):
        lineas += ["| Métrica | Valor |", "| --- | --- |"]
        lineas += [f"| {m} | {metricas[m]:.3f} |"
                   for m in METRICAS if m in metricas]
        lineas.append("")
    for clave, valor in sorted(modelo.tags.items()):
        lineas.append(f"- {clave}: {valor}")
    enlace = entrenar.enlace_mlflow(
        f"models/{entrenar.NOMBRE_MODELO}/versions/{version}")
    if enlace:
        lineas.append(f"- Versión en MLflow: {enlace}")
    return "\n".join(lineas)


def main() -> None:
    """Asigna el alias champion y/o escribe el resumen."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", type=version_valida, required=True)
    parser.add_argument("--asignar", action="store_true",
                        help="asigna el alias champion a la versión")
    parser.add_argument("--resumen", help="dónde guardar el resumen .md")
    parser.add_argument("--estado", default="",
                        help="frase para el resumen: qué pasó")
    args = parser.parse_args()

    mlflow.set_tracking_uri(entrenar.TRACKING_URI)
    cliente = mlflow.MlflowClient()
    if args.asignar:
        cliente.set_registered_model_alias(entrenar.NOMBRE_MODELO,
                                           "champion", str(args.version))
        print(f"Alias champion asignado a la versión {args.version}")
    if args.resumen:
        resumen = resumir(cliente, args.version, args.estado)
        Path(args.resumen).write_text(resumen)
        print(resumen)


if __name__ == "__main__":
    main()
