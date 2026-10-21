"""Prueba todas las combinaciones de la grilla y registra cada una.

Cada combinación de modelo, imputador y alpha se entrena con
entrenar.py y queda como una versión de precio-casas. Al final se
ordenan por rmse_cv (validación cruzada sobre train): el test no se usa
para elegir, así sigue siendo una estimación honesta del modelo que se
elija.

También compara la mejor contra el modelo de producción
(docker/modelo.joblib) con las mismas métricas. El workflow de
entrenamiento usa esa comparación para comentarla en el PR.

Uso:
    python buscar_hiperparametros.py
    python buscar_hiperparametros.py --resumen resumen.md
    python buscar_hiperparametros.py --autor franco
"""

import argparse
import itertools
from pathlib import Path

import joblib
from sklearn.base import clone

import entrenar

MODELO_PRODUCCION = "docker/modelo.joblib"

# Los modelos del TP. La regresión lineal no tiene alpha, así que se
# prueba una sola vez por imputador
MODELOS = ["lineal", "lasso", "ridge", "elasticnet"]
IMPUTADORES = ["knn", "mediana"]
# Valores de alpha espaciados en escala logarítmica, como los que
# probaban LassoCV y RidgeCV en el TP
ALPHAS = [0.01, 0.1, 1.0, 10.0]


def combinaciones() -> list[tuple[str, str, float]]:
    """Arma la grilla de configuraciones a probar.

    Returns:
        Una lista de (modelo, imputador, alpha).
    """
    grilla = []
    for modelo, imputador in itertools.product(MODELOS, IMPUTADORES):
        alphas = [1.0] if modelo == "lineal" else ALPHAS
        grilla += [(modelo, imputador, alpha) for alpha in alphas]
    return grilla


def medir_produccion() -> dict[str, float]:
    """Mide el modelo de producción con las métricas de la grilla.

    Para rmse_cv se reentrena su configuración en cada partición, igual
    que las combinaciones de la grilla; rmse_test es el modelo tal cual
    está en producción.

    Returns:
        Diccionario con el nombre del modelo, su alpha, rmse_cv,
        rmse_cv_desvio y rmse_test.
    """
    produccion = joblib.load(MODELO_PRODUCCION)
    X_train, X_test, y_train, y_test = entrenar.cargar_datos()
    metricas_cv = entrenar.validar_cruzado(clone(produccion), X_train,
                                           y_train)
    metricas = entrenar.evaluar(produccion, X_test, y_test)
    regresor = produccion.named_steps["modelo"]
    return {"modelo": type(regresor).__name__,
            "alpha": getattr(regresor, "alpha", ""),
            **metricas_cv, "rmse_test": metricas["rmse_test"]}


def resumir(resultados: list[dict], produccion: dict[str, float]) -> str:
    """Arma una tabla en Markdown con las mejores y producción.

    Args:
        resultados: lo que devolvió cada entrenamiento, de mejor
            a peor según rmse_cv.
        produccion: las métricas del modelo de producción.

    Returns:
        El resumen en Markdown.
    """
    lineas = [
        "| Versión | Modelo | Imputador | Alpha | RMSE cv | RMSE test |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| producción | {produccion['modelo']} | | {produccion['alpha']} "
        f"| {produccion['rmse_cv']:.2f} "
        f"± {produccion['rmse_cv_desvio']:.2f} "
        f"| {produccion['rmse_test']:.2f} |",
    ]
    for r in resultados[:5]:
        alpha = "" if r["modelo"] == "lineal" else r["alpha"]
        lineas.append(f"| v{r['version']} | {r['modelo']} | "
                      f"{r['imputador']} | {alpha} | {r['rmse_cv']:.2f} "
                      f"± {r['rmse_cv_desvio']:.2f} | {r['rmse_test']:.2f} |")
    mejor = resultados[0]
    diferencia = produccion["rmse_cv"] - mejor["rmse_cv"]
    lineas += ["", f"La mejor de la grilla es v{mejor['version']}: "
               f"{diferencia:+.2f} de RMSE cv respecto de producción "
               f"(positivo es mejor). Si la diferencia es menor que el "
               f"desvío, es ruido."]
    return "\n".join(lineas)


def main() -> None:
    """Entrena la grilla entera y la compara con producción."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--resumen", help="dónde guardar el resumen .md")
    parser.add_argument("--autor", help="quién entrena (por defecto, el "
                        "usuario del sistema)")
    args = parser.parse_args()

    grilla = combinaciones()
    resultados = []
    for i, (modelo, imputador, alpha) in enumerate(grilla, start=1):
        resultado = entrenar.entrenar_y_registrar(modelo, imputador, alpha,
                                                  autor=args.autor)
        resultados.append({"modelo": modelo, "imputador": imputador,
                           "alpha": alpha, **resultado})
        print(f"[{i}/{len(grilla)}] v{resultado['version']} {modelo} "
              f"{imputador} alpha={alpha}: rmse_cv "
              f"{resultado['rmse_cv']:.2f}")

    resultados.sort(key=lambda r: r["rmse_cv"])
    resumen = resumir(resultados, medir_produccion())
    print("\n" + resumen)
    if args.resumen:
        Path(args.resumen).write_text(resumen)


if __name__ == "__main__":
    main()
