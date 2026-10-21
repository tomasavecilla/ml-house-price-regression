"""Prueba todas las combinaciones de la grilla y registra cada una.

Cada combinación de modelo, imputador y alpha se entrena con
entrenar.py y queda como una versión de precio-casas. Al final se
ordenan por rmse_cv (validación cruzada sobre train): el test no se usa
para elegir, así sigue siendo una estimación honesta del modelo que se
elija.

Uso:
    python buscar_hiperparametros.py
"""

import itertools

import entrenar

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


def main() -> None:
    """Entrena la grilla entera y muestra las mejores."""
    grilla = combinaciones()
    resultados = []
    for i, (modelo, imputador, alpha) in enumerate(grilla, start=1):
        resultado = entrenar.entrenar_y_registrar(modelo, imputador, alpha)
        resultados.append({"modelo": modelo, "imputador": imputador,
                           "alpha": alpha, **resultado})
        print(f"[{i}/{len(grilla)}] v{resultado['version']} {modelo} "
              f"{imputador} alpha={alpha}: rmse_cv "
              f"{resultado['rmse_cv']:.2f}")

    resultados.sort(key=lambda r: r["rmse_cv"])
    print("\nLas 5 mejores por rmse_cv:")
    for r in resultados[:5]:
        print(f"  v{r['version']:<3} {r['modelo']:10} {r['imputador']:7} "
              f"alpha={r['alpha']:<5} rmse_cv {r['rmse_cv']:.2f} "
              f"(± {r['rmse_cv_desvio']:.2f})")
    mejor = resultados[0]
    print(f"\nCandidata: v{mejor['version']}. Si te convence, ponele el "
          "alias champion en la UI y corré exportar_modelo.py")


if __name__ == "__main__":
    main()
