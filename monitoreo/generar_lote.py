"""Simula un lote de casas nuevas que le llega a la inmobiliaria.

Boston no tiene fechas, así que los lotes se arman a partir del mismo
dataset. Son simulaciones, y cada tipo reproduce un problema distinto:

- zona-nueva (data drift): la inmobiliaria empieza a tasar en una zona
  residencial más cara. Se toman casas reales con más habitaciones,
  menos población de bajos recursos y menos criminalidad. Cambia la
  distribución de las entradas; la relación con el precio es la misma.
- suba-mercado (concept drift): casas como las de siempre, pero se
  venden un 30% más caras. Las entradas no cambian; cambia el precio
  que corresponde a cada casa.

Las casas son filas reales, sin ruido agregado: un ruido cambiaría la
forma de variables como ZN (casi siempre 0) y aparecería un drift que
no es el que se quiere mostrar.

Uso (desde la raíz del repo):
    python -m monitoreo.generar_lote --tipo zona-nueva
    python -m monitoreo.generar_lote --tipo suba-mercado --filas 150
"""

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import entrenar

SUBA_MERCADO = 1.3


def elegir_casas(datos: pd.DataFrame,
                 tipo: Literal["zona-nueva", "suba-mercado"],
                 filas: int, generador: np.random.Generator) -> pd.DataFrame:
    """Elige las casas del lote según el tipo de cambio a simular.

    Args:
        datos: el dataset original, sin B y con MEDV.
        tipo: "zona-nueva" sesga hacia casas grandes en zonas con poca
            pobreza y criminalidad; "suba-mercado" elige al azar.
        filas: cantidad de casas del lote.
        generador: generador de números aleatorios, para que el lote
            se pueda reproducir con la misma semilla.

    Returns:
        Las casas elegidas, con índice nuevo.
    """
    if tipo == "zona-nueva":
        zona = datos[(datos["RM"] > 6.3) & (datos["LSTAT"] < 10)
                     & (datos["CRIM"] < 1)]
    else:
        zona = datos
    elegidas = generador.choice(len(zona), size=filas, replace=True)
    return zona.iloc[elegidas].reset_index(drop=True)


def main() -> None:
    """Genera el lote y lo guarda en datos/lotes/<tipo>.csv."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tipo", required=True,
                        choices=["zona-nueva", "suba-mercado"])
    parser.add_argument("--filas", type=int, default=100)
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args()

    generador = np.random.default_rng(args.semilla)
    datos = pd.read_csv(entrenar.DATOS).drop(columns=entrenar.DESCARTADAS)
    datos = datos.dropna(subset=[entrenar.OBJETIVO])

    lote = elegir_casas(datos, args.tipo, args.filas, generador)
    if args.tipo == "suba-mercado":
        # No se recorta en 50: ese tope era un límite de cómo se
        # registraron los datos originales, no del mercado
        lote[entrenar.OBJETIVO] = lote[entrenar.OBJETIVO] * SUBA_MERCADO

    destino = Path(entrenar.LOTES) / f"{args.tipo}.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    lote.round(4).to_csv(destino, index=False)
    print(f"{len(lote)} casas de tipo {args.tipo} en {destino}")


if __name__ == "__main__":
    main()
