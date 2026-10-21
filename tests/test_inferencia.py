"""Tests del script de inferencia por lotes (docker/inferencia.py)."""

from pathlib import Path

import pandas as pd
import pytest

import inferencia

DOCKER = Path(__file__).resolve().parent.parent / "docker"
DATOS = Path(__file__).resolve().parent.parent / "datos"


def test_tasa_las_casas_de_ejemplo(tmp_path: Path) -> None:
    """Escribe una fila por casa con la columna precio_estimado."""
    salida = tmp_path / "precios.csv"
    inferencia.main(str(DOCKER / "casas_ejemplo.csv"), str(salida))
    precios = pd.read_csv(salida)
    assert len(precios) == 5
    assert "precio_estimado" in precios.columns
    # Una de las casas de ejemplo viene sin RM: el pipeline la imputa,
    # así que ninguna predicción puede quedar vacía
    assert precios["precio_estimado"].notna().all()


def test_falla_si_falta_una_columna(tmp_path: Path) -> None:
    """Termina con error, y dice qué columna falta, si el CSV no la
    trae."""
    with pytest.raises(SystemExit, match="RM"):
        inferencia.main(str(DATOS / "casas_sin_rm.csv"),
                        str(tmp_path / "precios.csv"))
