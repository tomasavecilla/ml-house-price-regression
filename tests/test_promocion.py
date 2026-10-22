"""Tests del reentrenamiento y la promoción de versiones.

Usan un MLflow sqlite en una carpeta temporal: nunca tocan el registro
publicado ni el mlflow.db local.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import mlflow
import pytest

from entrenamiento import entrenar, exportar_modelo, promover, reentrenar

MODELO = Path(__file__).resolve().parent.parent / "docker" / "modelo.joblib"
METRICAS_CHAMPION = {"rmse_test": 4.90, "mae_test": 3.40, "r2_test": 0.690}
METRICAS_CHALLENGER = {"rmse_test": 4.50, "mae_test": 3.20, "r2_test": 0.720}


@pytest.fixture
def registro(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Un registro de MLflow vacío con una versión del modelo actual.

    Returns:
        El número de la versión registrada.
    """
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(entrenar, "TRACKING_URI", uri)
    monkeypatch.setattr(exportar_modelo, "TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            joblib.load(MODELO), name="modelo",
            registered_model_name=entrenar.NOMBRE_MODELO,
            skops_trusted_types=["numpy.dtype"])
    return str(info.registered_model_version)


def comparacion(mejora: float) -> str:
    """Arma una comparación de ejemplo con la mejora indicada."""
    return reentrenar.armar_comparacion(
        ["datos/lotes/zona-nueva.csv"], (484, 122), METRICAS_CHAMPION,
        METRICAS_CHALLENGER, mejora, "21", "20",
        "https://mlflow.ejemplo/#/models/precio-casas/versions/21", None)


def test_comparacion_recomienda_promover_si_mejora() -> None:
    """Con una mejora de al menos MEJORA_MINIMA recomienda promover y
    dice cómo."""
    texto = comparacion(0.08)
    assert "Recomendación: promover" in texto
    assert "Champion (producción, versión 20)" in texto
    assert "Challenger (versión 21)" in texto
    assert "versions/21" in texto
    assert "promover.yml -f version=21" in texto


def test_comparacion_no_recomienda_si_no_mejora() -> None:
    """Con menos mejora que MEJORA_MINIMA recomienda no promover."""
    texto = comparacion(reentrenar.MEJORA_MINIMA / 2)
    assert "Recomendación: no promover" in texto


@pytest.mark.parametrize("texto", ["0", "-3", "abc", "2.5", ""])
def test_rechaza_versiones_invalidas(texto: str) -> None:
    """Solo se aceptan enteros positivos como número de versión."""
    with pytest.raises(argparse.ArgumentTypeError):
        exportar_modelo.version_valida(texto)


def test_exporta_una_version_sin_alias(
        registro: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con --version exporta esa versión aunque no haya @champion."""
    destino = tmp_path / "modelo.joblib"
    metadatos = tmp_path / "modelo.json"
    monkeypatch.setattr(exportar_modelo, "DESTINO", destino)
    monkeypatch.setattr(exportar_modelo, "METADATOS", metadatos)
    monkeypatch.setattr(sys, "argv", ["exportar_modelo", "--version",
                                      registro])
    exportar_modelo.main()
    assert destino.exists()
    assert json.loads(metadatos.read_text())["version"] == int(registro)


def test_solo_metadatos_no_toca_el_binario(
        registro: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con --solo-metadatos escribe modelo.json y no el binario, y
    repetirlo da el mismo archivo."""
    destino = tmp_path / "modelo.joblib"
    metadatos = tmp_path / "modelo.json"
    monkeypatch.setattr(exportar_modelo, "DESTINO", destino)
    monkeypatch.setattr(exportar_modelo, "METADATOS", metadatos)
    monkeypatch.setattr(sys, "argv", ["exportar_modelo", "--version",
                                      registro, "--solo-metadatos"])
    exportar_modelo.main()
    primero = metadatos.read_text()
    exportar_modelo.main()
    assert not destino.exists()
    assert metadatos.read_text() == primero


def test_promover_asigna_el_alias(
        registro: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asigna @champion a la versión pedida y escribe el resumen."""
    resumen = tmp_path / "resumen.md"
    monkeypatch.setattr(sys, "argv", [
        "promover", "--version", registro, "--asignar",
        "--resumen", str(resumen), "--estado", "Todo bien."])
    promover.main()
    cliente = mlflow.MlflowClient()
    champion = cliente.get_model_version_by_alias(entrenar.NOMBRE_MODELO,
                                                  "champion")
    assert str(champion.version) == registro
    assert f"Promoción de la versión {registro}" in resumen.read_text()
