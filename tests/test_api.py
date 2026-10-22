"""Tests de la API (api/main.py), sin levantar el servidor.

TestClient llama a la aplicación directamente, así que corren rápido
y no necesitan un puerto libre.
"""

import json
from pathlib import Path

import inferencia
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import main
from api.main import COLUMNAS, app

cliente = TestClient(app)
DOCKER = Path(__file__).resolve().parent.parent / "docker"

# Una casa válida, la primera de docker/casas_ejemplo.csv
CASA = {
    "CRIM": 0.09299, "ZN": 0.0, "INDUS": 25.65, "CHAS": 0.0, "NOX": 0.581,
    "RM": 5.961, "AGE": 92.9, "DIS": 2.0869, "RAD": 2.0, "TAX": 188.0,
    "PTRATIO": 19.1, "LSTAT": 17.93,
}


def test_salud() -> None:
    """La API responde y el modelo está cargado."""
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok"}


def test_predice_una_casa_valida() -> None:
    """Devuelve un precio numérico en miles de USD."""
    respuesta = cliente.post("/predecir", json=CASA)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert isinstance(cuerpo["precio_estimado"], float)
    assert cuerpo["unidad"] == "miles de USD"


def test_imputa_un_campo_en_null() -> None:
    """Un valor null no es error: el pipeline lo imputa."""
    respuesta = cliente.post("/predecir", json={**CASA, "RM": None})
    assert respuesta.status_code == 200
    assert isinstance(respuesta.json()["precio_estimado"], float)


def test_rechaza_si_falta_un_campo() -> None:
    """Que falte la clave sí es error del cliente: 422."""
    sin_rm = {k: v for k, v in CASA.items() if k != "RM"}
    respuesta = cliente.post("/predecir", json=sin_rm)
    assert respuesta.status_code == 422


def test_rechaza_texto_en_un_campo_numerico() -> None:
    """Pydantic valida los tipos antes de que el pedido llegue al
    modelo."""
    respuesta = cliente.post("/predecir", json={**CASA, "RM": "muchas"})
    assert respuesta.status_code == 422


def test_predice_un_lote() -> None:
    """Devuelve un precio por casa, en el mismo orden."""
    respuesta = cliente.post("/predecir-lote",
                             json=[CASA, {**CASA, "RM": 7.5}])
    assert respuesta.status_code == 200
    precios = respuesta.json()["precios_estimados"]
    assert len(precios) == 2
    # El primero coincide con /predecir: es el mismo pipeline
    individual = cliente.post("/predecir", json=CASA).json()
    assert precios[0] == individual["precio_estimado"]


def test_explica_sumando_las_contribuciones() -> None:
    """intercepto + contribuciones == precio: el modelo es lineal."""
    respuesta = cliente.post("/explicar", json=CASA)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    # Una contribución por columna de entrada, aunque reordenadas
    assert set(cuerpo["contribuciones"]) == set(CASA)
    suma = cuerpo["intercepto"] + sum(cuerpo["contribuciones"].values())
    # Tolerancia por los redondeos a 2 decimales de cada sumando
    assert abs(suma - cuerpo["precio_estimado"]) < 0.15
    individual = cliente.post("/predecir", json=CASA).json()
    assert cuerpo["precio_estimado"] == individual["precio_estimado"]


def test_describe_el_modelo() -> None:
    """Dice qué regresor e imputador hay en producción."""
    respuesta = cliente.get("/modelo")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["regresor"] in {"LinearRegression", "Ridge", "Lasso",
                                  "ElasticNet"}
    assert cuerpo["imputador"] in {"KNNImputer", "SimpleImputer"}
    assert len(cuerpo["columnas"]) == 12


def test_modelo_muestra_la_version_del_registro(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Con modelo.json, /modelo dice de qué versión salió el binario."""
    metadatos = tmp_path / "modelo.json"
    metadatos.write_text(json.dumps({
        "version": 21, "run_id": "abc123", "rmse_test": 4.5,
        "registrada": "2026-10-22"}))
    monkeypatch.setattr(main, "METADATOS", metadatos)
    cuerpo = cliente.get("/modelo").json()
    assert cuerpo["version"] == 21
    assert cuerpo["run_id"] == "abc123"
    assert cuerpo["rmse_test"] == 4.5


def test_modelo_sin_metadatos_no_falla(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin modelo.json, o con uno roto, /modelo responde igual y la
    versión queda en null."""
    monkeypatch.setattr(main, "METADATOS", tmp_path / "no-existe.json")
    respuesta = cliente.get("/modelo")
    assert respuesta.status_code == 200
    assert respuesta.json()["version"] is None
    roto = tmp_path / "roto.json"
    roto.write_text("{esto no es json")
    monkeypatch.setattr(main, "METADATOS", roto)
    respuesta = cliente.get("/modelo")
    assert respuesta.status_code == 200
    assert respuesta.json()["version"] is None


def test_api_y_lotes_dan_el_mismo_precio(tmp_path: Path) -> None:
    """La API y el script por lotes predicen exactamente igual.

    Las dos usan inferencia.predecir, así que no puede haber diferencias
    ni siquiera de redondeo. Si alguien volviera a repetir la lógica en
    la API, este test lo detecta.
    """
    entrada = DOCKER / "casas_ejemplo.csv"
    salida = tmp_path / "precios.csv"
    inferencia.main(str(entrada), str(salida))
    por_lotes = pd.read_csv(salida)["precio_estimado"].tolist()

    # La API valida los campos: los NaN del CSV viajan como null
    casas = pd.read_csv(entrada)[COLUMNAS].astype(object)
    casas = casas.where(casas.notna(), None).to_dict(orient="records")
    respuesta = cliente.post("/predecir-lote", json=casas)
    assert respuesta.status_code == 200
    assert respuesta.json()["precios_estimados"] == por_lotes

    # /predecir, una casa por pedido, también coincide
    for casa, esperado in zip(casas, por_lotes):
        individual = cliente.post("/predecir", json=casa).json()
        assert individual["precio_estimado"] == esperado
