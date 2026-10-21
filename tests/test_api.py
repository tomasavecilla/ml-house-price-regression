"""Tests de la API (api/main.py), sin levantar el servidor.

TestClient llama a la aplicación directamente, así que corren rápido
y no necesitan un puerto libre.
"""

from fastapi.testclient import TestClient

from api.main import app

cliente = TestClient(app)

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
