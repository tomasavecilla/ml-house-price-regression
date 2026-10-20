"""Sirve el modelo de precio de casas como una API HTTP.

Es el mismo pipeline que usa docker/inferencia.py, pero en vez de
procesar un CSV entero responde una casa por pedido.

Uso:
    uvicorn api.main:app --reload      # desde la raíz del repo
    curl localhost:8000/salud
    curl localhost:8000/predecir -H "Content-Type: application/json" \\
        -d '{"CRIM": 0.09, "ZN": 0, "INDUS": 25.65, "CHAS": 0,
             "NOX": 0.58, "RM": 5.96, "AGE": 92.9, "DIS": 2.09,
             "RAD": 2, "TAX": 188, "PTRATIO": 19.1, "LSTAT": 17.9}'
    La documentación interactiva queda en localhost:8000/docs
"""

import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# La ruta del modelo se configura por variable de entorno: en el repo
# está en docker/, dentro del contenedor en otro lado
MODELO = os.environ.get("MODELO", "docker/modelo.joblib")

app = FastAPI(title="Precio de casas")

# Se carga una sola vez, al arrancar: cargar el pipeline en cada pedido
# haría lenta la API sin ningún beneficio
pipeline = joblib.load(MODELO)
# El pipeline recuerda con qué columnas y en qué orden se entrenó
COLUMNAS = list(pipeline.feature_names_in_)


class Casa(BaseModel):
    """Características de una casa, las mismas que usó el entrenamiento.

    Todos los campos son obligatorios, pero aceptan null: si falta la
    clave es un error del cliente (422), si viene en null el pipeline
    imputa el valor como lo hizo al entrenar.
    """

    CRIM: float | None = Field(
        ge=0, description="Tasa de criminalidad per cápita de la zona")
    ZN: float | None = Field(
        ge=0, le=100, description="Porcentaje de suelo residencial en "
        "lotes de más de 25.000 pies cuadrados")
    INDUS: float | None = Field(
        ge=0, le=100, description="Porcentaje de suelo industrial")
    CHAS: float | None = Field(
        ge=0, le=1, description="1 si limita con el río Charles, 0 si no")
    NOX: float | None = Field(
        ge=0, description="Concentración de óxidos de nitrógeno, en "
        "partes por 10 millones")
    RM: float | None = Field(
        ge=0, description="Cantidad promedio de habitaciones")
    AGE: float | None = Field(
        ge=0, le=100, description="Porcentaje de viviendas construidas "
        "antes de 1940")
    DIS: float | None = Field(
        ge=0, description="Distancia ponderada a cinco centros de empleo")
    RAD: float | None = Field(
        ge=0, description="Índice de acceso a autopistas radiales")
    TAX: float | None = Field(
        ge=0, description="Impuesto inmobiliario cada 10.000 USD")
    PTRATIO: float | None = Field(
        ge=0, description="Alumnos por docente en la zona")
    LSTAT: float | None = Field(
        ge=0, le=100, description="Porcentaje de población de bajo nivel "
        "socioeconómico")


class Prediccion(BaseModel):
    """Respuesta de /predecir."""

    precio_estimado: float = Field(
        description="Precio estimado, redondeado a un decimal")
    unidad: str = "miles de USD"


@app.get("/salud")
def salud() -> dict[str, str]:
    """Responde ok si la API está levantada y el modelo cargado.

    Returns:
        {"estado": "ok"}.
    """
    return {"estado": "ok"}


@app.post("/predecir")
def predecir(casa: Casa) -> Prediccion:
    """Estima el precio de una casa.

    Args:
        casa: características de la casa. FastAPI ya la validó contra
            el modelo Casa; acá llega con los tipos correctos.

    Returns:
        El precio estimado en miles de dólares.
    """
    # Una fila con las columnas en el orden del entrenamiento; el
    # pipeline escala, imputa y predice igual que en inferencia.py
    fila = pd.DataFrame([casa.model_dump()], columns=COLUMNAS)
    precio = pipeline.predict(fila)[0]
    return Prediccion(precio_estimado=round(float(precio), 1))
