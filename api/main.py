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
    curl localhost:8000/modelo              # qué hay en producción
    La documentación interactiva queda en localhost:8000/docs
"""

import os
from datetime import datetime

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
preprocesamiento = pipeline.named_steps["preprocesamiento"]
modelo = pipeline.named_steps["modelo"]
# El preprocesamiento reordena las columnas (primero las continuas,
# después las discretas) y las nombra "continuas__RM": los nombres de
# los coeficientes hay que pedírselos a él, no tomarlos de la entrada
NOMBRES_TRANSFORMADAS = [n.split("__")[-1]
                         for n in preprocesamiento.get_feature_names_out()]


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


class PrediccionLote(BaseModel):
    """Respuesta de /predecir-lote: un precio por casa, en el mismo
    orden en que llegaron."""

    precios_estimados: list[float]
    unidad: str = "miles de USD"


class Explicacion(BaseModel):
    """Respuesta de /explicar: el precio y de dónde sale."""

    precio_estimado: float
    intercepto: float = Field(
        description="Precio de una casa promedio: lo que predice el "
        "modelo cuando todas las variables valen su media")
    contribuciones: dict[str, float] = Field(
        description="Cuántos miles de USD suma o resta cada variable "
        "respecto del intercepto")
    unidad: str = "miles de USD"


class InfoModelo(BaseModel):
    """Respuesta de /modelo: qué hay en producción."""

    regresor: str
    alpha: float | None = Field(
        description="Fuerza de la regularización; null si el regresor "
        "no tiene")
    imputador: str
    columnas: list[str]
    actualizado: str = Field(
        description="Fecha en que se exportó el binario (fecha de "
        "modificación del archivo)")


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


@app.post("/predecir-lote")
def predecir_lote(casas: list[Casa]) -> PrediccionLote:
    """Estima el precio de varias casas en un solo pedido.

    Args:
        casas: lista de casas, cada una validada contra el modelo Casa.

    Returns:
        Los precios estimados, uno por casa y en el mismo orden.
    """
    # Un solo predict para todas: el pipeline está pensado para trabajar
    # por lotes, y es mucho más rápido que un pedido por casa
    filas = pd.DataFrame([c.model_dump() for c in casas], columns=COLUMNAS)
    precios = pipeline.predict(filas)
    return PrediccionLote(
        precios_estimados=[round(float(p), 1) for p in precios])


@app.post("/explicar")
def explicar(casa: Casa) -> Explicacion:
    """Estima el precio de una casa y dice cuánto aporta cada variable.

    Esta descomposición es exacta solo porque el modelo es lineal: el
    precio es intercepto + suma de coeficiente × valor escalado, así
    que cada sumando se puede atribuir a una variable. Con un árbol o
    una red habría que usar otra técnica (SHAP, por ejemplo).

    Args:
        casa: características de la casa, igual que en /predecir.

    Returns:
        El precio, el intercepto y la contribución de cada variable en
        miles de USD. precio = intercepto + suma de contribuciones.
    """
    fila = pd.DataFrame([casa.model_dump()], columns=COLUMNAS)
    # Se aplica el preprocesamiento aparte para ver la fila como la ve
    # el regresor: escalada, imputada y con las columnas reordenadas
    transformada = preprocesamiento.transform(fila)[0]
    contribuciones = modelo.coef_ * transformada
    precio = modelo.intercept_ + contribuciones.sum()
    return Explicacion(
        precio_estimado=round(float(precio), 1),
        intercepto=round(float(modelo.intercept_), 2),
        contribuciones={nombre: round(float(valor), 2) for nombre, valor
                        in zip(NOMBRES_TRANSFORMADAS, contribuciones)},
    )


@app.get("/modelo")
def info_modelo() -> InfoModelo:
    """Describe el modelo que está en producción.

    Returns:
        Regresor, alpha, imputador, columnas de entrada y fecha de
        exportación del binario.
    """
    imputador = (preprocesamiento.named_transformers_["continuas"]
                 .named_steps["imputar"])
    actualizado = datetime.fromtimestamp(os.path.getmtime(MODELO))
    return InfoModelo(
        regresor=type(modelo).__name__,
        # LinearRegression no tiene alpha: para él get_params no la trae
        alpha=modelo.get_params().get("alpha"),
        imputador=type(imputador).__name__,
        columnas=COLUMNAS,
        actualizado=actualizado.strftime("%Y-%m-%d"),
    )
