"""Tasador de casas: la interfaz de los tasadores de la inmobiliaria.

No carga el modelo: cada tasación es un pedido HTTP a la API (api/).
Esa separación es el punto de la demo: la app puede cambiar de aspecto
y el modelo de versión sin que una se entere de la otra.

Uso:
    streamlit run app/app.py                  # API en localhost:8000
    API_URL=http://api:8000 streamlit run app.py   # dentro de docker
"""

import json
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TEAL = "#1d6f68"
NARANJA = "#e08a3c"
TINTA = "#1c2226"

# Columnas del modelo, en el orden del CSV de entrenamiento
COLUMNAS = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
            "RAD", "TAX", "PTRATIO", "LSTAT"]
# Etiqueta, valor por defecto, mínimo, máximo y paso de cada campo.
# El valor por defecto es la mediana de las casas válidas del dataset
# (entrenar.leer_dataset()); los límites, su mínimo y máximo. Están
# escritos acá para que la app no dependa del código de entrenamiento
CAMPOS = {
    "RM": ("Habitaciones promedio", 6.2, 3.0, 9.0, 0.1),
    "AGE": ("Viviendas anteriores a 1940 (%)", 77.5, 0.0, 100.0, 1.0),
    "CRIM": ("Criminalidad per cápita", 0.26, 0.0, 90.0, 0.05),
    "ZN": ("Suelo residencial en lotes grandes (%)", 0.0, 0.0, 100.0,
           1.0),
    "INDUS": ("Suelo industrial (%)", 9.7, 0.0, 28.0, 0.1),
    "LSTAT": ("Población de bajos ingresos (%)", 11.4, 1.0, 38.0, 0.1),
    "PTRATIO": ("Alumnos por docente", 19.0, 12.0, 22.0, 0.1),
    "NOX": ("Óxidos de nitrógeno (partes por 10M)", 0.538, 0.38, 0.88,
            0.001),
    "DIS": ("Distancia a centros de empleo", 3.2, 1.0, 12.5, 0.1),
    "TAX": ("Impuesto inmobiliario (USD cada 10.000)", 330.0, 187.0,
            711.0, 1.0),
}
# RAD es un índice: solo existen estos valores en el dataset
VALORES_RAD = [1, 2, 3, 4, 5, 6, 7, 8, 24]
SECCIONES = {
    "Vivienda": ["RM", "AGE"],
    "Zona": ["CRIM", "ZN", "INDUS", "LSTAT", "PTRATIO"],
    "Entorno y accesibilidad": ["NOX", "DIS", "TAX"],
}
# Lote de ejemplo (docker/casas_ejemplo.csv): la última casa viene sin
# RM, para mostrar que el pipeline lo imputa
CSV_EJEMPLO = """CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,LSTAT
0.09299,0.0,25.65,0.0,0.581,5.961,92.9,2.0869,2.0,188.0,19.1,17.93
2.24236,0.0,19.58,0.0,0.605,5.854,91.8,2.422,5.0,403.0,14.7,11.64
9.91655,0.0,18.1,0.0,0.693,5.852,77.8,1.5004,24.0,666.0,20.2,29.97
10.233,0.0,18.1,0.0,0.614,6.185,96.7,2.1705,24.0,666.0,20.2,18.03
11.9511,0.0,18.1,0.0,0.659,,100.0,1.2852,24.0,666.0,20.2,12.13
"""

FUENTES = ("https://fonts.googleapis.com/css2?display=swap"
           "&family=IBM+Plex+Mono:wght@400;500"
           "&family=IBM+Plex+Sans:wght@400;500;600")
# CSS para lo que el tema de config.toml no cubre: fuentes, el menú de
# Streamlit y la píldora de estado
ESTILOS = f"<style>@import url('{FUENTES}');" + """
#MainMenu, footer, .stAppDeployButton, [data-testid="stStatusWidget"]
  {visibility: hidden;}
.block-container {padding-top: 2.5rem; max-width: 1100px;}
h1 {font-weight: 600; letter-spacing: -0.02em; margin-bottom: 0;}
.pildora {display: inline-block; font-family: 'IBM Plex Mono', monospace;
  font-size: 0.8rem; padding: 0.2rem 0.7rem; border-radius: 999px;
  border: 1px solid; margin-top: 0.4rem;}
.ok {color: #1d6f68; border-color: #1d6f68; background: #e4efed;}
.error {color: #b4541a; border-color: #e08a3c; background: #f8e9dc;}
[data-testid="stMetricValue"] {font-family: 'IBM Plex Mono', monospace;
  font-weight: 500; color: #1d6f68;}
</style>
"""


def consultar_api(ruta: str, cuerpo: dict | list | None = None
                  ) -> dict | None:
    """Hace un pedido a la API y devuelve la respuesta como diccionario.

    Args:
        ruta: endpoint, por ejemplo "/predecir".
        cuerpo: JSON a enviar. None hace un GET; si no, un POST.

    Returns:
        La respuesta decodificada, o None si la API no respondió o
        devolvió un error (en ese caso ya se mostró el mensaje).
    """
    try:
        if cuerpo is None:
            respuesta = requests.get(API_URL + ruta, timeout=5)
        else:
            respuesta = requests.post(API_URL + ruta, json=cuerpo,
                                      timeout=30)
        respuesta.raise_for_status()
        return respuesta.json()
    except requests.ConnectionError:
        st.error(f"No se pudo conectar con la API en {API_URL}. "
                 "¿Está levantada?")
    except requests.RequestException as error:
        st.error(f"La API respondió con un error: {error}")
    return None


def formatear_usd(miles: float) -> str:
    """Pasa de miles de USD a un texto como "USD 23.400".

    Args:
        miles: precio en miles de dólares (la unidad de MEDV).

    Returns:
        El precio en dólares, con punto como separador de miles.
    """
    return "USD " + f"{miles * 1000:,.0f}".replace(",", ".")


def encabezado() -> None:
    """Muestra el título y una píldora con el estado de la API."""
    st.title("Tasador · Inmobiliaria")
    info = consultar_api("/modelo")
    if info is None:
        pildora = '<span class="pildora error">● API sin conexión</span>'
    else:
        alpha = f" α={info['alpha']}" if info["alpha"] is not None else ""
        pildora = (f'<span class="pildora ok">● API en línea · '
                   f'{info["regresor"]}{alpha} · {info["imputador"]} · '
                   f'modelo del {info["actualizado"]}</span>')
    st.markdown(pildora, unsafe_allow_html=True)


def formulario_casa() -> dict[str, float | None]:
    """Dibuja los campos de una casa y devuelve sus valores.

    Returns:
        Un diccionario columna -> valor, con None en las que el tasador
        marcó como desconocidas (la API las imputa).
    """
    desconocidas = st.multiselect(
        "Datos que no conozco (el modelo los completa)",
        options=list(CAMPOS), format_func=lambda c: CAMPOS[c][0],
        placeholder="Ninguno: tengo todos los datos")
    casa: dict[str, float | None] = {}
    for columna_ui, (seccion, campos) in zip(st.columns(3),
                                             SECCIONES.items()):
        columna_ui.subheader(seccion)
        for campo in campos:
            etiqueta, valor, minimo, maximo, paso = CAMPOS[campo]
            casa[campo] = columna_ui.number_input(
                f"{etiqueta} · {campo}", min_value=minimo,
                max_value=maximo, value=valor, step=paso,
                disabled=campo in desconocidas)
        if seccion == "Zona":
            casa["CHAS"] = float(columna_ui.toggle(
                "Limita con el río Charles · CHAS"))
        if seccion == "Entorno y accesibilidad":
            casa["RAD"] = float(columna_ui.selectbox(
                "Acceso a autopistas radiales · RAD", VALORES_RAD,
                index=4))
    for campo in desconocidas:
        casa[campo] = None
    return casa


def grafico_contribuciones(contribuciones: dict[str, float]) -> alt.Chart:
    """Arma el gráfico de barras de cuánto aporta cada variable.

    Args:
        contribuciones: variable -> miles de USD que suma o resta.

    Returns:
        Barras horizontales ordenadas por magnitud: teal las que suben
        el precio, naranja las que lo bajan.
    """
    datos = pd.DataFrame({
        "variable": [f"{CAMPOS.get(v, (v,))[0]} · {v}"
                     for v in contribuciones],
        "aporte": list(contribuciones.values()),
    })
    datos["etiqueta"] = datos["aporte"].map(lambda a: f"{a:+.1f}")
    base = alt.Chart(datos).encode(
        y=alt.Y("variable:N", sort=alt.EncodingSortField(
            "aporte", op="sum", order="descending"), title=None),
        x=alt.X("aporte:Q", title="Miles de USD"),
        tooltip=["variable", alt.Tooltip("aporte", format="+.2f")],
    )
    barras = base.mark_bar(cornerRadiusEnd=4, size=18).encode(
        color=alt.condition(alt.datum.aporte >= 0, alt.value(TEAL),
                            alt.value(NARANJA)))
    # Valor al final de cada barra, en tinta: la barra da el signo, el
    # texto la cifra exacta
    textos = base.mark_text(align="left", dx=6, color=TINTA,
                            font="IBM Plex Mono").encode(
        text="etiqueta:N",
        x=alt.X("aporte:Q", title="Miles de USD"))
    return (barras + textos).properties(height=26 * len(datos))


def pestania_una_casa() -> None:
    """Pestaña para tasar una casa y ver por qué vale lo que vale."""
    with st.form("casa", border=False):
        casa = formulario_casa()
        tasar = st.form_submit_button("Tasar", type="primary",
                                      width="stretch")
    if not tasar:
        return
    explicacion = consultar_api("/explicar", casa)
    if explicacion is None:
        return
    precio = explicacion["precio_estimado"]
    intercepto = explicacion["intercepto"]
    resultado, porque = st.columns([1, 2], gap="large")
    resultado.metric("Precio sugerido (miles de USD)", f"{precio:.1f}",
                     delta=f"{precio - intercepto:+.1f} vs. casa promedio")
    resultado.markdown(f"### {formatear_usd(precio)}")
    if any(v is None for v in casa.values()):
        resultado.caption("Los datos marcados como desconocidos los "
                          "completó el modelo con lo que aprendió en "
                          "el entrenamiento.")
    porque.subheader("¿Por qué este precio?")
    porque.altair_chart(grafico_contribuciones(
        explicacion["contribuciones"]), width="stretch")
    porque.caption(
        f"Se parte del precio de una casa promedio, "
        f"{formatear_usd(intercepto)}, y cada variable lo empuja hacia "
        "arriba (teal) o hacia abajo (naranja). Las barras suman "
        "exactamente el precio porque el modelo es lineal.")


def pestania_lote() -> None:
    """Pestaña para tasar un CSV de casas y descargar el resultado."""
    st.download_button("Descargar un CSV de ejemplo", CSV_EJEMPLO,
                       "casas_ejemplo.csv", "text/csv")
    archivo = st.file_uploader("CSV con una casa por fila", type="csv")
    if archivo is None:
        return
    casas = pd.read_csv(archivo)
    faltantes = [c for c in COLUMNAS if c not in casas.columns]
    if faltantes:
        st.error(f"Al CSV le faltan las columnas: {', '.join(faltantes)}. "
                 f"Tiene que tener {', '.join(COLUMNAS)}.")
        return
    # Las celdas vacías llegan como NaN, que no existe en JSON: pasando
    # por to_json quedan como null y la API las imputa
    filas = json.loads(casas[COLUMNAS].to_json(orient="records"))
    respuesta = consultar_api("/predecir-lote", filas)
    if respuesta is None:
        return
    casas["precio_estimado"] = respuesta["precios_estimados"]
    st.success(f"{len(casas)} casas tasadas. precio_estimado está en "
               "miles de USD.")
    st.dataframe(casas, width="stretch", hide_index=True)
    st.download_button("Descargar con precios", casas.to_csv(index=False),
                       "precios.csv", "text/csv", type="primary")


st.set_page_config(page_title="Tasador · Inmobiliaria", page_icon="🏠",
                   layout="wide")
st.markdown(ESTILOS, unsafe_allow_html=True)
encabezado()
una_casa, lote = st.tabs(["Tasar una casa", "Tasar un lote (CSV)"])
with una_casa:
    pestania_una_casa()
with lote:
    pestania_lote()
