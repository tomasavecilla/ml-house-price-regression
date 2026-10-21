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
GRIS = "#7a807f"

# Columnas del modelo, en el orden del CSV de entrenamiento
COLUMNAS = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
            "RAD", "TAX", "PTRATIO", "LSTAT"]
# Nombre corto de cada variable, para las etiquetas y el gráfico
NOMBRES = {
    "RM": "Habitaciones", "AGE": "Antigüedad", "CRIM": "Criminalidad",
    "ZN": "Lotes grandes", "INDUS": "Industria", "LSTAT": "Bajos ingresos",
    "PTRATIO": "Alumnos/docente", "NOX": "Contaminación",
    "DIS": "Dist. a empleo", "TAX": "Impuesto", "CHAS": "Río Charles",
    "RAD": "Autopistas",
}
# Descripción, valor por defecto, mínimo, máximo y paso de cada campo
# numérico. El valor por defecto es la mediana de las casas válidas del
# dataset (entrenar.leer_dataset()); los límites, su mínimo y máximo.
# Están escritos acá para que la app no dependa del código de
# entrenamiento
CAMPOS = {
    "RM": ("Cantidad promedio de habitaciones", 6.2, 3.0, 9.0, 0.1),
    "AGE": ("Porcentaje de viviendas construidas antes de 1940", 77.5,
            0.0, 100.0, 1.0),
    "CRIM": ("Tasa de criminalidad per cápita de la zona", 0.26, 0.0,
             90.0, 0.05),
    "ZN": ("Porcentaje de suelo residencial en lotes grandes", 0.0, 0.0,
           100.0, 1.0),
    "INDUS": ("Porcentaje de suelo industrial", 9.7, 0.0, 28.0, 0.1),
    "LSTAT": ("Porcentaje de población de bajos ingresos", 11.4, 1.0,
              38.0, 0.1),
    "PTRATIO": ("Alumnos por docente en la zona", 19.0, 12.0, 22.0, 0.1),
    "NOX": ("Óxidos de nitrógeno, en partes por 10 millones", 0.538,
            0.38, 0.88, 0.001),
    "DIS": ("Distancia ponderada a cinco centros de empleo", 3.2, 1.0,
            12.5, 0.1),
    "TAX": ("Impuesto inmobiliario cada 10.000 USD", 330.0, 187.0, 711.0,
            1.0),
}
# RAD es un índice: solo existen estos valores en el dataset
VALORES_RAD = [1, 2, 3, 4, 5, 6, 7, 8, 24]
# Cómo se agrupan los campos en el formulario, de a tres por fila
SECCIONES = {
    "Vivienda": ["RM", "AGE", "CHAS"],
    "Zona": ["CRIM", "ZN", "INDUS", "LSTAT", "PTRATIO"],
    "Entorno y accesibilidad": ["NOX", "DIS", "TAX", "RAD"],
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
# Streamlit, las tarjetas y los textos del encabezado y el resultado
ESTILOS = f"<style>@import url('{FUENTES}');" + """
#MainMenu, footer, .stAppDeployButton, [data-testid="stStatusWidget"],
.stNumberInput button {display: none;}
.block-container {padding-top: 1.6rem; max-width: 1440px;}
[data-testid="stVerticalBlock"] {gap: 0.55rem;}
/* Tarjetas: los contenedores con borde */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: #fdfcfa; border: 1px solid #e4e0d6; border-radius: 14px;
  box-shadow: 0 1px 2px rgba(28,34,38,0.04);}
[data-testid="stVerticalBlockBorderWrapper"] > div {padding: 1.1rem 1.4rem;}
/* La columna del resultado acompaña el scroll */
[data-testid="stColumn"]:nth-of-type(2) {position: sticky; top: 1rem;
  align-self: flex-start;}
.stNumberInput label p, .stSelectbox label p, .stMultiSelect label p,
.stToggle label p {font-size: 0.82rem; color: #4a5054;}
.mono, .eyebrow, .pildora, .precio, .detalle {
  font-family: 'IBM Plex Mono', monospace;}
.eyebrow {font-size: 0.72rem; letter-spacing: 0.12em; color: #1d6f68;
  text-transform: uppercase; margin: 0.7rem 0 0.1rem;}
.encabezado {display: flex; justify-content: space-between;
  align-items: center; margin-bottom: 0.6rem;}
.marca {font-size: 1.7rem; font-weight: 600; letter-spacing: -0.02em;}
.marca span {color: #7a807f; font-weight: 400; margin-left: 0.5rem;}
.marca b {color: #1d6f68;}
.pildora {font-size: 0.78rem; padding: 0.25rem 0.8rem;
  border-radius: 999px; border: 1px solid;}
.ok {color: #1d6f68; border-color: #1d6f68; background: #e4efed;}
.error {color: #b4541a; border-color: #e08a3c; background: #f8e9dc;}
.precio {font-size: 3rem; font-weight: 500; color: #1d6f68;
  line-height: 1.1; margin: 0.2rem 0;}
.detalle {font-size: 0.85rem; color: #4a5054; margin-bottom: 0.6rem;}
.sube {color: #1d6f68;} .baja {color: #b4541a;}
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


def html(contenedor: st.delta_generator.DeltaGenerator, texto: str
         ) -> None:
    """Escribe HTML con las clases de ESTILOS dentro de un contenedor.

    Args:
        contenedor: dónde escribir (st, una columna, una tarjeta).
        texto: el HTML.
    """
    contenedor.markdown(texto, unsafe_allow_html=True)


def encabezado() -> None:
    """Muestra la marca y una píldora con el estado de la API."""
    info = consultar_api("/modelo")
    if info is None:
        pildora = '<span class="pildora error">● API sin conexión</span>'
    else:
        alpha = f" α={info['alpha']}" if info["alpha"] is not None else ""
        pildora = (f'<span class="pildora ok">● API en línea · '
                   f'{info["regresor"]}{alpha} · {info["imputador"]} · '
                   f'modelo del {info["actualizado"]}</span>')
    html(st, f'<div class="encabezado"><div class="marca"><b>◆</b> Tasador'
             f'<span>Inmobiliaria</span></div>{pildora}</div>')


def campo_entrada(columna: st.delta_generator.DeltaGenerator,
                  campo: str) -> float:
    """Dibuja el control de un campo y devuelve su valor.

    Args:
        columna: dónde dibujarlo.
        campo: nombre de la variable (RM, CHAS, RAD, ...).

    Returns:
        El valor elegido, como float.
    """
    etiqueta = f"{NOMBRES[campo]} · {campo}"
    if campo == "CHAS":
        return float(columna.toggle(etiqueta,
                                    help="Limita con el río Charles"))
    if campo == "RAD":
        return float(columna.selectbox(
            etiqueta, VALORES_RAD, index=4,
            help="Índice de acceso a autopistas radiales"))
    ayuda, valor, minimo, maximo, paso = CAMPOS[campo]
    return columna.number_input(etiqueta, min_value=minimo,
                                max_value=maximo, value=valor, step=paso,
                                help=ayuda)


def formulario_casa() -> dict[str, float | None]:
    """Dibuja los campos de una casa y devuelve sus valores.

    Returns:
        Un diccionario columna -> valor, con None en las que el tasador
        marcó como desconocidas (la API las imputa).
    """
    casa: dict[str, float | None] = {}
    for seccion, campos in SECCIONES.items():
        html(st, f'<p class="eyebrow">{seccion}</p>')
        # Filas de tres columnas; la última puede quedar incompleta
        for inicio in range(0, len(campos), 3):
            columnas = st.columns(3, vertical_alignment="bottom")
            for campo, columna in zip(campos[inicio:inicio + 3], columnas):
                casa[campo] = campo_entrada(columna, campo)
    faltan, boton = st.columns([3, 1], vertical_alignment="bottom")
    desconocidas = faltan.multiselect(
        "Datos que no conozco (el modelo los completa)",
        options=list(CAMPOS), format_func=lambda c: NOMBRES[c],
        placeholder="Ninguno")
    boton.form_submit_button("Tasar", type="primary", width="stretch")
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
        "variable": [f"{NOMBRES[v]} · {v}" for v in contribuciones],
        "aporte": [float(a) for a in contribuciones.values()],
    })
    datos["signo"] = datos["aporte"].map(lambda a: "sube" if a >= 0
                                         else "baja")
    datos["etiqueta"] = datos["aporte"].map(lambda a: f"{a:+.1f}")
    orden = datos.reindex(datos["aporte"].abs().sort_values(
        ascending=False).index)["variable"].tolist()
    # Eje simétrico y con aire a los costados, para que las cifras de
    # las barras negativas no pisen los nombres
    limite = float(datos["aporte"].abs().max()) * 1.4 or 1.0
    base = alt.Chart(datos).encode(
        y=alt.Y("variable:N", sort=orden, title=None,
                axis=alt.Axis(labelFont="IBM Plex Mono", labelFontSize=11,
                              labelColor=TINTA, ticks=False,
                              labelPadding=8)),
        x=alt.X("aporte:Q", title="miles de USD",
                scale=alt.Scale(domain=[-limite, limite], nice=False),
                axis=alt.Axis(labelFont="IBM Plex Mono", labelColor=GRIS,
                              titleFont="IBM Plex Mono", titleColor=GRIS,
                              titleFontSize=10, titleFontWeight="normal",
                              gridColor="#ebe8e0", tickCount=5)),
        tooltip=[alt.Tooltip("variable", title="Variable"),
                 alt.Tooltip("aporte", title="Miles de USD",
                             format="+.2f")],
    )
    barras = base.mark_bar(cornerRadiusEnd=4, size=16).encode(
        color=alt.Color("signo:N", legend=None, scale=alt.Scale(
            domain=["sube", "baja"], range=[TEAL, NARANJA])))
    # La cifra exacta al final de cada barra, del lado hacia donde
    # crece: una capa para las positivas y otra para las negativas
    texto = base.encode(text="etiqueta:N")
    positivas = texto.transform_filter(alt.datum.aporte >= 0).mark_text(
        align="left", dx=5, font="IBM Plex Mono", fontSize=11, color=TINTA)
    negativas = texto.transform_filter(alt.datum.aporte < 0).mark_text(
        align="right", dx=-5, font="IBM Plex Mono", fontSize=11,
        color=TINTA)
    cero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color=GRIS).encode(x="x:Q")
    return alt.layer(cero, barras, positivas, negativas).properties(
        height=26 * len(datos), background="transparent",
        padding={"left": 0, "right": 10, "top": 4, "bottom": 0},
    ).configure_view(strokeWidth=0).configure_axis(domain=False)


def tarjeta_resultado(casa: dict[str, float | None]) -> None:
    """Tasa la casa y muestra el precio y de dónde sale.

    Args:
        casa: valores del formulario, con None en los desconocidos.
    """
    explicacion = consultar_api("/explicar", casa)
    if explicacion is None:
        return
    precio = explicacion["precio_estimado"]
    delta = precio - explicacion["intercepto"]
    clase = "sube" if delta >= 0 else "baja"
    html(st, f'<p class="eyebrow">Precio sugerido</p>'
             f'<div class="precio">{formatear_usd(precio)}</div>'
             f'<div class="detalle">{precio:.1f} miles de USD · '
             f'<span class="{clase}">{delta:+.1f}</span> respecto de '
             f'una casa promedio</div>')
    if any(v is None for v in casa.values()):
        st.caption("Los datos que no conocías los completó el modelo con "
                   "lo que aprendió en el entrenamiento.")
    html(st, '<p class="eyebrow">¿Por qué este precio?</p>')
    st.altair_chart(grafico_contribuciones(explicacion["contribuciones"]),
                    width="stretch", theme=None)
    st.caption(
        f"Se parte del precio de una casa promedio, "
        f"{formatear_usd(explicacion['intercepto'])}, y cada variable lo "
        "empuja hacia arriba (teal) o hacia abajo (naranja). Las barras "
        "suman exactamente el precio porque el modelo es lineal.")


def pestania_una_casa() -> None:
    """Pestaña para tasar una casa: formulario y resultado, lado a
    lado."""
    formulario, resultado = st.columns([58, 42], gap="large")
    # Dentro de un form los campos no disparan nada hasta apretar Tasar;
    # en la primera carga valen sus medianas, así que ya hay resultado
    with formulario, st.container(border=True):
        with st.form("casa", border=False):
            casa = formulario_casa()
    with resultado, st.container(border=True):
        tarjeta_resultado(casa)


def pestania_lote() -> None:
    """Pestaña para tasar un CSV de casas y descargar el resultado."""
    html(st, '<p class="eyebrow">Tasar un lote</p>')
    st.caption("Un CSV con las 12 columnas del modelo, una casa por fila. "
               "Las celdas vacías las completa el modelo.")
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


st.set_page_config(page_title="Tasador · Inmobiliaria", page_icon="◆",
                   layout="wide")
st.markdown(ESTILOS, unsafe_allow_html=True)
encabezado()
una_casa, lote = st.tabs(["Tasar una casa", "Tasar un lote (CSV)"])
with una_casa:
    pestania_una_casa()
with lote, st.container(border=True):
    pestania_lote()
