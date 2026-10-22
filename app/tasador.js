/* Lógica del tasador: arma los controles, llama a la API y dibuja el
   resultado. No hay modelo acá: cada tasación es un pedido HTTP.

   Las rutas empiezan con /api/: el navegador habla siempre con el mismo
   origen que sirvió la página, y nginx (nginx.conf) reenvía esos pedidos
   al servicio api:8000. Así no hace falta CORS ni saber dónde está la
   API desde el navegador.

   La ilustración de la casa está en ilustracion.js, que se carga antes. */

"use strict";

const API = "/api";

// Columnas del modelo, en el orden del CSV de entrenamiento
const COLUMNAS = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
                  "RAD", "TAX", "PTRATIO", "LSTAT"];
// Nombre corto de cada variable, para la lista de aportes
const NOMBRES = {
  RM: "Tamaño de las casas", AGE: "Antigüedad del barrio",
  CRIM: "Seguridad", ZN: "Lotes grandes", INDUS: "Industria cerca",
  LSTAT: "Nivel socioeconómico", PTRATIO: "Escuelas",
  NOX: "Calidad del aire", DIS: "Cercanía al empleo",
  TAX: "Impuesto inmobiliario", CHAS: "Río Charles",
  RAD: "Acceso a autopistas",
};

// Cada campo numérico se muestra como un slider. Los valores por
// defecto son la mediana de las casas válidas del dataset
// (entrenar.leer_dataset()) y los límites, su mínimo y máximo. Están
// escritos acá para que la app no dependa del código de entrenamiento.
//   etiqueta: lo que lee el usuario; unidad: va al lado del valor
//   decimales: cuántos se muestran; potencia: para variables muy
//   sesgadas (CRIM) el slider recorre el rango con una curva, así el
//   extremo bajo, donde está casi toda la gente, tiene espacio
//   niveles: [límite, texto] en orden; sirven para poner en palabras el
//   número ("Zona tranquila"). El último no tiene límite
const CAMPOS = {
  RM: {
    etiqueta: "Habitaciones promedio de la zona", unidad: "hab.",
    defecto: 6.2, min: 3, max: 9, paso: 0.1, decimales: 1,
    niveles: [[5, "Casas chicas"], [6.5, "Casas medianas"],
              [7.5, "Casas amplias"], [Infinity, "Casas muy grandes"]],
  },
  AGE: {
    etiqueta: "Viviendas anteriores a 1940", unidad: "%",
    defecto: 77.5, min: 0, max: 100, paso: 0.5, decimales: 0,
    niveles: [[30, "Barrio moderno"], [65, "Barrio mixto"],
              [Infinity, "Barrio antiguo"]],
  },
  CRIM: {
    etiqueta: "Delitos en la zona", unidad: "per cápita",
    defecto: 0.26, min: 0, max: 90, paso: 0.01, decimales: 2,
    potencia: 3,
    niveles: [[1, "Zona tranquila"], [4, "Seguridad media"],
              [Infinity, "Zona con muchos delitos"]],
  },
  ZN: {
    etiqueta: "Suelo en lotes grandes", unidad: "%",
    defecto: 0, min: 0, max: 100, paso: 1, decimales: 0,
    niveles: [[15, "Lotes chicos"], [50, "Algunos lotes grandes"],
              [Infinity, "Muchos lotes grandes"]],
  },
  INDUS: {
    etiqueta: "Suelo industrial", unidad: "%",
    defecto: 9.7, min: 0, max: 28, paso: 0.1, decimales: 1,
    niveles: [[4, "Casi sin industria"], [12, "Alguna fábrica"],
              [20, "Zona industrial"], [Infinity, "Muy industrial"]],
  },
  LSTAT: {
    etiqueta: "Población de bajos ingresos", unidad: "%",
    defecto: 11.4, min: 1, max: 38, paso: 0.1, decimales: 1,
    niveles: [[7, "Barrio acomodado"], [15, "Barrio de clase media"],
              [25, "Barrio humilde"], [Infinity, "Barrio muy vulnerable"]],
  },
  PTRATIO: {
    etiqueta: "Alumnos por docente", unidad: "alumnos",
    defecto: 19, min: 12, max: 22, paso: 0.1, decimales: 1,
    niveles: [[16, "Aulas con pocos alumnos"], [20, "Aulas medianas"],
              [Infinity, "Aulas llenas"]],
  },
  NOX: {
    etiqueta: "Contaminación del aire", unidad: "NOx",
    defecto: 0.538, min: 0.38, max: 0.88, paso: 0.001, decimales: 3,
    niveles: [[0.45, "Aire limpio"], [0.6, "Aire regular"],
              [Infinity, "Aire contaminado"]],
  },
  DIS: {
    etiqueta: "Distancia a los centros de empleo", unidad: "km aprox.",
    defecto: 3.2, min: 1, max: 12.5, paso: 0.1, decimales: 1,
    niveles: [[2.5, "Muy cerca del centro"], [5, "A distancia media"],
              [Infinity, "Lejos del empleo"]],
  },
  TAX: {
    etiqueta: "Impuesto inmobiliario", unidad: "USD cada 10.000",
    defecto: 330, min: 187, max: 711, paso: 1, decimales: 0,
    niveles: [[300, "Impuesto bajo"], [450, "Impuesto medio"],
              [Infinity, "Impuesto alto"]],
  },
};
// Cómo se agrupan los controles en la pantalla
const SECCIONES = [
  {titulo: "La propiedad", detalle: "Tamaño y antigüedad de las casas, y " +
    "si queda sobre el río.", campos: ["RM", "AGE", "CHAS"]},
  {titulo: "El barrio", detalle: "Quiénes viven, cómo se vive y qué hay " +
    "alrededor.", campos: ["CRIM", "LSTAT", "ZN", "PTRATIO", "INDUS"]},
  {titulo: "Entorno y accesibilidad", detalle: "Aire, distancia al " +
    "trabajo y a las autopistas.", campos: ["NOX", "DIS", "RAD", "TAX"]},
];
// RAD es un índice: solo existen estos valores en el dataset
const VALORES_RAD = [1, 2, 3, 4, 5, 6, 7, 8, 24];
const RAD_POR_DEFECTO = 5;
// Rango de precios de la escala de la ficha, en miles de USD: el del
// dataset (la variable objetivo va de 5 a 50)
const PRECIO_MIN = 5;
const PRECIO_MAX = 50;
// Cuánto espera el tasador desde el último cambio antes de preguntarle
// a la API. Así un slider que se arrastra no manda decenas de pedidos
const ESPERA_MS = 200;

// Lote de ejemplo (docker/casas_ejemplo.csv): la última casa viene sin
// RM, para mostrar que el pipeline lo imputa
const CSV_EJEMPLO = `CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,LSTAT
0.09299,0.0,25.65,0.0,0.581,5.961,92.9,2.0869,2.0,188.0,19.1,17.93
2.24236,0.0,19.58,0.0,0.605,5.854,91.8,2.422,5.0,403.0,14.7,11.64
9.91655,0.0,18.1,0.0,0.693,5.852,77.8,1.5004,24.0,666.0,20.2,29.97
10.233,0.0,18.1,0.0,0.614,6.185,96.7,2.1705,24.0,666.0,20.2,18.03
11.9511,0.0,18.1,0.0,0.659,,100.0,1.2852,24.0,666.0,20.2,12.13
`;

const $ = (selector) => document.querySelector(selector);

// Estado de la pantalla: el valor de cada columna y cuáles se marcaron
// como desconocidas. Los controles lo escriben y leerCasa() lo lee
const casaActual = {CHAS: 0, RAD: RAD_POR_DEFECTO};
const desconocidas = new Set();
for (const [campo, datos] of Object.entries(CAMPOS)) {
  casaActual[campo] = datos.defecto;
}
let ultimaExplicacion = null;
let pedidoVigente = 0;
let temporizador = null;
let precioMostrado = null;


/* ---------- Formato ---------- */

/** Pasa de miles de USD a un texto como "USD 23.400". */
function formatearUsd(miles, conSigno = false) {
  const valor = Math.round(miles * 1000);
  const texto = "USD " + Math.abs(valor).toLocaleString("es-AR");
  if (!conSigno) return valor < 0 ? "−" + texto : texto;
  return (valor < 0 ? "−" : "+") + texto;
}

/** Un número con coma decimal y la cantidad de decimales pedida. */
function formatearNumero(valor, decimales) {
  return valor.toLocaleString("es-AR", {
    minimumFractionDigits: decimales, maximumFractionDigits: decimales,
  });
}

/** Un número con un decimal y coma: "23,4". */
function formatearMiles(valor) {
  return formatearNumero(valor, 1);
}

/** La fecha "2026-10-21" del modelo como "21/10/2026". */
function formatearFecha(texto) {
  const [anio, mes, dia] = texto.split("-");
  return `${dia}/${mes}/${anio}`;
}

/** Una cantidad de miles de USD como "14 %" respecto de una base. */
function formatearPorcentaje(parte, base) {
  const porcentaje = Math.round(parte / base * 100);
  return `${porcentaje >= 0 ? "+" : "−"}${Math.abs(porcentaje)} %`;
}

/** La frase que describe el valor de un campo ("Zona tranquila"). */
function describirNivel(campo, valor) {
  return CAMPOS[campo].niveles.find(([limite]) => valor < limite)[1];
}


/* ---------- API ---------- */

/** Traduce el detalle de un error 422 de FastAPI a una frase simple. */
function explicarRechazo(detalle) {
  if (!Array.isArray(detalle)) return JSON.stringify(detalle);
  return detalle.slice(0, 3).map((problema) => {
    // loc es algo como ["body", 2, "RM"]: pedido, fila, columna
    const donde = problema.loc.filter((p) => p !== "body");
    const fila = donde.find((p) => typeof p === "number");
    const columna = donde.find((p) => typeof p === "string");
    const lugar = [fila === undefined ? "" : `fila ${fila + 1}`,
                   columna === undefined ? "" : `columna ${columna}`]
      .filter(Boolean).join(", ");
    return `${lugar ? lugar + ": " : ""}${problema.msg}`;
  }).join(" · ");
}

/**
 * Hace un pedido a la API y devuelve la respuesta decodificada.
 * Sin cuerpo hace un GET; con cuerpo, un POST con JSON. Si la API no
 * responde o devuelve un error, lanza una excepción con un mensaje
 * para mostrar.
 */
async function consultarApi(ruta, cuerpo) {
  let respuesta;
  try {
    respuesta = await fetch(API + ruta, cuerpo === undefined ? {} : {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(cuerpo),
    });
  } catch {
    throw new Error("No se pudo conectar con la API. ¿Está levantada?");
  }
  if (!respuesta.ok) {
    let detalle = "";
    try {
      detalle = explicarRechazo((await respuesta.json()).detail);
    } catch { /* la respuesta no era JSON; alcanza con el código */ }
    throw new Error(`La API respondió ${respuesta.status}. ${detalle}`);
  }
  return respuesta.json();
}

/** Muestra qué modelo hay en producción, de forma discreta. */
async function mostrarEstadoApi() {
  const estado = $("#estado-api");
  try {
    const info = await consultarApi("/modelo");
    const alpha = info.alpha === null ? "" : ` (α=${info.alpha})`;
    $("#estado-texto").textContent = "API en línea";
    estado.className = "estado-api ok";
    $("#modelo").textContent = `Estimación generada por el modelo ` +
      `${info.regresor}${alpha} con imputación ${info.imputador}, ` +
      `publicado el ${formatearFecha(info.actualizado)}.`;
  } catch {
    $("#estado-texto").textContent = "API sin conexión";
    estado.className = "estado-api error";
    $("#modelo").textContent = "No se pudo consultar el modelo publicado.";
  }
}


/* ---------- Controles ---------- */

/** Convierte el valor de un slider (índice) en el valor del campo. */
function valorDesdeIndice(campo, indice) {
  const {min, max, paso, decimales, potencia} = CAMPOS[campo];
  const pasos = pasosDelSlider(campo);
  const fraccion = Math.pow(indice / pasos, potencia || 1);
  const valor = potencia ? min + (max - min) * fraccion
    : min + indice * paso;
  return Number(valor.toFixed(Math.max(decimales, 2)));
}

/** El inverso: de un valor del campo al índice del slider. */
function indiceDesdeValor(campo, valor) {
  const {min, max, paso, potencia} = CAMPOS[campo];
  if (potencia) {
    return Math.round(pasosDelSlider(campo) *
      Math.pow((valor - min) / (max - min), 1 / potencia));
  }
  return Math.round((valor - min) / paso);
}

/** Cuántas posiciones tiene el slider de un campo. */
function pasosDelSlider(campo) {
  const {min, max, paso, potencia} = CAMPOS[campo];
  return potencia ? 300 : Math.round((max - min) / paso);
}

/** Actualiza el valor visible, el relleno del slider y la frase. */
function pintarControl(campo, desdeSlider = false) {
  const control = $(`.control[data-campo="${campo}"]`);
  const datos = CAMPOS[campo];
  const desconocido = desconocidas.has(campo);
  const valor = casaActual[campo];
  const rango = control.querySelector("input[type=range]");
  rango.disabled = desconocido;
  // Mientras se arrastra, el slider ya está donde corresponde: moverlo
  // de nuevo lo haría saltar por el redondeo del valor
  if (!desdeSlider) rango.value = indiceDesdeValor(campo, valor);
  rango.style.setProperty("--relleno",
    `${100 * rango.value / rango.max}%`);
  control.classList.toggle("sin-dato", desconocido);
  const texto = desconocido ? "Lo completa el modelo"
    : formatearNumero(valor, datos.decimales);
  control.querySelector("output").textContent = texto;
  control.querySelector(".unidad").textContent = desconocido ? ""
    : datos.unidad;
  control.querySelector(".nivel").textContent = desconocido
    ? "Se usa el valor que aprendió el modelo"
    : describirNivel(campo, valor);
  rango.setAttribute("aria-valuetext", desconocido ? "desconocido"
    : `${texto} ${datos.unidad}`);
}

/** Dibuja el control de un campo numérico: slider, valor y "No sé". */
function crearSlider(campo) {
  const datos = CAMPOS[campo];
  const div = document.createElement("div");
  div.className = "control";
  div.dataset.campo = campo;
  div.innerHTML = `
    <div class="control-cabecera">
      <label for="campo-${campo}">${datos.etiqueta}</label>
      <span class="control-valor"><output for="campo-${campo}"></output>
        <span class="unidad"></span></span>
    </div>
    <input type="range" id="campo-${campo}" name="${campo}" min="0"
           max="${pasosDelSlider(campo)}" step="1">
    <div class="control-pie">
      <span class="nivel"></span>
      <label class="nose"><input type="checkbox" name="nose-${campo}">
        <span>No sé</span></label>
    </div>`;
  const rango = div.querySelector("input[type=range]");
  rango.addEventListener("input", () => {
    casaActual[campo] = valorDesdeIndice(campo, Number(rango.value));
    pintarControl(campo, true);
    cambioLaCasa();
  });
  div.querySelector(".nose input").addEventListener("change", (evento) => {
    if (evento.target.checked) desconocidas.add(campo);
    else desconocidas.delete(campo);
    pintarControl(campo);
    cambioLaCasa();
  });
  return div;
}

/** El interruptor "Cerca del río Charles". */
function crearRio() {
  const div = document.createElement("div");
  div.className = "control control-rio";
  div.innerHTML = `
    <label class="interruptor">
      <input type="checkbox" name="CHAS" id="campo-CHAS">
      <span class="pista" aria-hidden="true"></span>
      <span class="interruptor-texto">
        <strong>Cerca del río Charles</strong>
        <small>La propiedad limita con el río: aparece en el dibujo</small>
      </span>
    </label>`;
  div.querySelector("input").addEventListener("change", (evento) => {
    casaActual.CHAS = evento.target.checked ? 1 : 0;
    cambioLaCasa();
  });
  return div;
}

/** El selector de acceso a autopistas: un chip por valor del índice. */
function crearAutopistas() {
  const grupo = document.createElement("fieldset");
  grupo.className = "control control-rad";
  const chips = VALORES_RAD.map((v) => `
    <label class="opcion">
      <input type="radio" name="RAD" value="${v}"
             ${v === RAD_POR_DEFECTO ? "checked" : ""}>
      <span>${v}</span>
    </label>`).join("");
  grupo.innerHTML = `
    <legend>Acceso a autopistas</legend>
    <div class="opciones">${chips}</div>
    <div class="control-pie">
      <span>1 = poco acceso</span><span>24 = autopista a la puerta</span>
    </div>`;
  grupo.addEventListener("change", (evento) => {
    casaActual.RAD = Number(evento.target.value);
    cambioLaCasa();
  });
  return grupo;
}

/** Arma las secciones con todos sus controles. */
function armarFormulario() {
  const formulario = $("#formulario");
  for (const seccion of SECCIONES) {
    const tarjeta = document.createElement("section");
    tarjeta.className = "tarjeta seccion";
    tarjeta.innerHTML = `<h2>${seccion.titulo}</h2>
      <p class="ayuda">${seccion.detalle}</p>`;
    for (const campo of seccion.campos) {
      tarjeta.append(campo === "CHAS" ? crearRio()
        : campo === "RAD" ? crearAutopistas() : crearSlider(campo));
    }
    formulario.append(tarjeta);
  }
  for (const campo of Object.keys(CAMPOS)) pintarControl(campo);
}

/**
 * Devuelve la casa como la espera la API: un diccionario columna ->
 * número, con null en las desconocidas.
 */
function leerCasa() {
  const casa = {};
  for (const campo of COLUMNAS) {
    casa[campo] = desconocidas.has(campo) ? null : casaActual[campo];
  }
  return casa;
}

/** Vuelve a poner cada control en su valor por defecto. */
function restablecerFormulario() {
  desconocidas.clear();
  for (const [campo, datos] of Object.entries(CAMPOS)) {
    casaActual[campo] = datos.defecto;
  }
  casaActual.CHAS = 0;
  casaActual.RAD = RAD_POR_DEFECTO;
  $("#campo-CHAS").checked = false;
  $(`input[name="RAD"][value="${RAD_POR_DEFECTO}"]`).checked = true;
  for (const marca of document.querySelectorAll(".nose input")) {
    marca.checked = false;
  }
  for (const campo of Object.keys(CAMPOS)) pintarControl(campo);
  cambioLaCasa(0);
}

/**
 * Se llama en cada cambio de un control: el dibujo se actualiza al
 * instante (no necesita la API) y la tasación espera un instante, por
 * si el usuario sigue moviendo el slider.
 */
function cambioLaCasa(espera = ESPERA_MS) {
  actualizarDibujo();
  clearTimeout(temporizador);
  temporizador = setTimeout(tasar, espera);
}


/* ---------- Ilustración y datos clave ---------- */

/** Pasa la casa actual al dibujo y a los chips que lo acompañan. */
function actualizarDibujo() {
  // Lo desconocido se dibuja con el valor típico: no hay otro mejor
  const casa = {...casaActual};
  for (const campo of desconocidas) casa[campo] = CAMPOS[campo].defecto;
  actualizarIlustracion($("#ilustracion"), casa);

  const dato = (campo, texto) => desconocidas.has(campo)
    ? `<li class="desconocido">${NOMBRES[campo]}: sin dato</li>`
    : `<li>${texto}</li>`;
  const chips = [
    dato("RM", `<strong>${formatearNumero(casa.RM, 1)}</strong> ` +
      "habitaciones"),
    dato("AGE", `<strong>${formatearNumero(casa.AGE, 0)} %</strong> ` +
      "anteriores a 1940"),
    dato("CRIM", describirNivel("CRIM", casa.CRIM)),
    dato("NOX", describirNivel("NOX", casa.NOX)),
    dato("LSTAT", describirNivel("LSTAT", casa.LSTAT)),
  ];
  if (casa.CHAS >= 0.5) chips.push("<li class=\"destacado\">Sobre el río " +
    "Charles</li>");
  if (casa.RAD >= 8) chips.push("<li>Autopista cerca</li>");
  $("#datos-clave").innerHTML = `<ul>${chips.join("")}</ul>`;
}


/* ---------- Resultado ---------- */

/** Anima el precio grande de un valor al siguiente, sin pasarse. */
function animarPrecio(destino) {
  const elemento = $("#precio");
  const sinMovimiento = window.matchMedia(
    "(prefers-reduced-motion: reduce)").matches;
  if (precioMostrado === null || sinMovimiento) {
    elemento.textContent = formatearUsd(destino);
    precioMostrado = destino;
    return;
  }
  const origen = precioMostrado;
  const inicio = performance.now();
  const duracion = 350;
  precioMostrado = destino;
  const paso = (ahora) => {
    // Si llegó otra tasación mientras tanto, esa se encarga
    if (precioMostrado !== destino) return;
    const avance = Math.min(1, (ahora - inicio) / duracion);
    const suave = 1 - Math.pow(1 - avance, 3);
    elemento.textContent = formatearUsd(origen + (destino - origen) * suave);
    if (avance < 1) requestAnimationFrame(paso);
  };
  requestAnimationFrame(paso);
}

/** Posición en la escala de la ficha, como porcentaje del ancho. */
function posicionEnEscala(miles) {
  return `${100 * limitar((miles - PRECIO_MIN) / (PRECIO_MAX - PRECIO_MIN),
    0, 1)}%`;
}

/** Un texto con los datos que más suman y más restan. */
function dibujarResumen(filas) {
  const suma = filas.filter((f) => f.aporte > 0)
    .sort((a, b) => b.aporte - a.aporte)[0];
  const resta = filas.filter((f) => f.aporte < 0)
    .sort((a, b) => a.aporte - b.aporte)[0];
  const partes = [];
  if (suma) partes.push(`Lo que más suma: <strong>${NOMBRES[suma.variable]}` +
    `</strong> <span class="sube">${formatearUsd(suma.aporte, true)}</span>`);
  if (resta) partes.push(`Lo que más resta: <strong>` +
    `${NOMBRES[resta.variable]}</strong> <span class="baja">` +
    `${formatearUsd(resta.aporte, true)}</span>`);
  $("#resumen").innerHTML = partes.join("<br>");
}

/** El texto que dice qué valor tiene una variable en esta casa. */
function describirValor(campo) {
  if (desconocidas.has(campo)) return "Dato estimado por el modelo";
  if (campo === "CHAS") return casaActual.CHAS ? "Cerca del río" : "Sin río";
  if (campo === "RAD") return `Índice de acceso ${casaActual.RAD}`;
  const datos = CAMPOS[campo];
  return `${formatearNumero(casaActual[campo], datos.decimales)} ` +
    `${datos.unidad} · ${describirNivel(campo, casaActual[campo])}`;
}

/** Una lista de variables con su aporte y una barra proporcional. */
function listaDeAportes(titulo, clase, filas, maximo) {
  if (filas.length === 0) return "";
  const items = filas.map((fila) => `
    <li class="aporte ${clase}">
      <div class="aporte-texto">
        <span class="aporte-nombre">${NOMBRES[fila.variable]}</span>
        <span class="aporte-valor">${formatearUsd(fila.aporte, true)}</span>
      </div>
      <div class="aporte-detalle">${describirValor(fila.variable)}</div>
      <div class="aporte-barra"><span style="width:${
        Math.max(2, 100 * Math.abs(fila.aporte) / maximo)}%"></span></div>
    </li>`).join("");
  return `<div class="aportes-grupo"><h3 class="${clase}">${titulo}</h3>
    <ul>${items}</ul></div>`;
}

/** Dibuja "Qué suma" y "Qué resta", ordenadas por magnitud. */
function dibujarAportes(contribuciones) {
  const filas = Object.entries(contribuciones)
    .map(([variable, aporte]) => ({variable, aporte}));
  const maximo = Math.max(...filas.map((f) => Math.abs(f.aporte))) || 1;
  // Menos de 100 USD no se ve ni se cuenta: se agrupa aparte
  const sumas = filas.filter((f) => f.aporte >= 0.1)
    .sort((a, b) => b.aporte - a.aporte);
  const restas = filas.filter((f) => f.aporte <= -0.1)
    .sort((a, b) => a.aporte - b.aporte);
  const neutras = filas.filter((f) => Math.abs(f.aporte) < 0.1);
  let html = listaDeAportes("Suma al precio", "sube", sumas, maximo) +
    listaDeAportes("Resta al precio", "baja", restas, maximo);
  if (neutras.length > 0) {
    html += `<p class="nota">Casi no influyen: ` +
      `${neutras.map((f) => NOMBRES[f.variable]).join(", ")}.</p>`;
  }
  $("#aportes").innerHTML = html;
  dibujarResumen(filas);
}

/** Manda la casa actual a /explicar y muestra el resultado. */
async function tasar() {
  const resultado = $(".estimacion");
  const error = $("#error");
  const numeroDePedido = ++pedidoVigente;
  error.classList.add("oculto");
  resultado.classList.add("cargando");
  const casa = leerCasa();
  try {
    const explicacion = await consultarApi("/explicar", casa);
    // Si el usuario siguió moviendo controles, esta respuesta ya quedó vieja
    if (numeroDePedido !== pedidoVigente) return;
    ultimaExplicacion = explicacion;
    const precio = explicacion.precio_estimado;
    const delta = precio - explicacion.intercepto;
    const clase = delta >= 0 ? "sube" : "baja";
    animarPrecio(precio);
    ponerPrecioCartel($("#ilustracion"), formatearUsd(precio));
    $("#contexto").innerHTML = `<span class="${clase}">` +
      `${formatearUsd(delta, true)} ` +
      `(${formatearPorcentaje(delta, explicacion.intercepto)})</span> ` +
      `respecto de una casa promedio, que vale ` +
      `${formatearUsd(explicacion.intercepto)}`;
    $("#marca-promedio").style.left = posicionEnEscala(
      explicacion.intercepto);
    $("#marca-casa").style.left = posicionEnEscala(precio);
    $("#imputados").classList.toggle("oculto", desconocidas.size === 0);
    $("#precio-base").textContent = formatearUsd(explicacion.intercepto);
    dibujarAportes(explicacion.contribuciones);
  } catch (excepcion) {
    if (numeroDePedido !== pedidoVigente) return;
    error.textContent = excepcion.message;
    error.classList.remove("oculto");
  } finally {
    if (numeroDePedido === pedidoVigente) {
      resultado.classList.remove("cargando");
    }
  }
}


/* ---------- Tasar un lote ---------- */

/**
 * Separa un CSV simple (sin comillas ni comas dentro de las celdas) en
 * encabezado y filas. Las celdas vacías quedan como null, que la API
 * imputa. Si una celda no es un número, lanza un error que dice dónde:
 * sin esto, JSON.stringify convertiría el NaN en null y el modelo
 * imputaría el dato sin avisar.
 */
function leerCsv(texto) {
  const lineas = texto.trim().split(/\r?\n/).filter((l) => l.trim() !== "");
  if (lineas.length < 2) {
    throw new Error("El CSV está vacío: tiene que traer el encabezado y " +
      "al menos una casa.");
  }
  const encabezado = lineas[0].split(",").map((c) => c.trim());
  const faltantes = COLUMNAS.filter((c) => !encabezado.includes(c));
  if (faltantes.length > 0) {
    throw new Error(`Al CSV le faltan las columnas: ${faltantes.join(", ")}.` +
      ` Tiene que tener ${COLUMNAS.join(", ")}.`);
  }
  const filas = lineas.slice(1).map((linea, i) => {
    const celdas = linea.split(",");
    const fila = {};
    encabezado.forEach((columna, j) => {
      const celda = (celdas[j] ?? "").trim();
      const numero = celda === "" ? null : Number(celda);
      if (Number.isNaN(numero)) {
        throw new Error(`Fila ${i + 1}, columna ${columna}: "${celda}" no ` +
          "es un número.");
      }
      fila[columna] = numero;
    });
    return fila;
  });
  return {encabezado, filas};
}

/** Arma un CSV con las filas tasadas, para descargarlo. */
function armarCsv(filas, precios) {
  const columnas = [...COLUMNAS, "precio_estimado"];
  const lineas = filas.map((fila, i) => [
    ...COLUMNAS.map((c) => fila[c] === null ? "" : fila[c]), precios[i],
  ].join(","));
  return [columnas.join(","), ...lineas].join("\n") + "\n";
}

/** Dispara la descarga de un texto como archivo. */
function descargar(nombre, texto) {
  const enlace = document.createElement("a");
  enlace.href = URL.createObjectURL(new Blob([texto], {type: "text/csv"}));
  enlace.download = nombre;
  enlace.click();
  URL.revokeObjectURL(enlace.href);
}

/** Dibuja la tabla con las casas y su precio estimado. */
function dibujarTabla(filas, precios) {
  const encabezado = [...COLUMNAS, "precio_estimado"].map(
    (c) => `<th>${c}</th>`).join("");
  const cuerpo = filas.map((fila, i) => {
    const celdas = COLUMNAS.map((c) => fila[c] === null
      ? '<td class="imputado">imputado</td>' : `<td>${fila[c]}</td>`);
    return `<tr>${celdas.join("")}<td class="precio-lote">` +
      `${formatearMiles(precios[i])}</td></tr>`;
  }).join("");
  $("#tabla-lote").innerHTML =
    `<table><thead><tr>${encabezado}</tr></thead>` +
    `<tbody>${cuerpo}</tbody></table>`;
}

/** Lee el CSV elegido, lo manda a /predecir-lote y muestra la tabla. */
async function tasarLote(archivo) {
  const mensaje = $("#mensaje-lote");
  const descargarBoton = $("#descargar");
  $("#nombre-archivo").textContent = archivo.name;
  $("#tabla-lote").innerHTML = "";
  descargarBoton.classList.add("oculto");
  let filas;
  try {
    ({filas} = leerCsv(await archivo.text()));
  } catch (excepcion) {
    mensaje.textContent = excepcion.message;
    mensaje.className = "mensaje error";
    return;
  }
  // Solo las columnas del modelo, en su orden: lo demás la API lo rechaza
  const casas = filas.map((fila) => Object.fromEntries(
    COLUMNAS.map((c) => [c, fila[c]])));
  try {
    const respuesta = await consultarApi("/predecir-lote", casas);
    const precios = respuesta.precios_estimados;
    mensaje.textContent = `${casas.length} casas tasadas. precio_estimado ` +
      "está en miles de USD.";
    mensaje.className = "mensaje ok";
    dibujarTabla(casas, precios);
    descargarBoton.classList.remove("oculto");
    descargarBoton.onclick = () =>
      descargar("precios.csv", armarCsv(casas, precios));
  } catch (excepcion) {
    mensaje.textContent = excepcion.message;
    mensaje.className = "mensaje error";
  }
}


/* ---------- Arranque ---------- */

/** Cambia de pestaña: una propiedad o un lote. */
function activarPestanias() {
  for (const boton of document.querySelectorAll(".pestania")) {
    boton.addEventListener("click", () => {
      for (const otro of document.querySelectorAll(".pestania")) {
        otro.classList.toggle("activa", otro === boton);
        otro.setAttribute("aria-selected", otro === boton);
      }
      for (const panel of document.querySelectorAll(".panel")) {
        panel.classList.toggle("activa", panel.id === boton.dataset.panel);
      }
    });
  }
}

crearEscena($("#ilustracion"));
armarFormulario();
activarPestanias();
mostrarEstadoApi();
$("#formulario").addEventListener("submit", (evento) => {
  evento.preventDefault();
});
$("#restablecer").addEventListener("click", restablecerFormulario);
$("#archivo").addEventListener("change", (evento) => {
  if (evento.target.files[0]) tasarLote(evento.target.files[0]);
  // Vaciar el input deja volver a elegir el mismo archivo después de
  // corregirlo
  evento.target.value = "";
});
$("#ejemplo").addEventListener("click", () =>
  descargar("casas_ejemplo.csv", CSV_EJEMPLO));
// En la primera carga los controles valen sus medianas, así que ya hay
// una casa para dibujar y tasar
actualizarDibujo();
tasar();
