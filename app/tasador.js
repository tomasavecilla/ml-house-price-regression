/* Lógica del tasador: arma el formulario, llama a la API y dibuja el
   resultado. No hay modelo acá: cada tasación es un pedido HTTP.

   Las rutas empiezan con /api/: el navegador habla siempre con el mismo
   origen que sirvió la página, y nginx (nginx.conf) reenvía esos pedidos
   al servicio api:8000. Así no hace falta CORS ni saber dónde está la
   API desde el navegador. */

"use strict";

const API = "/api";

// Columnas del modelo, en el orden del CSV de entrenamiento
const COLUMNAS = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
                  "RAD", "TAX", "PTRATIO", "LSTAT"];
// Nombre corto de cada variable, para las etiquetas y el gráfico
const NOMBRES = {
  RM: "Habitaciones", AGE: "Antigüedad", CRIM: "Criminalidad",
  ZN: "Lotes grandes", INDUS: "Industria", LSTAT: "Bajos ingresos",
  PTRATIO: "Alumnos/docente", NOX: "Contaminación",
  DIS: "Dist. a empleo", TAX: "Impuesto", CHAS: "Río Charles",
  RAD: "Autopistas",
};
// Descripción, valor por defecto, mínimo, máximo y paso de cada campo
// numérico. El valor por defecto es la mediana de las casas válidas del
// dataset (entrenar.leer_dataset()); los límites, su mínimo y máximo.
// Están escritos acá para que la app no dependa del código de
// entrenamiento
const CAMPOS = {
  RM: ["Cantidad promedio de habitaciones", 6.2, 3, 9, 0.1],
  AGE: ["Porcentaje de viviendas construidas antes de 1940", 77.5, 0,
        100, 1],
  CRIM: ["Tasa de criminalidad per cápita de la zona", 0.26, 0, 90, 0.05],
  ZN: ["Porcentaje de suelo residencial en lotes grandes", 0, 0, 100, 1],
  INDUS: ["Porcentaje de suelo industrial", 9.7, 0, 28, 0.1],
  LSTAT: ["Porcentaje de población de bajos ingresos", 11.4, 1, 38, 0.1],
  PTRATIO: ["Alumnos por docente en la zona", 19, 12, 22, 0.1],
  NOX: ["Óxidos de nitrógeno, en partes por 10 millones", 0.538, 0.38,
        0.88, 0.001],
  DIS: ["Distancia ponderada a cinco centros de empleo", 3.2, 1, 12.5,
        0.1],
  TAX: ["Impuesto inmobiliario cada 10.000 USD", 330, 187, 711, 1],
};
// RAD es un índice: solo existen estos valores en el dataset
const VALORES_RAD = [1, 2, 3, 4, 5, 6, 7, 8, 24];
const RAD_POR_DEFECTO = 5;
// Cómo se agrupan los campos en el formulario
const SECCIONES = {
  "Vivienda": ["RM", "AGE", "CHAS"],
  "Zona": ["CRIM", "ZN", "INDUS", "LSTAT", "PTRATIO"],
  "Entorno y accesibilidad": ["NOX", "DIS", "TAX", "RAD"],
};
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


/* ---------- Formato ---------- */

/** Pasa de miles de USD a un texto como "USD 23.400". */
function formatearUsd(miles) {
  return "USD " + Math.round(miles * 1000).toLocaleString("es-AR");
}

/** Un número con un decimal y coma, con signo si se pide: "+2,3". */
function formatearMiles(valor, conSigno = false) {
  // Se redondea antes de mirar el signo, para que -0,04 no salga "+-0,0"
  const redondeado = Math.round(valor * 10) / 10 || 0;
  const texto = redondeado.toLocaleString("es-AR", {
    minimumFractionDigits: 1, maximumFractionDigits: 1,
  });
  return conSigno && redondeado >= 0 ? "+" + texto : texto;
}


/* ---------- API ---------- */

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
      detalle = JSON.stringify((await respuesta.json()).detail);
    } catch { /* la respuesta no era JSON; alcanza con el código */ }
    throw new Error(`La API respondió ${respuesta.status}. ${detalle}`);
  }
  return respuesta.json();
}

/** Muestra en el encabezado qué modelo hay en producción. */
async function mostrarEstadoApi() {
  const pildora = $("#estado-api");
  try {
    const info = await consultarApi("/modelo");
    const alpha = info.alpha === null ? "" : ` α=${info.alpha}`;
    pildora.textContent = `● API en línea · ${info.regresor}${alpha} · ` +
      `${info.imputador} · modelo del ${info.actualizado}`;
    pildora.className = "pildora ok";
  } catch {
    pildora.textContent = "● API sin conexión";
    pildora.className = "pildora error";
  }
}


/* ---------- Formulario ---------- */

/** Dibuja el control de un campo y lo devuelve como elemento. */
function crearCampo(campo) {
  const div = document.createElement("div");
  div.className = "campo";
  const etiqueta = `${NOMBRES[campo]} <code>${campo}</code>`;
  if (campo === "CHAS") {
    div.innerHTML = `
      <label title="Limita con el río Charles">${etiqueta}</label>
      <label class="interruptor">
        <input type="checkbox" name="CHAS"><span class="pista"></span>
        <span>Limita con el río</span>
      </label>`;
    return div;
  }
  if (campo === "RAD") {
    const opciones = VALORES_RAD.map((v) => `<option value="${v}"` +
      `${v === RAD_POR_DEFECTO ? " selected" : ""}>${v}</option>`);
    div.innerHTML = `
      <label for="campo-RAD" title="Índice de acceso a autopistas radiales">
        ${etiqueta}</label>
      <select id="campo-RAD" name="RAD">${opciones.join("")}</select>`;
    return div;
  }
  const [ayuda, valor, minimo, maximo, paso] = CAMPOS[campo];
  div.innerHTML = `
    <label for="campo-${campo}" title="${ayuda}">${etiqueta}</label>
    <input type="number" id="campo-${campo}" name="${campo}"
           value="${valor}" min="${minimo}" max="${maximo}" step="${paso}"
           required>`;
  return div;
}

/** Arma las secciones del formulario y los chips de desconocidos. */
function armarFormulario() {
  const secciones = $("#secciones");
  for (const [titulo, campos] of Object.entries(SECCIONES)) {
    const eyebrow = document.createElement("p");
    eyebrow.className = "eyebrow";
    eyebrow.textContent = titulo;
    const grilla = document.createElement("div");
    grilla.className = "campos";
    campos.forEach((campo) => grilla.append(crearCampo(campo)));
    secciones.append(eyebrow, grilla);
  }
  const chips = $("#desconocidos");
  for (const campo of Object.keys(CAMPOS)) {
    const chip = document.createElement("label");
    chip.className = "chip";
    chip.innerHTML = `<input type="checkbox" value="${campo}">` +
      `${NOMBRES[campo]}`;
    // Un dato desconocido se ve tachado en el formulario
    chip.querySelector("input").addEventListener("change", (evento) => {
      $(`#campo-${campo}`).disabled = evento.target.checked;
    });
    chips.append(chip);
  }
}

/**
 * Lee el formulario y devuelve la casa como la espera la API: un
 * diccionario columna -> número, con null en las desconocidas.
 */
function leerCasa() {
  const casa = {};
  for (const campo of COLUMNAS) {
    const control = $(`[name="${campo}"]`);
    if (campo === "CHAS") {
      casa[campo] = control.checked ? 1 : 0;
    } else if (control.disabled) {
      casa[campo] = null;
    } else {
      casa[campo] = Number(control.value);
    }
  }
  return casa;
}

/** Vuelve a poner cada campo en su valor por defecto. */
function restablecerFormulario() {
  $("#formulario").reset();
  for (const chip of document.querySelectorAll("#desconocidos input")) {
    chip.checked = false;
    $(`#campo-${chip.value}`).disabled = false;
  }
  tasar();
}


/* ---------- Gráfico de aportes ---------- */

const ALTO_FILA = 30;
const ALTO_BARRA = 16;
const ANCHO = 560;
const MARGEN_IZQ = 215;
const MARGEN_DER = 20;
const MARGEN_SUP = 6;
const MARGEN_INF = 34;

/**
 * Dibuja en SVG las barras de cuánto aporta cada variable, ordenadas
 * por magnitud: teal las que suben el precio, naranja las que lo
 * bajan. El eje es simétrico para que el cero quede en el medio.
 */
function dibujarContribuciones(contribuciones) {
  const filas = Object.entries(contribuciones)
    .map(([variable, aporte]) => ({variable, aporte}))
    .sort((a, b) => Math.abs(b.aporte) - Math.abs(a.aporte));
  // Un 40 % de aire a los costados, para que las cifras entren
  const limite = Math.max(...filas.map((f) => Math.abs(f.aporte))) * 1.4
    || 1;
  const anchoUtil = ANCHO - MARGEN_IZQ - MARGEN_DER;
  const centro = MARGEN_IZQ + anchoUtil / 2;
  const escala = (valor) => centro + (valor / limite) * (anchoUtil / 2);
  const alto = MARGEN_SUP + ALTO_FILA * filas.length + MARGEN_INF;
  const abajo = MARGEN_SUP + ALTO_FILA * filas.length;

  // Grilla: cinco marcas redondas, a izquierda y derecha del cero
  const paso = pasoRedondo(limite);
  let grilla = "";
  for (let v = -paso * 2; v <= paso * 2 + 1e-9; v += paso) {
    if (Math.abs(v) > limite) continue;
    const x = escala(v);
    grilla += `<line class="grilla" x1="${x}" y1="${MARGEN_SUP}" x2="${x}"
      y2="${abajo}"></line>
      <text class="eje" x="${x}" y="${abajo + 14}" text-anchor="middle">
      ${formatearMiles(v, v !== 0)}</text>`;
  }

  const barras = filas.map((fila, i) => {
    const y = MARGEN_SUP + ALTO_FILA * i;
    const yBarra = y + (ALTO_FILA - ALTO_BARRA) / 2;
    const positiva = fila.aporte >= 0;
    const x0 = Math.min(centro, escala(fila.aporte));
    const ancho = Math.abs(escala(fila.aporte) - centro);
    const clase = positiva ? "sube" : "baja";
    // Esquinas redondeadas solo del lado que crece la barra; las
    // barras más angostas que el radio van rectas
    let camino;
    if (ancho <= 4) {
      camino = `M${x0},${yBarra} h${ancho} v${ALTO_BARRA} h${-ancho} z`;
    } else if (positiva) {
      camino = `M${x0},${yBarra} h${ancho - 4} a4,4 0 0 1 4,4
        v${ALTO_BARRA - 8} a4,4 0 0 1 -4,4 h${-(ancho - 4)} z`;
    } else {
      camino = `M${x0 + ancho},${yBarra} h${-(ancho - 4)}
        a4,4 0 0 0 -4,4 v${ALTO_BARRA - 8} a4,4 0 0 0 4,4 h${ancho - 4} z`;
    }
    const xValor = positiva ? x0 + ancho + 6 : x0 - 6;
    return `<g class="fila" data-variable="${fila.variable}"
        data-aporte="${fila.aporte}">
      <rect x="0" y="${y}" width="${ANCHO}" height="${ALTO_FILA}"
            fill="transparent"></rect>
      <text class="nombre" x="${MARGEN_IZQ - 12}" y="${y + ALTO_FILA / 2}"
            text-anchor="end" dominant-baseline="middle">
        ${NOMBRES[fila.variable]} · ${fila.variable}</text>
      <path class="barra ${clase}" d="${camino}"></path>
      <text class="valor" x="${xValor}" y="${y + ALTO_FILA / 2}"
            text-anchor="${positiva ? "start" : "end"}"
            dominant-baseline="middle">
        ${formatearMiles(fila.aporte, true)}</text>
    </g>`;
  });

  $("#grafico").innerHTML = `
    <svg viewBox="0 0 ${ANCHO} ${alto}" role="img"
         aria-label="Aporte de cada variable al precio, en miles de USD">
      ${grilla}
      <line class="cero" x1="${centro}" y1="${MARGEN_SUP}" x2="${centro}"
            y2="${abajo}"></line>
      ${barras.join("")}
      <text class="eje" x="${ANCHO - MARGEN_DER}" y="${alto - 2}"
            text-anchor="end">miles de USD</text>
    </svg>
    <div class="globo"></div>`;
  activarGlobo();
}

/** Un paso "redondo" (1, 2, 5, 10, ...) para unas dos marcas por lado. */
function pasoRedondo(limite) {
  const crudo = limite / 2;
  const potencia = Math.pow(10, Math.floor(Math.log10(crudo)));
  const resto = crudo / potencia;
  const factor = resto >= 5 ? 5 : resto >= 2 ? 2 : 1;
  return factor * potencia;
}

/** Muestra un globo con la cifra exacta al pasar el mouse por una fila. */
function activarGlobo() {
  const grafico = $("#grafico");
  const globo = grafico.querySelector(".globo");
  for (const fila of grafico.querySelectorAll(".fila")) {
    fila.addEventListener("mousemove", (evento) => {
      const caja = grafico.getBoundingClientRect();
      const aporte = Number(fila.dataset.aporte);
      globo.textContent = `${NOMBRES[fila.dataset.variable]}: ` +
        `${formatearMiles(aporte, true)} miles de USD`;
      globo.style.left = `${evento.clientX - caja.left}px`;
      globo.style.top = `${evento.clientY - caja.top}px`;
      globo.classList.add("visible");
    });
    fila.addEventListener("mouseleave", () => {
      globo.classList.remove("visible");
    });
  }
}


/* ---------- Tasar una casa ---------- */

/** Manda la casa del formulario a /explicar y muestra el resultado. */
async function tasar() {
  const resultado = $(".resultado");
  const boton = $("#formulario button[type=submit]");
  const error = $("#error");
  error.classList.add("oculto");
  resultado.classList.add("cargando");
  boton.disabled = true;
  const casa = leerCasa();
  try {
    const explicacion = await consultarApi("/explicar", casa);
    const precio = explicacion.precio_estimado;
    const delta = precio - explicacion.intercepto;
    const clase = delta >= 0 ? "sube" : "baja";
    $("#precio").textContent = formatearUsd(precio);
    $("#detalle").innerHTML = `${formatearMiles(precio)} miles de USD · ` +
      `<span class="${clase}">${formatearMiles(delta, true)}</span> ` +
      "respecto de una casa promedio";
    $("#imputados").classList.toggle(
      "oculto", !Object.values(casa).some((v) => v === null));
    dibujarContribuciones(explicacion.contribuciones);
    $("#explicacion").textContent = "Se parte del precio de una casa " +
      `promedio, ${formatearUsd(explicacion.intercepto)}, y cada ` +
      "variable lo empuja hacia arriba (teal) o hacia abajo (naranja). " +
      "Las barras suman exactamente el precio porque el modelo es lineal.";
  } catch (excepcion) {
    error.textContent = excepcion.message;
    error.classList.remove("oculto");
  } finally {
    resultado.classList.remove("cargando");
    boton.disabled = false;
  }
}


/* ---------- Tasar un lote ---------- */

/**
 * Separa un CSV simple (sin comillas ni comas dentro de las celdas) en
 * encabezado y filas. Las celdas vacías quedan como null, que la API
 * imputa.
 */
function leerCsv(texto) {
  const lineas = texto.trim().split(/\r?\n/);
  const encabezado = lineas[0].split(",").map((c) => c.trim());
  const filas = lineas.slice(1).map((linea) => {
    const celdas = linea.split(",");
    const fila = {};
    encabezado.forEach((columna, i) => {
      const celda = (celdas[i] ?? "").trim();
      fila[columna] = celda === "" ? null : Number(celda);
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
  const {encabezado, filas} = leerCsv(await archivo.text());
  const faltantes = COLUMNAS.filter((c) => !encabezado.includes(c));
  if (faltantes.length > 0) {
    mensaje.textContent = `Al CSV le faltan las columnas: ` +
      `${faltantes.join(", ")}. Tiene que tener ${COLUMNAS.join(", ")}.`;
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

/** Cambia de pestaña: una casa o un lote. */
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

armarFormulario();
activarPestanias();
mostrarEstadoApi();
$("#formulario").addEventListener("submit", (evento) => {
  evento.preventDefault();
  tasar();
});
$("#restablecer").addEventListener("click", restablecerFormulario);
$("#archivo").addEventListener("change", (evento) => {
  if (evento.target.files[0]) tasarLote(evento.target.files[0]);
});
$("#ejemplo").addEventListener("click", () =>
  descargar("casas_ejemplo.csv", CSV_EJEMPLO));
// En la primera carga los campos valen sus medianas, así que ya hay un
// resultado para mostrar
tasar();
