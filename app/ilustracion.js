/* Ilustración de la casa: un SVG dibujado a mano que cambia con los
   datos que carga el tasador. Es solo decoración que ayuda a leer el
   modelo: no calcula ningún precio.

   Cómo se usa cada variable (ver tambien la leyenda en tasador.js):
     RM      tamaño de la casa, cantidad de ventanas y de pisos
     AGE     estilo: moderna (techo bajo, vidrio) o antigua (ladrillo,
             techo empinado, chimenea, postigos)
     CHAS    aparece el río Charles al frente (si no, la calle)
     NOX     bruma en el cielo y humo más denso en las fábricas
     INDUS   cuántas fábricas hay en el fondo
     LSTAT   estado de las casas vecinas y del pasto
     CRIM    cerco de rejas, rejas en las ventanas y farola apagada
     ZN      cuántos árboles hay: más lotes grandes, más verde
     DIS     el centro de la ciudad se ve más o menos cerca
     RAD     una autopista con autos en el fondo
   TAX y PTRATIO no tienen dibujo propio: solo cuentan en el precio.

   La escena se arma una sola vez (crearEscena) y después solo se
   cambian atributos y estilos. Así las animaciones (humo, nubes, río,
   autos) no se reinician cada vez que se mueve un control. La casa
   principal sí se vuelve a dibujar entera, porque su forma cambia. */

"use strict";

const CENTRO = 480;       // la casa está centrada en el ancho del dibujo
const SUELO_CASA = 420;   // altura (y) donde apoya la casa
const ALTO_PISO = 86;

/** Limita un número al intervalo [minimo, maximo]. */
function limitar(valor, minimo, maximo) {
  return Math.min(maximo, Math.max(minimo, valor));
}

/** Mezcla dos colores #rrggbb: t=0 da el primero, t=1 el segundo. */
function mezclar(colorA, colorB, t) {
  const canal = (color, i) => parseInt(color.slice(1 + 2 * i, 3 + 2 * i), 16);
  const partes = [0, 1, 2].map((i) => Math.round(
    canal(colorA, i) + (canal(colorB, i) - canal(colorA, i)) * t));
  return "#" + partes.map((p) => p.toString(16).padStart(2, "0")).join("");
}


/* ---------- Piezas fijas de la escena ---------- */

/** Un árbol: tronco y tres copas, con x, y del pie y escala. */
function arbol(x, y, escala) {
  return `<g transform="translate(${x} ${y}) scale(${escala})">
    <rect x="-5" y="-42" width="10" height="42" fill="#7a5638"/>
    <circle cx="0" cy="-66" r="30" fill="#4f9150"/>
    <circle cx="-20" cy="-48" r="21" fill="#5ea35a"/>
    <circle cx="20" cy="-50" r="22" fill="#448545"/>
    <circle cx="-6" cy="-76" r="14" fill="#6cb468" opacity=".7"/>
  </g>`;
}

/** Una fábrica del fondo con su chimenea y tres bocanadas de humo. */
function fabrica(x, suelo, clase) {
  const cx = x + 66;
  const bocanadas = [0, 1.6, 3.2].map((demora) =>
    `<circle class="il-bocanada" cx="${cx}" cy="${suelo - 112}" r="13"
       style="animation-delay:-${demora}s"/>`).join("");
  return `<g class="il-fabrica ${clase}">
    <g class="il-humo">${bocanadas}</g>
    <rect x="${x + 59}" y="${suelo - 104}" width="14" height="62"
          fill="#6e625b"/>
    <rect x="${x + 57}" y="${suelo - 108}" width="18" height="7"
          fill="#a64b3a"/>
    <path d="M${x} ${suelo} V${suelo - 44} l22 -14 v14 l22 -14 v14 l22 -14
             V${suelo} Z" fill="#857970"/>
    <rect x="${x + 6}" y="${suelo - 26}" width="12" height="9"
          fill="#e8c76a" opacity=".8"/>
    <rect x="${x + 28}" y="${suelo - 26}" width="12" height="9"
          fill="#e8c76a" opacity=".8"/>
  </g>`;
}

/** Una casa vecina del fondo; sus colores salen de variables CSS. */
function vecino(x, ancho, alto, claseParedes) {
  const suelo = 392;
  const y = suelo - alto;
  const ventanas = [0.2, 0.62].map((p) => {
    const vx = x + ancho * p;
    return `<rect class="il-ventana-v" x="${vx}" y="${y + 14}" width="14"
              height="16" rx="2"/>
            <rect class="il-tabla" x="${vx - 1}" y="${y + 13}" width="16"
              height="18" rx="2"/>`;
  }).join("");
  return `<g>
    <rect class="${claseParedes}" x="${x}" y="${y}" width="${ancho}"
          height="${alto}"/>
    <path class="il-techo-v" d="M${x - 7} ${y} L${x + ancho / 2} ${y - 32}
          L${x + ancho + 7} ${y} Z"/>
    ${ventanas}
    <rect x="${x + ancho * 0.42}" y="${suelo - 24}" width="13" height="24"
          rx="2" fill="#4a3a30"/>
  </g>`;
}

/** Un auto de la autopista; la animación lo cruza de lado a lado. */
function auto(color, duracion, demora, carril) {
  return `<g class="il-auto" style="animation-duration:${duracion}s;
      animation-delay:-${demora}s">
    <rect x="0" y="${carril}" width="36" height="8" rx="3" fill="${color}"/>
    <rect x="8" y="${carril - 6}" width="19" height="7" rx="3"
          fill="${color}"/>
    <circle cx="9" cy="${carril + 8}" r="3.2" fill="#2b2f33"/>
    <circle cx="28" cy="${carril + 8}" r="3.2" fill="#2b2f33"/>
  </g>`;
}

/** Una cresta de ola, para el río. */
function olas() {
  let camino = "";
  for (let fila = 0; fila < 3; fila++) {
    const y = 498 + fila * 17;
    for (let x = -60 + fila * 40; x < 1020; x += 130) {
      camino += `M${x} ${y} q12 -5 24 0 t24 0 `;
    }
  }
  return `<path class="il-ola" d="${camino}" fill="none"
    stroke="#d8eef7" stroke-width="2.2" stroke-linecap="round"/>`;
}

/** Arma toda la escena fija dentro del SVG. Se llama una sola vez. */
function crearEscena(svg) {
  const rayas = Array.from({length: 12}, (_, i) =>
    `<rect x="${20 + i * 82}" y="515" width="44" height="4"
           fill="#f3efe2"/>`).join("");
  const columnas = Array.from({length: 9}, (_, i) =>
    `<rect x="${40 + i * 120}" y="331" width="10" height="22"
           fill="#8d9193"/>`).join("");
  const ciudad = [[330, 60], [366, 96], [402, 52], [436, 128], [472, 74],
                  [508, 108], [544, 58], [580, 88], [616, 66]]
    .map(([x, alto]) => `<rect x="${x}" y="${346 - alto}" width="32"
      height="${alto}" fill="#8da1b3"/>
      <rect x="${x + 6}" y="${346 - alto + 8}" width="6" height="${alto - 14}"
        fill="#a8b9c8" opacity=".7"/>`).join("");

  svg.innerHTML = `
  <defs>
    <linearGradient id="il-cielo" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#7cc0e8"/>
      <stop offset="1" stop-color="#e3f1f3"/>
    </linearGradient>
    <linearGradient id="il-cesped" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#8cc275"/>
      <stop offset="1" stop-color="#68a257"/>
    </linearGradient>
    <linearGradient id="il-agua" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#6bb2d2"/>
      <stop offset="1" stop-color="#3f86ad"/>
    </linearGradient>
  </defs>

  <rect width="960" height="400" fill="url(#il-cielo)"/>
  <g class="il-sol">
    <circle cx="810" cy="84" r="62" fill="#fff3c4" opacity=".45"/>
    <circle cx="810" cy="84" r="38" fill="#ffe27a"/>
  </g>
  <g class="il-nube" style="animation-duration:70s">
    <ellipse cx="170" cy="86" rx="62" ry="20" fill="#fff"/>
    <ellipse cx="205" cy="70" rx="40" ry="20" fill="#fff"/>
    <ellipse cx="140" cy="76" rx="30" ry="15" fill="#fff"/>
  </g>
  <g class="il-nube" style="animation-duration:95s;animation-delay:-40s">
    <ellipse cx="560" cy="56" rx="54" ry="16" fill="#fff" opacity=".9"/>
    <ellipse cx="590" cy="44" rx="34" ry="16" fill="#fff" opacity=".9"/>
  </g>

  <path d="M0 350 Q150 288 320 335 T640 318 T960 338 V390 H0 Z"
        fill="#a9d09b"/>
  <path d="M0 366 Q200 330 420 358 T960 350 V392 H0 Z" fill="#8fbd7e"/>

  <g id="il-ciudad">${ciudad}</g>

  <g id="il-autopista">
    <g>${columnas}</g>
    <rect x="0" y="322" width="960" height="10" fill="#6b6f73"/>
    <rect x="0" y="320" width="960" height="3" fill="#9a9ea1"/>
    ${auto("#c4533c", 7, 1, 311)}
    ${auto("#e2b23c", 9, 5, 311)}
    <g id="il-autos-extra">
      ${auto("#3f7fb0", 6, 3, 311)}
      ${auto("#f2f2ee", 8, 7, 311)}
      ${auto("#4c8c5a", 5.5, 0, 311)}
    </g>
  </g>

  <g id="il-fabricas">
    ${fabrica(30, 366, "il-fabrica-1")}
    ${fabrica(846, 366, "il-fabrica-2")}
    ${fabrica(236, 362, "il-fabrica-3")}
  </g>

  <rect id="il-bruma" width="960" height="395" fill="#b99f78"/>

  <g id="il-vecinos">
    ${vecino(18, 104, 54, "il-pared-v")}
    ${vecino(130, 100, 60, "il-pared-v2")}
    ${vecino(730, 100, 58, "il-pared-v2")}
    ${vecino(842, 102, 52, "il-pared-v")}
  </g>

  <rect y="372" width="960" height="168" fill="url(#il-cesped)"/>
  <rect id="il-seco" y="372" width="960" height="168" fill="#c9b266"/>

  <g id="il-rio">
    <rect y="470" width="960" height="70" fill="url(#il-agua)"/>
    ${olas()}
    <path d="M0 470 Q60 460 130 470 T270 470 T410 470 T550 470 T690 470
             T830 470 T960 470 V478 H0 Z" fill="#78ae63"/>
    <g class="il-barco">
      <path d="M0 0 H70 L58 14 H12 Z" fill="#fbf7ea" stroke="#2f3a42"
            stroke-width="2"/>
      <path d="M34 -4 V-62 L8 -8 Z" fill="#fff" stroke="#2f3a42"
            stroke-width="2"/>
      <path d="M38 -4 V-48 L62 -8 Z" fill="#e8b24a" stroke="#2f3a42"
            stroke-width="2"/>
    </g>
  </g>
  <g id="il-calle">
    <rect y="470" width="960" height="70" fill="#80858a"/>
    <rect y="468" width="960" height="9" fill="#cfc8b9"/>
    ${rayas}
  </g>

  <g class="il-arbol-a">${arbol(104, 436, 1.15)}${arbol(862, 432, 1.05)}</g>
  <g class="il-arbol-b">${arbol(52, 452, 0.95)}${arbol(912, 448, 0.9)}</g>
  <g class="il-arbol-c">${arbol(250, 436, 0.8)}${arbol(704, 434, 0.8)}
    ${arbol(176, 454, 0.7)}${arbol(790, 452, 0.72)}</g>

  <g id="il-casa"></g>

  <g id="il-farola">
    <rect x="768" y="372" width="6" height="78" fill="#3b4348"/>
    <rect x="762" y="366" width="18" height="10" rx="3" fill="#3b4348"/>
    <circle id="il-luz" cx="771" cy="382" r="26" fill="#ffe27a"/>
  </g>

  <g id="il-cartel">
    <rect x="222" y="418" width="6" height="52" fill="#6b4a32"/>
    <rect x="262" y="418" width="6" height="52" fill="#6b4a32"/>
    <rect x="196" y="400" width="96" height="46" rx="5" fill="#fffdf6"
          stroke="#1d6f68" stroke-width="3"/>
    <text x="244" y="418" text-anchor="middle" font-size="12"
          font-weight="700" fill="#1d6f68" letter-spacing="1.5">SE VENDE</text>
    <text id="il-cartel-precio" x="244" y="437" text-anchor="middle"
          font-size="15" font-weight="700" fill="#1c2226">—</text>
  </g>`;
}


/* ---------- La casa principal ---------- */

/** Una ventana; moderna (ancha, con reflejo) o antigua (con arco). */
function ventana(cx, cy, ancho, alto, ctx) {
  const {t, marco, vidrio, rejas, postigos} = ctx;
  const x = cx - ancho / 2;
  const y = cy - alto / 2;
  let forma;
  if (t >= 0.5) {
    const r = ancho / 2;
    forma = `<path d="M${x} ${y + alto} V${y + r} A${r} ${r} 0 0 1
        ${x + ancho} ${y + r} V${y + alto} Z" fill="${vidrio}"
        stroke="${marco}" stroke-width="4"/>
      <path d="M${cx} ${y} V${y + alto} M${x} ${y + alto * 0.55}
        H${x + ancho}" stroke="${marco}" stroke-width="2.5"/>`;
  } else {
    forma = `<rect x="${x}" y="${y}" width="${ancho}" height="${alto}"
        rx="2" fill="${vidrio}" stroke="${marco}" stroke-width="3.5"/>
      <path d="M${x + 5} ${y + alto - 5} L${x + ancho * 0.5} ${y + 5}
        H${x + ancho * 0.72} L${x + 12} ${y + alto - 5} Z"
        fill="#fff" opacity=".35"/>
      <path d="M${cx} ${y} V${y + alto}" stroke="${marco}"
        stroke-width="2"/>`;
  }
  let extra = "";
  if (postigos) {
    for (const lado of [-1, 1]) {
      const px = lado < 0 ? x - 11 : x + ancho + 2;
      extra += `<rect x="${px}" y="${y}" width="9" height="${alto}"
        fill="#3f6b5a"/>`;
    }
  }
  if (rejas) {
    for (let bx = x + 4; bx < x + ancho; bx += 7) {
      extra += `<path d="M${bx} ${y} V${y + alto}" stroke="#20272b"
        stroke-width="2"/>`;
    }
  }
  return forma + extra;
}

/** Distribuye `cantidad` centros parejos dentro de [x0, x1]. */
function repartir(cantidad, x0, x1) {
  return Array.from({length: cantidad},
    (_, i) => x0 + (i + 0.5) * (x1 - x0) / cantidad);
}

/** Devuelve el SVG de la casa según habitaciones, antigüedad y delitos. */
function generarCasa({rm, age, crim}) {
  const n = Math.round(rm);
  const pisos = rm >= 6.5 ? 2 : 1;
  const ancho = Math.round(230 + rm * 17);
  const x0 = CENTRO - ancho / 2;
  const x1 = x0 + ancho;
  const yMuro = SUELO_CASA - pisos * ALTO_PISO;
  const t = limitar(age / 100, 0, 1);

  const ctx = {
    t,
    marco: mezclar("#2f3a42", "#fbf6ea", t),
    vidrio: mezclar("#a9cfe3", "#c9dfe6", t),
    rejas: crim >= 4,
    postigos: t > 0.65,
  };
  const pared = mezclar("#ece7dc", "#b8634a", t);
  const techo = mezclar("#4b5560", "#7a3b2e", t);
  const altoTecho = Math.round(30 + 58 * t);
  const mitad = ancho / 2 + 16;
  // Altura (y) de la superficie del techo a una distancia dx del centro
  const yTecho = (dx) => yMuro - altoTecho * (1 - Math.abs(dx) / mitad);

  // Cuántas ventanas van en cada lugar según las habitaciones. Una casa
  // de un piso lleva hasta 4 ventanas; el resto va a buhardillas
  const arriba = pisos === 2 ? Math.floor(n / 2) : 0;
  const abajo = Math.min(n - arriba, 4);
  const buhardillas = n - arriba - abajo;

  const ranuras = abajo + 1;             // las ventanas de abajo y la puerta
  const indicePuerta = Math.floor(ranuras / 2);
  const centros = repartir(ranuras, x0 + 6, x1 - 6);
  const xPuerta = centros[indicePuerta];
  const anchoVentana = Math.round(Math.min(
    52 - 18 * t, (ancho - 12) / ranuras - 16));
  const altoVentana = Math.round(40 + 12 * t);

  let partes = "";
  // Camino de baldosas desde la puerta hasta la vereda
  partes += `<path d="M${xPuerta - 20} ${SUELO_CASA} L${xPuerta - 40} 470
    H${xPuerta + 40} L${xPuerta + 20} ${SUELO_CASA} Z" fill="#dcd3bf"/>`;
  // La chimenea va detrás del techo
  if (t > 0.45) {
    const cx = CENTRO + ancho * 0.28;
    partes += `<rect x="${cx}" y="${yTecho(cx - CENTRO) - 40}" width="24"
      height="64" fill="#7a3b2e"/>
      <rect x="${cx - 3}" y="${yTecho(cx - CENTRO) - 44}" width="30"
      height="7" fill="#5e2d23"/>`;
  }
  partes += `<rect x="${x0}" y="${yMuro}" width="${ancho}"
    height="${pisos * ALTO_PISO}" fill="${pared}"
    style="transition: fill .4s"/>
    <rect x="${x0}" y="${SUELO_CASA - 8}" width="${ancho}" height="8"
    fill="#000" opacity=".12"/>`;
  if (pisos === 2) {
    partes += `<rect x="${x0}" y="${SUELO_CASA - ALTO_PISO - 3}"
      width="${ancho}" height="5" fill="${ctx.marco}" opacity=".55"/>`;
  }
  partes += `<path d="M${x0 - 16} ${yMuro} L${CENTRO} ${yMuro - altoTecho}
    L${x1 + 16} ${yMuro} Z" fill="${techo}"/>`;
  // Un panel solar en la casa moderna, lo que hoy se espera de una nueva
  if (t < 0.3) {
    partes += `<path d="M${CENTRO + 20} ${yTecho(20) + 3} L${CENTRO + 86}
      ${yTecho(86) + 3} L${CENTRO + 80} ${yTecho(80) + 11} L${CENTRO + 14}
      ${yTecho(14) + 11} Z" fill="#26456b" stroke="#9fb8d4"
      stroke-width="1.5"/>`;
  }

  // Ventanas de abajo y puerta
  const yVentana = SUELO_CASA - 52;
  centros.forEach((cx, i) => {
    if (i === indicePuerta) return;
    partes += ventana(cx, yVentana, anchoVentana, altoVentana, ctx);
  });
  const arco = t >= 0.5;
  const colorPuerta = mezclar("#2f5d62", "#4a2f23", t);
  partes += arco
    ? `<path d="M${xPuerta - 17} ${SUELO_CASA} V${SUELO_CASA - 46}
        A17 17 0 0 1 ${xPuerta + 17} ${SUELO_CASA - 46} V${SUELO_CASA} Z"
        fill="${colorPuerta}"/>`
    : `<rect x="${xPuerta - 17}" y="${SUELO_CASA - 62}" width="34"
        height="62" fill="${colorPuerta}"/>`;
  partes += `<circle cx="${xPuerta + 10}" cy="${SUELO_CASA - 28}" r="2.4"
    fill="#e8c76a"/>
    <rect x="${xPuerta - 24}" y="${SUELO_CASA - 4}" width="48" height="6"
    fill="#c9c1b0"/>`;

  // Ventanas de arriba
  if (arriba > 0) {
    const yAlta = SUELO_CASA - ALTO_PISO - 52;
    const anchoAlta = Math.round(Math.min(
      52 - 18 * t, (ancho - 12) / arriba - 16));
    repartir(arriba, x0 + 6, x1 - 6).forEach((cx) => {
      partes += ventana(cx, yAlta, anchoAlta, altoVentana, ctx);
    });
  }

  // Buhardillas sobre el techo
  const centrosBuhardilla = buhardillas === 1 ? [CENTRO - 30]
    : buhardillas === 2 ? [CENTRO - 56, CENTRO + 4] : [];
  for (const cx of centrosBuhardilla) {
    const yb = yTecho(cx - CENTRO) + 16;
    partes += `<rect x="${cx - 17}" y="${yb - 30}" width="34" height="30"
      fill="${pared}"/>
      <path d="M${cx - 22} ${yb - 30} L${cx} ${yb - 46} L${cx + 22}
      ${yb - 30} Z" fill="${techo}" stroke="${mezclar(techo, "#000000", .25)}"
      stroke-width="1.5"/>
      <rect x="${cx - 9}" y="${yb - 24}" width="18" height="17" rx="2"
      fill="${ctx.vidrio}" stroke="${ctx.marco}" stroke-width="2.5"/>`;
  }

  // Arbustos junto a la puerta
  partes += `<circle cx="${x0 + 14}" cy="${SUELO_CASA - 4}" r="14"
    fill="#4a8a48"/>
    <circle cx="${x1 - 14}" cy="${SUELO_CASA - 4}" r="14" fill="#4a8a48"/>`;
  if (crim < 4) {
    partes += [-8, 0, 9].map((d, i) => `<circle cx="${x0 + 14 + d}"
      cy="${SUELO_CASA - 10 + i * 2}" r="2.6" fill="#ee7b8a"/>`).join("");
  }

  partes += cerco(crim, xPuerta);
  return partes;
}

/** Cerco del frente: de madera blanca en zona tranquila, de rejas si no. */
function cerco(crim, xPuerta) {
  const claro = crim < 4;
  let cerco = "";
  for (let x = 150; x <= 810; x += claro ? 13 : 11) {
    // Un hueco en el cerco para el camino de la puerta
    if (Math.abs(x - xPuerta) < 36) continue;
    cerco += claro
      ? `<path d="M${x} 450 V428 L${x + 4} 422 L${x + 8} 428 V450 Z"
          fill="#fdfcf8" stroke="#cfcabb" stroke-width=".8"/>`
      : `<path d="M${x} 450 V404 M${x - 3} 408 L${x} 398 L${x + 3} 408 Z"
          stroke="#23292d" stroke-width="2.4" fill="#23292d"/>`;
  }
  const riel = claro ? "#fdfcf8" : "#23292d";
  const rieles = claro ? [434, 444] : [412, 440];
  for (const y of rieles) {
    cerco += `<rect x="150" y="${y}" width="${xPuerta - 36 - 150}"
      height="4" fill="${riel}"/>
      <rect x="${xPuerta + 36}" y="${y}" width="${810 - xPuerta - 36}"
      height="4" fill="${riel}"/>`;
  }
  return cerco;
}


/* ---------- Actualización con los datos ---------- */

/**
 * Cambia la escena según la casa. `casa` trae las 12 columnas; las
 * desconocidas deben llegar ya reemplazadas por un valor típico.
 */
function actualizarIlustracion(svg, casa) {
  // Humo y bruma crecen con los óxidos de nitrógeno
  const contaminacion = limitar((casa.NOX - 0.38) / 0.5, 0, 1);
  svg.querySelector("#il-bruma").style.opacity = contaminacion * 0.5;
  svg.style.setProperty("--humo", 0.12 + 0.88 * contaminacion);
  svg.querySelector(".il-sol").style.opacity = 1 - contaminacion * 0.7;

  // Fábricas según el suelo industrial
  const fabricas = casa.INDUS >= 20 ? 3 : casa.INDUS >= 12 ? 2
    : casa.INDUS >= 4 ? 1 : 0;
  [1, 2, 3].forEach((i) => {
    svg.querySelector(`.il-fabrica-${i}`).style.opacity = fabricas >= i ? 1 : 0;
  });

  // La ciudad se ve más cerca cuanto menor es la distancia al empleo
  svg.querySelector("#il-ciudad").style.opacity =
    limitar((6.5 - casa.DIS) / 4.5, 0, 0.95);

  // Autopista: con acceso alto (8 o 24) hay autos; con 24, más
  svg.querySelector("#il-autopista").style.opacity = casa.RAD >= 8 ? 1 : 0;
  svg.querySelector("#il-autos-extra").style.opacity = casa.RAD >= 24 ? 1 : 0;

  // El río reemplaza a la calle
  const rio = casa.CHAS >= 0.5;
  svg.querySelector("#il-rio").style.opacity = rio ? 1 : 0;
  svg.querySelector("#il-calle").style.opacity = rio ? 0 : 1;

  // Barrio: colores de los vecinos, pasto seco y tablas en las ventanas
  const deterioro = limitar((casa.LSTAT - 4) / 24, 0, 1);
  svg.style.setProperty("--v-pared", mezclar("#e9c98f", "#9a9489", deterioro));
  svg.style.setProperty("--v-pared2", mezclar("#a8c9b0", "#8b8479", deterioro));
  svg.style.setProperty("--v-techo", mezclar("#b5543a", "#5c5a56", deterioro));
  svg.style.setProperty("--v-ventana", mezclar("#cfe6f0", "#4a5258", deterioro));
  svg.querySelector("#il-seco").style.opacity = deterioro * 0.45;
  svg.classList.toggle("barrio-duro", casa.LSTAT >= 22);

  // Más suelo en lotes grandes, más árboles
  svg.querySelector(".il-arbol-b").style.opacity = casa.ZN >= 15 ? 1 : 0;
  svg.querySelector(".il-arbol-c").style.opacity = casa.ZN >= 50 ? 1 : 0;

  // Con delitos altos la farola está apagada
  svg.querySelector("#il-luz").style.opacity = casa.CRIM < 4 ? 0.28 : 0;

  svg.querySelector("#il-casa").innerHTML = generarCasa({
    rm: casa.RM, age: casa.AGE, crim: casa.CRIM,
  });
}

/** Escribe el precio en el cartel de "se vende" del jardín. */
function ponerPrecioCartel(svg, texto) {
  svg.querySelector("#il-cartel-precio").textContent = texto;
}
