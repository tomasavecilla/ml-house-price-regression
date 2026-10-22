# La interfaz del tasador:  docker build -t precio-casas-app app/
# Es una página estática, así que alcanza con nginx: sirve los archivos
# y reenvía /api/ a la API (ver nginx.conf). La imagen pesa menos de
# 100 MB, contra los cientos que pesaba la versión en Python
FROM nginx:1.29-alpine

# La configuración propia reemplaza la que trae la imagen
COPY nginx.conf /etc/nginx/conf.d/default.conf

# La página. Acá no viaja ningún modelo: la app solo habla con la API
COPY index.html estilos.css tasador.js /usr/share/nginx/html/

EXPOSE 80
