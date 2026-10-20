# Inferencia por lotes en Docker

Esta carpeta es la entrega de la consigna de Docker: una imagen que
recibe un CSV de casas y devuelve el mismo CSV con el precio estimado.
Contiene solo lo que la imagen necesita:

- `inferencia.py`: el script. Carga el pipeline y predice.
- `modelo.joblib`: el pipeline entrenado (preprocesamiento + modelo),
  exportado desde el registro de MLflow con `exportar_modelo.py`.
- `requirements.txt`: las librerías de inferencia, nada más.
- `Dockerfile`: cómo se arma la imagen.
- `casas_ejemplo.csv`: cinco casas para probar.

## Construir la imagen

Desde esta carpeta:

```bash
docker build -t precio-casas .
```

## Correr la inferencia

El contenedor no ve los archivos de la máquina: hay que montarle la
carpeta donde están los CSV (`-v`). Acá se monta la carpeta actual
como `/datos` y se le pasan la entrada y la salida con esa ruta.

```bash
docker run --rm -v "$(pwd)":/datos precio-casas \
    /datos/casas_ejemplo.csv /datos/precios.csv
```

Imprime `5 casas tasadas en /datos/precios.csv` y deja `precios.csv`
en la carpeta actual. `--rm` borra el contenedor al terminar.

## Formato de entrada y salida

Entrada: un CSV con una fila por casa y, al menos, las columnas con
las que se entrenó el modelo:

`CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, LSTAT`

Las celdas vacías se imputan con lo que el pipeline aprendió al
entrenar. Las columnas que sobran se ignoran y se copian a la salida.
Si falta alguna, el script termina con error y dice cuáles.

Salida: el mismo CSV con una columna más, `precio_estimado`, en miles
de dólares (la unidad de `MEDV`), redondeada a un decimal.
