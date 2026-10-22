#!/bin/bash
# Copia las bases sqlite de MLflow (runs, registro de modelos y usuarios) al
# bucket de S3, para no perderlas si se rompe la instancia. Corre antes de
# cada despliegue y una vez por día (tasador-respaldo.timer).
# Necesita la variable BUCKET (el despliegue la deja en respaldo.env).
#
# No alcanza con copiar el archivo: con el servidor escribiendo, la copia
# puede quedar rota. La API de backup de sqlite saca una copia consistente.
# Cada copia pisa a la anterior, pero el bucket tiene versionado: S3
# guarda las anteriores por 30 días.
#
# Para restaurar en una instancia nueva, antes de levantar MLflow:
#   aws s3 cp s3://$BUCKET/respaldos/mlflow.db /opt/tasador/datos/
#   aws s3 cp s3://$BUCKET/respaldos/basic_auth.db /opt/tasador/datos/
#   chown 1000:1000 /opt/tasador/datos/*.db
set -euo pipefail
cd /opt/tasador
mkdir -p respaldo-tmp
for base in mlflow basic_auth; do
  # Si todavía no existe (primer despliegue), no hay nada que copiar
  [ -f "datos/$base.db" ] || continue
  rm -f "respaldo-tmp/$base.db"
  python3 -c "
import sqlite3
origen = sqlite3.connect('datos/$base.db')
destino = sqlite3.connect('respaldo-tmp/$base.db')
origen.backup(destino)
"
  aws s3 cp --only-show-errors "respaldo-tmp/$base.db" "s3://$BUCKET/respaldos/$base.db"
  rm -f "respaldo-tmp/$base.db"
done
echo "Respaldo en s3://$BUCKET/respaldos/"
