#!/bin/bash
set -e

: "${S3_BUCKET:?S3_BUCKET requerido}"
: "${AWS_DEFAULT_REGION:?AWS_DEFAULT_REGION requerido}"

mkdir -p /usr/share/nginx/html/reports /usr/share/nginx/html/artifacts

# Sync inicial (bloqueante: arrancamos nginx con data ya cargada)
aws s3 sync "s3://${S3_BUCKET}/reports/"   /usr/share/nginx/html/reports/   --no-progress || true
aws s3 sync "s3://${S3_BUCKET}/artifacts/" /usr/share/nginx/html/artifacts/ --no-progress || true

# Sync loop en background (cada 60s)
(
  while true; do
    sleep 60
    aws s3 sync "s3://${S3_BUCKET}/reports/"   /usr/share/nginx/html/reports/   --delete --no-progress >/dev/null 2>&1 || true
    aws s3 sync "s3://${S3_BUCKET}/artifacts/" /usr/share/nginx/html/artifacts/ --delete --no-progress >/dev/null 2>&1 || true
  done
) &

# Foreground: nginx
exec nginx -g 'daemon off;'
