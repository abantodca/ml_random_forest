-- Crea la base `forecasts` (de la API) junto a la base `mlflow` en el MISMO
-- Postgres local. Idempotente: solo crea si no existe. El entrypoint de la
-- imagen postgres ejecuta este script una vez, al inicializar el volumen.
--
-- En producción NO se usa este script: la API auto-crea la base `forecasts`
-- en el RDS de MLflow en su primer arranque (app/models/database.ensure_database).
SELECT 'CREATE DATABASE forecasts'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'forecasts')\gexec
