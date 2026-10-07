#!/bin/bash
set -e

echo "================================================="
echo "=== INICIANDO DESPLIEGUE AUTOPORTONAGER (PROD) =="
echo "================================================="

# 1. Aplicar migraciones
echo "[1/5] Aplicando migraciones de base de datos..."
python manage.py migrate --no-input

# 2. Recolectar archivos estáticos
echo "[2/5] Recolectando archivos estáticos (WhiteNoise)..."
python manage.py collectstatic --no-input

# 3. Configurar tareas periódicas y schedules
echo "[3/5] Configurando tareas y schedules automatizados de trading..."
python manage.py setup_botops_schedules

export DJANGO_SETTINGS_MODULE=${DJANGO_SETTINGS_MODULE:-config.settings.prod}

# Limpiar posibles locks o pidfiles antiguos de Celery Beat
rm -f /tmp/celerybeat.pid /tmp/celerybeat-schedule* celerybeat.pid celerybeat-schedule*

# 4. Iniciar Celery Worker y Celery Beat en segundo plano
echo "[4/5] Iniciando Celery Worker y Celery Beat..."
celery -A config worker -l info -Q celery,marketdata,signals,execution,accounting --concurrency=2 &
celery -A config beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler --pidfile=/tmp/celerybeat.pid -s /tmp/celerybeat-schedule &



# 5. Iniciar Servidor Gunicorn
echo "[5/5] Iniciando servidor Gunicorn en puerto ${PORT:-8000}..."
exec gunicorn config.wsgi:application \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers 3 \
    --threads 2 \
    --timeout 120
