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

# 4. Iniciar Celery Workers y Celery Beat en segundo plano
# Dos workers: `exec` solo atiende la cola `execution` (despacho de 15:57 y conciliaciones) para que
# el envío de órdenes nunca espere detrás del cálculo de señales; `main` atiende el pipeline y el
# resto de colas. Coste: un proceso Celery extra (~150-250 MB de RAM según dependencias cargadas);
# validar que el plan del hosting lo admite.
# IMPORTANTE: workers y schedules (setup_botops_schedules) deben desplegarse juntos; un worker viejo
# que no escuche la cola `pipeline` dejaría el pipeline sin ejecutar.
echo "[4/5] Iniciando Celery Workers (exec + main) y Celery Beat..."
celery -A config worker -n exec@%h -l info -Q execution --concurrency=1 --prefetch-multiplier=1 &
celery -A config worker -n main@%h -l info -Q celery,pipeline,marketdata,signals,accounting --concurrency=${CELERY_MAIN_CONCURRENCY:-2} --prefetch-multiplier=1 &
celery -A config beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler --pidfile=/tmp/celerybeat.pid -s /tmp/celerybeat-schedule &



# 5. Iniciar Servidor Gunicorn
echo "[5/5] Iniciando servidor Gunicorn en puerto ${PORT:-8000}..."
exec gunicorn config.wsgi:application \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers 3 \
    --threads 2 \
    --timeout 120
