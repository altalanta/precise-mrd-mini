
#!/bin/bash
set -e

# Wait for Redis to be available
# This is a simple loop, in production you might want a more robust solution like wait-for-it.sh
until nc -z -v -w30 redis 6379
do
  echo "Waiting for Redis connection..."
  sleep 1
done
echo "Redis is up - executing command"

# Run Celery worker
celery -A src.precise_mrd.celery_app.celery_app worker --loglevel=info -c 1



