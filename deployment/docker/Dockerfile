FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    procps \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/data/raw /app/data/cache /app/logs

COPY . .

ENV PYTHONPATH=/app
ENV DOCKER_ENV=true
ENV PYTHONFAULTHANDLER=1
ENV PYTHONUNBUFFERED=1

# Настройки для многопоточной модели
ENV MAX_WORKERS=12
ENV THREAD_NAME_PREFIX=search_worker

# Настройки для оптимизации PyTorch
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4
ENV TORCH_NUM_THREADS=4

# Настройки для FAISS
ENV FAISS_OMP_NUM_THREADS=4
ENV FAISS_NO_TEMP_MEMORY=0

# Настройки для uvicorn
ENV UVICORN_WORKERS=1
ENV UVICORN_TIMEOUT=120

# Настройки для кэширования
ENV CACHE_SIZE=1000
ENV QUERY_CACHE_SIZE=1000
ENV RESPONSE_CACHE_SIZE=500

# Настройка для улучшения отладки потоков
ENV PYTHONTHREADDEBUG=0

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8005/ || exit 1

EXPOSE 8005

CMD ["python", "-u", "main.py"] 