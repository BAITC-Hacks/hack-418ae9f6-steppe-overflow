# Веб-интерфейс Steppe Wind. Данные, кэш погоды и обученная модель — внутри образа,
# поэтому контейнер работает без доступа к Open-Meteo.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Сначала зависимости (слой кэшируется, пока не меняется requirements.txt)
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install --no-deps -e . \
    && useradd --create-home --uid 1000 app \
    && chown -R app:app /app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["python", "app/serve.py", "--server.port=8501", "--server.address=0.0.0.0"]
