FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY app ./app
COPY train.py ./train.py
COPY pyproject.toml ./pyproject.toml
COPY scripts ./scripts
COPY models ./models
COPY outputs ./outputs

EXPOSE 8501 8000

# APP_MODE=api launches FastAPI; default is the Streamlit dashboard.
CMD ["sh", "-c", "if [ \"$APP_MODE\" = \"api\" ]; then uvicorn app.api:app --host 0.0.0.0 --port 8000; else streamlit run app/app.py --server.address=0.0.0.0 --server.port=8501; fi"]
