# ---- build stage: install dependencies into a virtualenv ----
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY src ./src
# serving needs only the core deps + postgres driver (no torch: inference runs on ONNX Runtime)
RUN pip install ".[postgres]"

# ---- runtime stage ----
FROM python:3.12-slim
ENV PATH="/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build /venv /venv
COPY --chown=app:app src ./src
COPY --chown=app:app configs ./configs
COPY --chown=app:app dashboard ./dashboard
# model artifacts: tokenizer, INT8 ONNX model, secret detector, drift reference
COPY --chown=app:app artifacts/ner/tokenizer.json artifacts/ner/tokenizer_config.json artifacts/ner/promptshield.json artifacts/ner/model.int8.onnx ./artifacts/ner/
COPY --chown=app:app artifacts/secret_detector.joblib artifacts/drift_reference.json ./artifacts/
COPY --chown=app:app reports/*.json ./reports/
USER app
ENV PROMPTSHIELD_DATABASE_URL=sqlite:////tmp/promptshield.db \
    PROMPTSHIELD_NER_BACKEND=onnx \
    PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/v1/health')"
CMD ["sh", "-c", "uvicorn promptshield.serving.app:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS:-1}"]
