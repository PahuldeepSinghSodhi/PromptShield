# Demo image for free hosts (Render): builds from the repo and pulls the serving
# artifacts from the v0.1.0 GitHub release, since the models are not in git.
# Build from the repo root: docker build -f deploy/demo/Dockerfile .
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY src ./src
# the secret detector was pickled with scikit-learn 1.9.1
RUN pip install . "scikit-learn==1.9.1"

FROM python:3.12-slim
ENV PATH="/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build /venv /venv
COPY --chown=app:app src ./src
COPY --chown=app:app configs ./configs
COPY --chown=app:app dashboard ./dashboard
COPY --chown=app:app reports ./reports
ARG REL=https://github.com/PahuldeepSinghSodhi/PromptShield/releases/download/v0.1.0
ADD --chown=app:app ${REL}/model.int8.onnx ${REL}/tokenizer.json ${REL}/tokenizer_config.json ${REL}/promptshield.json ./artifacts/ner/
ADD --chown=app:app ${REL}/secret_detector.joblib ${REL}/drift_reference.json ./artifacts/
USER app
# the host injects PORT; SQLite in /tmp is ephemeral, which is fine for a demo
ENV PROMPTSHIELD_HOME=/app \
    PROMPTSHIELD_DATABASE_URL=sqlite:////tmp/promptshield.db \
    PROMPTSHIELD_NER_BACKEND=onnx \
    PROMPTSHIELD_LLM_PROVIDER=mock \
    PORT=10000
EXPOSE 10000
CMD ["sh", "-c", "uvicorn promptshield.serving.app:app --host 0.0.0.0 --port ${PORT}"]
