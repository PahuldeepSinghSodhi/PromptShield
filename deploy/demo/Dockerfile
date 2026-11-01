# Hugging Face Space image: builds PromptShield from GitHub and pulls the serving
# artifacts from the v0.1.0 release, so the Space repo holds only this file + README.
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
ARG REF=main
RUN git clone --depth 1 --branch ${REF} https://github.com/PahuldeepSinghSodhi/PromptShield.git /app
WORKDIR /app
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
# the secret detector was pickled with scikit-learn 1.9.1
RUN pip install . "scikit-learn==1.9.1"

FROM python:3.12-slim
ENV PATH="/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build /venv /venv
COPY --from=build --chown=app:app /app/src ./src
COPY --from=build --chown=app:app /app/configs ./configs
COPY --from=build --chown=app:app /app/dashboard ./dashboard
COPY --from=build --chown=app:app /app/reports ./reports
ARG REL=https://github.com/PahuldeepSinghSodhi/PromptShield/releases/download/v0.1.0
ADD --chown=app:app ${REL}/model.int8.onnx ${REL}/tokenizer.json ${REL}/tokenizer_config.json ${REL}/promptshield.json ./artifacts/ner/
ADD --chown=app:app ${REL}/secret_detector.joblib ${REL}/drift_reference.json ./artifacts/
USER app
ENV PROMPTSHIELD_HOME=/app \
    PROMPTSHIELD_DATABASE_URL=sqlite:////tmp/promptshield.db \
    PROMPTSHIELD_NER_BACKEND=onnx \
    PROMPTSHIELD_LLM_PROVIDER=mock \
    PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn promptshield.serving.app:app --host 0.0.0.0 --port ${PORT}"]
