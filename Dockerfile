FROM python:3.13-slim

RUN useradd -m -u 1000 user
WORKDIR /app
RUN chown user /app
USER user
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY --chown=user src ./src
COPY --chown=user api ./api
COPY --chown=user data/processed/transactions_clean.csv ./data/processed/transactions_clean.csv

ENV USE_EMBEDDINGS=false
RUN python -m src.train

EXPOSE 10000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]