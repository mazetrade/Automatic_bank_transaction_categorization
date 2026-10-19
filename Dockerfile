FROM python:3.13-slim

RUN useradd -m -u 1000 user
WORKDIR /app
RUN chown user /app
USER user
ENV PATH="/home/user/.local/bin:$PATH"

RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY --chown=user src ./src
COPY --chown=user api ./api
COPY --chown=user data/processed/transactions_clean.csv ./data/processed/transactions_clean.csv

RUN python -m src.train

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]