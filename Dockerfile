
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY mlflow.db .
COPY mlruns/ ./mlruns/

# Make Mac-specific MLflow artifact paths accessible
RUN mkdir -p /Users/kapilvyas/Downloads/G058_MP282 && ln -s /app/mlruns /Users/kapilvyas/Downloads/G058_MP282/mlruns

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
