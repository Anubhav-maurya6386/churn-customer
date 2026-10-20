FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app/src
# Train inside the image so the artifact matches the code (or mount your own artifacts/ volume)
RUN python -m churn.data generate && python -m churn.train
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1
CMD uvicorn churn.api:app --host 0.0.0.0 --port ${PORT:-8000}
