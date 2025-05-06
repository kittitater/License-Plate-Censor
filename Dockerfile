FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve_model.py .

EXPOSE 80
CMD ["uvicorn", "serve_model:app", "--host", "0.0.0.0", "--port", "80"]
