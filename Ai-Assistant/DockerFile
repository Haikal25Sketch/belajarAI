FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

COPY . .

# Railway akan otomatis mengisi variable $PORT, jadi kita panggil secara dinamis
CMD ["sh", "-c", "uvicorn api:app_api --host 0.0.0.0 --port $PORT"]
