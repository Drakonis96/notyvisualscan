FROM python:3.9-slim

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create persistent data directory
RUN mkdir -p /data

# Copia e instala las dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código de la aplicación
COPY app/ ./app/

ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app/app.py
ENV FLASK_ENV=development

EXPOSE 5007

CMD ["flask", "run", "--host=0.0.0.0", "--port=5007"]
