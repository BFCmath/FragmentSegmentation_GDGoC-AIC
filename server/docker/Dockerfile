FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Copy requirements.txt from parent directory
COPY ../requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt -q

# Copy all server files from parent directory
COPY .. .

# Set environment for container
ENV LOG_TO_STDOUT=1

EXPOSE 3000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"] 