# Base image
FROM python:3.11-slim

# Working directory
WORKDIR /app

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install pytest

# Copy project files
COPY . .

# Optional: Train model during build
RUN python -m src.model

# Expose FastAPI port
EXPOSE 8000

# Start API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
