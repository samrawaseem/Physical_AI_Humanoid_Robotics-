# Multi-stage Dockerfile for Hugging Face Spaces

# Stage 1: Build Frontend
FROM node:18-alpine as build-frontend
WORKDIR /app/frontend

# Copy frontend dependency files
COPY frontend/package.json frontend/package-lock.json ./

# Install dependencies
RUN npm ci

# Copy frontend source code
COPY frontend/ ./

# Build the Docusaurus app
RUN npm run build

# Stage 2: Setup Backend and Serve
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements.txt ./backend/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy built frontend assets from Stage 1 to a static directory
COPY --from=build-frontend /app/frontend/build ./static

# Create a non-root user (good practice for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose the port HF Spaces expects (7860)
EXPOSE 7860

# Run the deployment script
CMD ["python", "backend/deploy.py"]
