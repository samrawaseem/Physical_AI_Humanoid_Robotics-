# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app/backend
# We use this structure to match the layout expected by deploy.py
WORKDIR /app/backend

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Expose the port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Run the deployment script
CMD ["python", "deploy.py"]
