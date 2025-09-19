# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (curl, nodejs, npm)
RUN apt-get update && apt-get install -y curl gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries
RUN pip install --no-cache-dir pandas scikit-learn numpy

# Copy project files
COPY . .

# Install React dependencies (package.json must exist in your project)
RUN cd frontend && npm install && npm run build

# Expose ports (5000 for Python backend, 3000 for React dev server if needed)
EXPOSE 5000 3000

# Default command (run backend, adjust if using Flask/Django/FastAPI)
CMD ["python", "app.py"]
