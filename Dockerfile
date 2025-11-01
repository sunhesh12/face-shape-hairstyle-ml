# Use the official Python image as the base
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# --- 1. Install System Dependencies (Needed for Node/NPM and ML libs) ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        build-essential \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js v18 (required to build the React frontend)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs 

# --- 2. Install Python Dependencies (ML and Web Server) ---
# NOTE: We install TensorFlow first as it can have complex dependencies
# We use the versions required for the Face Shape Predictor application
RUN pip install --no-cache-dir \
    tensorflow \
    scikit-learn \
    numpy \
    pandas \
    joblib \
    Pillow \
    Flask \
    Flask-CORS

# --- 3. Copy Project Files ---
# Copy everything needed for the build and runtime
COPY . .

# --- 4. Build React Frontend ---
# Change to the frontend directory
WORKDIR /app/frontend

# Install React dependencies and run the production build
RUN npm install
RUN npm run build

# Change back to the application root directory
WORKDIR /app

# --- 5. Final Configuration and Execution ---
# Expose the port the Flask server uses
EXPOSE 5000

# Command to run the Flask backend server
# IMPORTANT: Adjust 'server.py' to 'app.py' if that is your main file name
CMD ["python3", "server.py"]
