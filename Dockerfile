# Multi-stage build for the Smart Parking Lot Edge Service
FROM python:3.10-slim AS base

# Install system dependencies for OpenCV and DepthAI VPU communication
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libusb-1.0-0-dev \
    udev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Flask Production Inference Stage ---
FROM base AS prod-flask
COPY Edge-Device-Code/ /app/
EXPOSE 5000
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Command to run the Flask application
# Note: DepthAI requires USB access. Run on host with:
# docker run --device /dev/bus/usb --network host smart-parking-flask
CMD ["python", "main.py"]

# --- Streamlit Calibration Stage ---
FROM base AS prod-streamlit
COPY Edge-Device-Code/ /app/
EXPOSE 8501
ENV PYTHONUNBUFFERED=1

# Command to run Streamlit Calibration
# Run with:
# docker run --device /dev/bus/usb -p 8501:8501 smart-parking-streamlit
CMD ["streamlit", "run", "parking_app.py", "--server.address=0.0.0.0"]
