FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Remove broken Rust and install fresh
RUN rm -rf /root/.rustup /root/.cargo && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Force install numpy 1.x FIRST
RUN pip3 install --no-cache-dir --force-reinstall "numpy>=1.21.0,<2.0.0"

# Copy requirements WITHOUT opencv-python (use base image's OpenCV)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt || pip3 install --no-cache-dir $(grep -v opencv-python requirements.txt)

# Copy model files and directories
COPY hc_model/ ./hc_model/
COPY skin_type_model/ ./skin_type_model/
COPY DepiModels.py .
COPY ml_service.py .

# Set environment variables
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV DEPI_MODEL_ROOT=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "ml_service.py"]