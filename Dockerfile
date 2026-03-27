FROM python:3.12-slim

WORKDIR /app

# System deps required by PyMuPDF and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install uv --no-cache-dir

# Install CPU-only PyTorch FIRST so uv doesn't pull the CUDA wheel when
# resolving glmocr[layout] (torch>=2.10.0). The extra-index-url flag tells
# pip to prefer the cpu build from the PyTorch index.
RUN uv pip install --system --no-cache \
    torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining deps (layer cache)
COPY pyproject.toml .
RUN uv pip install --system --no-cache -e ".[layout]"

COPY src/ src/
COPY scripts/ scripts/
COPY config.yaml .
COPY ollama/config.yaml ollama/config.yaml

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000
CMD ["python", "scripts/serve.py"]
