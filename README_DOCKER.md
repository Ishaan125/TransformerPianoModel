# Running the Piano AI API in Docker

This file explains how to build and run the `api.py` FastAPI server using Docker.

Two modes are supported:
- CPU-only (default): simpler, installs Python deps from `requirements.txt` and does not install a GPU-specific `torch` wheel.
- GPU-accelerated: requires selecting a matching PyTorch wheel for your CUDA and Python version and passing it as a build-arg.

Prerequisites
- Docker installed (and `docker-compose` if you plan to use `docker-compose`).
- If you want GPU support: `nvidia-container-toolkit` installed on the host and an appropriate CUDA driver.

CPU-only build & run (recommended for first test)
```bash
# build
docker build -t piano-ai:latest .

# run (foreground)
docker run --rm -p 8000:8000 -v "${PWD}":/app piano-ai:latest
```

Visit `http://127.0.0.1:8000/docs` to view the OpenAPI docs and try endpoints.

Passing a PyTorch wheel for GPU
1. Determine the right wheel URL from https://pytorch.org/get-started/locally/ for your CUDA and Python.
2. Build with the `TORCH_WHEEL` build-arg, URL must be URL-encoded if it contains `+`:
```bash
docker build --build-arg TORCH_WHEEL="https://download.pytorch.org/whl/cu118/torch-2.2.0%2Bcu118-cp311-cp311-linux_x86_64.whl" -t piano-ai-gpu:latest .
```
3. Run with the NVIDIA runtime (example):
```bash
docker run --gpus all --rm -p 8000:8000 -v "${PWD}":/app piano-ai-gpu:latest
```

Using docker-compose
```bash
docker-compose up --build
```
