# NeuralArch-Bench — OpenEnv environment
# Base: official PyTorch image (CPU-only for HF Spaces free tier)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# HF Spaces runs as a non-root user; create one to match
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ── system deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ────────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── pre-download all datasets so reset() is instant ───────────────────────────
RUN python - <<'EOF'
from torchvision import datasets, transforms
t = transforms.ToTensor()

datasets.FashionMNIST("/tmp/fashion_mnist", train=True,  download=True, transform=t)
datasets.FashionMNIST("/tmp/fashion_mnist", train=False, download=True, transform=t)
print("Fashion-MNIST cached OK")

datasets.MNIST("/tmp/mnist", train=True,  download=True, transform=t)
datasets.MNIST("/tmp/mnist", train=False, download=True, transform=t)
print("MNIST cached OK")

datasets.CIFAR10("/tmp/cifar10", train=True,  download=True, transform=t)
datasets.CIFAR10("/tmp/cifar10", train=False, download=True, transform=t)
print("CIFAR-10 cached OK")
EOF

# ── copy project ───────────────────────────────────────────────────────────────
COPY . .

# Fix ownership so appuser can write temp dirs
RUN chown -R appuser:appuser /app /tmp/fashion_mnist /tmp/mnist /tmp/cifar10

USER appuser

ENV FASHION_MNIST_DIR=/tmp/fashion_mnist
ENV MNIST_DIR=/tmp/mnist
ENV CIFAR10_DIR=/tmp/cifar10
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
