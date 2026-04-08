---
title: NeuralArch-Bench
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# NeuralArch-Bench

An OpenEnv hackathon submission where an LLM agent acts as a **Deep Learning Researcher**, modifying PyTorch model architectures and receiving rewards based on accuracy and efficiency on Fashion-MNIST, MNIST, and CIFAR-10.

## How it works

1. `reset()` — randomly picks an architecture from the library and a dataset for the episode
2. `step(action)` — the agent either provides a full model replacement or a natural-language layer modification (e.g. `"add BatchNorm1d after fc1"`)
3. The environment trains the model in an isolated subprocess and returns accuracy, loss curve, and parameter count as observations

## Reward

| Signal | Value |
|--------|-------|
| Code runs without error | +0.1 |
| Accuracy improvement | +10.0 × Δacc |
| Parameter efficiency | −0.01 × (params / 1000) |
| RuntimeError | −1.0 |

## Architecture Library

`simple_mlp` · `batch_norm_mlp` · `conv_net` · `dropout_regularized` · `resnet_like`

## Dataset Library

`fashion_mnist` · `mnist` · `cifar10`
