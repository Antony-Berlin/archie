---
title: NeuralArch-Bench
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# NeuralArch-Bench

An OpenEnv hackathon submission where an LLM agent acts as a **Deep Learning Researcher**, improving PyTorch model architectures through a structured 3-phase cycle on tabular datasets.

## 3-Phase Cycle (per turn)

| Phase | Action | Reward |
|-------|--------|--------|
| **DIAGNOSE** | Agent explains what is wrong with the current model | 0–0.7 heuristic |
| **PLAN** | Agent describes the specific change to make | 0–0.7 heuristic |
| **IMPLEMENT** | Agent writes new model code; training runs | accuracy-based |

## Reward Signals

| Signal | Value |
|--------|-------|
| Code runs without error | +0.1 |
| Accuracy improvement | +10.0 × Δacc |
| Parameter efficiency | −0.01 × (params / 1000) |
| RuntimeError | −1.0 |
| Diagnosis mentions accuracy/performance | +0.3 |
| Diagnosis mentions overfitting/underfitting | +0.2 |
| Plan mentions specific layer types | +0.3 |
| Plan explains expected improvement | +0.2 |

## Datasets (tabular, trains in seconds)

`iris` · `wine` · `breast_cancer`

## Architecture Library

`tabular_simple_mlp` · `tabular_batch_norm_mlp` · `tabular_dropout_mlp` · `tabular_deep_mlp` · `tabular_minimal`
