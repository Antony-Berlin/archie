---
title: NeuralArch-Bench
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
---

# NeuralArch-Bench

An **OpenEnv** environment where an LLM agent acts as a *Deep Learning Researcher*, diagnosing and improving PyTorch model architectures through a structured 3-phase cycle on tabular datasets.

## Environment Description

The agent is presented with a broken or suboptimal PyTorch model and must iteratively improve it through three structured phases per turn:

1. **DIAGNOSE** — Explain what is wrong with the current model
2. **PLAN** — Describe the specific architectural change to make
3. **IMPLEMENT** — Write the improved model code (triggers real training)

Training runs on real tabular datasets (iris, wine, breast_cancer) with PyTorch. Each implement step measures actual test accuracy, parameter count, and loss curve — providing genuine signal for the agent.

## Action Space

```python
class NeuralArchAction(Action):
    diagnosis:      Optional[str]  # Phase 1: plain-text diagnosis
    change_plan:    Optional[str]  # Phase 2: plain-text architectural plan
    new_model_code: Optional[str]  # Phase 3: full Python source (ArchModel)
```

Exactly one field should be set per step, matching `current_phase` in the observation.

## Observation Space

```python
class NeuralArchObservation(Observation):
    # Training metrics
    current_code:      str            # Source of the current PyTorch model
    architecture_name: str            # Name from arch library, or "custom"
    dataset_name:      str            # Active dataset: iris | wine | breast_cancer
    last_accuracy:     float          # Test accuracy from most recent training run (0–1)
    param_count:       int            # Total trainable parameters
    loss_curve:        List[float]    # Loss values from last 5 epochs
    error_logs:        Optional[str]  # Errors from last training run

    # Phase tracking
    current_phase:  str               # "diagnose" | "plan" | "implement"
    last_diagnosis: Optional[str]     # Agent's diagnosis (phase 1)
    last_plan:      Optional[str]     # Agent's change plan (phase 2)
    phase_rewards:  List[float]       # Rewards so far in current cycle

    # Task grading
    task_id:    str    # Active task ID
    task_score: float  # Normalized task score in [0.0, 1.0] from grader
```

## Reward Function

| Signal | Value |
|--------|-------|
| Code runs without error | +0.1 |
| Accuracy improvement | +10.0 × Δacc |
| Parameter efficiency penalty | −0.01 × (params / 1000) |
| RuntimeError during training | −1.0 |
| Diagnosis mentions accuracy/performance | +0.3 |
| Diagnosis mentions overfitting/underfitting | +0.2 |
| Diagnosis mentions specific layer names | +0.1 |
| Diagnosis length > 100 chars | +0.1 |
| Plan mentions specific layer types | +0.3 |
| Plan explains expected improvement | +0.2 |
| Plan references diagnosis (shared words) | +0.1 |
| Plan length > 80 chars | +0.1 |

Rewards are dense — the agent receives a signal on every step, not just at episode end.

## Tasks

| Task ID | Difficulty | Success Criteria | Grader |
|---------|-----------|-----------------|--------|
| `arch-foundations` | Easy | accuracy > 0.85 AND implement_steps ≤ 2 | `ArchFoundationsGrader` |
| `efficient-net` | Medium | accuracy > 0.80 AND param_count < 10,000 | `EfficientNetGrader` |
| `residual-depth` | Hard | accuracy > 0.75 on breast_cancer dataset | `ResidualDepthGrader` |

Each grader returns a normalized score in `[0.0, 1.0]`. Partial credit is awarded for progress toward the success criteria (see `core/task_graders.py`).

### Task Details

**arch-foundations (Easy)**
Demonstrate quick architectural improvement. Full credit requires reaching > 85% accuracy within 2 implement steps. Exceeding the step budget caps score at 50%.

**efficient-net (Medium)**
Balance accuracy and model size. Score = 70% accuracy sub-score + 30% efficiency sub-score. Models exceeding 10k parameters receive partial efficiency credit based on how far over budget they are.

**residual-depth (Hard)**
Handle a high-dimensional dataset (30 features, binary classification). Score is proportional to accuracy achievement on the breast_cancer dataset. Wrong dataset yields 50% penalty.

## Datasets

| Dataset | Features | Classes | Notes |
|---------|----------|---------|-------|
| `iris` | 4 | 3 | Classic multi-class, very separable |
| `wine` | 13 | 3 | Chemical analysis, moderate difficulty |
| `breast_cancer` | 30 | 2 | High-dimensional binary classification |

## Architecture Library (starting models)

| Name | Description |
|------|-------------|
| `tabular_simple_mlp` | 2-layer MLP, no regularization |
| `tabular_batch_norm_mlp` | MLP with BatchNorm1d |
| `tabular_dropout_mlp` | MLP with Dropout(0.3) |
| `tabular_deep_mlp` | 4-layer deep MLP, prone to overfitting |
| `tabular_minimal` | Under-parameterized, 2 layers of 8 units |

## Setup & Usage

### Local (without Docker)

```bash
pip install openenv-core openai torch scikit-learn

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run inference (set your API key first)
export HF_TOKEN=<your-key>
export MY_ENV_V4_TASK=arch-foundations   # or efficient-net, residual-depth
python inference.py
```

### Docker

```bash
docker build -t neural-arch-bench .
docker run -p 7860:7860 \
  -e HF_TOKEN=<your-key> \
  -e MY_ENV_V4_TASK=arch-foundations \
  neural-arch-bench
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `HF_TOKEN` / `API_KEY` | — | Auth key for LLM API |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `MY_ENV_V4_TASK` | `arch-foundations` | Task to evaluate |
| `MY_ENV_V4_BENCHMARK` | `neural-arch-bench` | Benchmark name |

### OpenEnv Validation

```bash
pip install openenv-core
openenv validate .
```

## Baseline Scores

Baseline measured with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference Router, 4 DPI cycles (12 steps), temperature 0.7.

| Task | Score | Steps | Notes |
|------|-------|-------|-------|
| `arch-foundations` | 0.412 | 12 | Accuracy improves but often exceeds 2-step budget |
| `efficient-net` | 0.285 | 12 | Struggles to balance accuracy with param constraint |
| `residual-depth` | 0.348 | 12 | Moderate; depends on dataset selection luck |

> Scores are normalized to `[0.0, 1.0]` as `sum(rewards) / max_possible_reward`. A score of `0.5` is considered a pass threshold.

## Repository Structure

```
scaler-hack/
├── core/
│   ├── arch_library.py      # 5 starting architecture templates
│   ├── dataset_library.py   # iris, wine, breast_cancer configs
│   ├── models.py            # NeuralArchAction + NeuralArchObservation (Pydantic)
│   ├── task_graders.py      # ArchFoundationsGrader, EfficientNetGrader, ResidualDepthGrader
│   └── trainer.py           # Subprocess training script; writes results.json
├── server/
│   ├── app.py               # FastAPI OpenEnv server
│   └── neural_arch_environment.py  # reset() / step() / state() logic
├── inference.py             # LLM agent evaluation loop (OpenAI client)
├── openenv.yaml             # OpenEnv metadata + task definitions
├── Dockerfile               # Container config
└── requirements.txt
```
