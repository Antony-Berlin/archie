# NeuralArch-Bench: Environment Design Document

## Overview

**NeuralArch-Bench** is a Reinforcement Learning environment built on the OpenEnv SDK. An LLM agent acts as a *Deep Learning Researcher* that iteratively diagnoses, plans, and implements improvements to PyTorch neural network architectures for tabular classification tasks. Each episode follows a structured 3-phase cycle (Diagnose → Plan → Implement), repeated for a configurable number of rounds.

---

## Environment Interface

**Class:** `NeuralArchEnvironment` (`server/neural_arch_environment.py`)

Inherits from OpenEnv's `Environment` base class.

| Method | Description |
|--------|-------------|
| `reset(seed, episode_id)` | Randomly picks an architecture and dataset; returns the initial observation |
| `step(action)` | Dispatches to the phase handler matching `current_phase`; advances phase state |
| `close()` | Cleans up the temporary working directory |

**Concurrency:** `SUPPORTS_CONCURRENT_SESSIONS = True` — each episode runs in its own `tempfile.mkdtemp()` directory.

---

## Episode Lifecycle

1. `reset()` is called — a random architecture and dataset are selected; phase resets to `diagnose`.
2. The agent calls `step()` three times per cycle:
   - **Step 1 — DIAGNOSE:** Submit a diagnosis text; receive heuristic reward.
   - **Step 2 — PLAN:** Submit a change plan; receive heuristic reward.
   - **Step 3 — IMPLEMENT:** Submit new model Python code; training runs; receive accuracy-based reward.
3. After IMPLEMENT, the cycle resets to DIAGNOSE. This repeats for `MAX_CYCLES` (default: 2), giving 6 total steps.
4. The final score is `clamp(sum(rewards) / max_possible_reward, 0, 1)`.

---

## Action Space

**Class:** `NeuralArchAction` (`core/models.py`)

Exactly one field should be non-null per step, matching the current phase.

| Field | Type | Phase | Description |
|-------|------|-------|-------------|
| `diagnosis` | `Optional[str]` | DIAGNOSE | Text explaining what is wrong with the current model and why |
| `change_plan` | `Optional[str]` | PLAN | Text describing the specific architectural change to make |
| `new_model_code` | `Optional[str]` | IMPLEMENT | Full Python source string defining `ArchModel(nn.Module)` |

**Rules for `new_model_code`:**
- Class must be named `ArchModel` and inherit from `nn.Module`
- Must implement `__init__` and `forward`
- Use `NUM_FEATURES` as input dim and `NUM_CLASSES` as output dim (constants injected at runtime by the trainer)
- Tabular data only: use 1D layers (`Linear`, `BatchNorm1d`, `Dropout`, etc.)
- No image-processing layers (`Conv2d`, etc.)
- No markdown fences — raw Python only

---

## Observation Space

**Class:** `NeuralArchObservation` (`core/models.py`)

Returned by both `reset()` and `step()`.

| Field | Type | Description |
|-------|------|-------------|
| `current_code` | `str` | Full Python source of the current model |
| `architecture_name` | `str` | Name from arch library, or `"custom"` after agent edits |
| `dataset_name` | `str` | Dataset being used this episode (`iris`, `wine`, `breast_cancer`) |
| `last_accuracy` | `float` | Test accuracy from the most recent training run (0.0–1.0) |
| `param_count` | `int` | Total trainable parameter count |
| `loss_curve` | `List[float]` | Loss values from last 5 training epochs |
| `error_logs` | `Optional[str]` | Errors from the last training run (None if clean) |
| `current_phase` | `str` | Active phase: `"diagnose"` \| `"plan"` \| `"implement"` |
| `last_diagnosis` | `Optional[str]` | Agent's diagnosis text from phase 1 of the current cycle |
| `last_plan` | `Optional[str]` | Agent's change plan from phase 2 of the current cycle |
| `phase_rewards` | `List[float]` | Rewards earned so far in the current cycle |
| `reward` | `float` | Reward for the step that produced this observation |
| `done` | `bool` | Always `False` (episode termination controlled externally by step count) |

---

## Reward Model

The environment uses three distinct reward functions, one per phase.

### Phase 1 — DIAGNOSE Reward (max 0.7)

Heuristic scoring of the diagnosis text against the current model code.

| Condition | Reward |
|-----------|--------|
| Mentions `accuracy` or `performance` | +0.3 |
| Mentions `overfitting`, `underfitting`, `generalization`, or `generalisation` | +0.2 |
| References a specific layer variable name from the current code (e.g. `fc1`, `bn2`) | +0.1 |
| Diagnosis is longer than 100 characters | +0.1 |
| Diagnosis is fewer than 10 characters | 0.0 (short-circuit) |

### Phase 2 — PLAN Reward (max 0.7)

Heuristic scoring of the plan text for specificity and coherence with the diagnosis.

| Condition | Reward |
|-----------|--------|
| Mentions a layer type: `batchnorm`, `dropout`, `linear`, `relu`, `sigmoid`, `tanh`, `layernorm`, `gelu`, `conv`, `embedding` | +0.3 |
| Contains a reasoning word: `improve`, `expect`, `because`, `reduce`, `increase`, `prevent`, `help`, `avoid`, `allow`, `should`, `will` | +0.2 |
| Shares ≥3 non-stopword tokens with the diagnosis | +0.1 |
| Plan is longer than 80 characters | +0.1 |
| Empty plan | 0.0 |

### Phase 3 — IMPLEMENT Reward

Computed after a real training run.

| Condition | Reward |
|-----------|--------|
| Code trains without error | +0.1 |
| Accuracy improvement | +10.0 × (new_accuracy − old_accuracy) |
| Parameter count penalty | −0.01 × (param_count / 1000) |
| `RuntimeError` during training | −1.0 (and replaces other components) |
| No code provided | −1.0 |

**Formula:**

```
reward = 0.1 + 10.0 × Δacc − 0.01 × (params / 1000)   [if no error]
reward = −1.0                                             [if RuntimeError]
```

### Episode Score

```
max_reward = MAX_CYCLES × (0.7 + 0.7 + 10.1)   # ≈ 22.6 for 2 cycles
score = clamp(sum(all_rewards) / max_reward, 0.0, 1.0)
success = score >= 0.5
```

---

## Architecture Library

Five starting architectures are available (`core/arch_library.py`). One is randomly selected per episode.

| Name | Layers | Hidden Dims | Notes |
|------|--------|-------------|-------|
| `tabular_simple_mlp` | 3 Linear + ReLU | 64 → 32 | Baseline; no BatchNorm/Dropout |
| `tabular_batch_norm_mlp` | 3 Linear + BatchNorm1d + ReLU | 128 → 64 | Normalized training |
| `tabular_dropout_mlp` | 3 Linear + Dropout(0.3) + ReLU | 128 → 64 | Regularized |
| `tabular_deep_mlp` | 5 Linear + ReLU | 256 → 128 → 64 → 32 | Intentionally deep; invites overfitting diagnosis |
| `tabular_minimal` | 2 Linear + ReLU | 8 | Intentionally under-parameterized; invites underfitting diagnosis |

All architectures reference module-level constants `NUM_FEATURES` and `NUM_CLASSES`, which are injected by the trainer before loading.

---

## Dataset Library

Three tabular CSV datasets are supported (`core/dataset_library.py`). One is randomly selected per episode.

| Name | Features | Classes | Notes |
|------|----------|---------|-------|
| `iris` | 4 | 3 | Fisher's Iris; ~150 samples |
| `wine` | 13 | 3 | Wine recognition; ~178 samples |
| `breast_cancer` | 30 | 2 | Wisconsin Breast Cancer; ~569 samples |

Data files live in `data/`. All features are standardized with `StandardScaler` before training.

---

## Training Subprocess

**File:** `core/trainer.py`

The trainer is **never imported directly** — it is always launched as a subprocess by the environment. This isolates the main process from crashes caused by invalid agent code.

```
subprocess.run(
    [sys.executable, "core/trainer.py",
     "--model-file", model_to_edit.py,
     "--results-file", results.json,
     "--dataset", dataset_name],
    timeout=60,
    capture_output=True,
)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 30 |
| Batch size | 32 |
| Learning rate | 1e-3 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Train/test split | 80/20, stratified, `random_state=42` |
| Device | CUDA if available, else CPU |
| Subprocess timeout | 60 seconds |

### Trainer Output

The trainer writes `results.json` to the working directory:

```json
{
  "accuracy": 0.9333,
  "param_count": 4419,
  "loss_curve": [0.8, 0.5, 0.3, 0.2, 0.18],
  "error": null
}
```

`loss_curve` contains the **last 5 epochs** of average batch loss. On error, `error` is set to the traceback string and `accuracy` remains 0.0.

### Constant Injection

Before loading the model, the trainer prepends the dataset-specific constants if not already present:

```python
NUM_FEATURES = <n>
NUM_CLASSES = <k>
```

This allows agent-written code to reference these names without hardcoding values.

---

## Tasks

Defined in `openenv.yaml`. The active task is selected via the `MY_ENV_V4_TASK` environment variable.

| Task ID | Difficulty | Success Criteria |
|---------|-----------|-----------------|
| `arch-foundations` | Easy | Accuracy > 85% in < 2 steps |
| `efficient-net` | Medium | Accuracy > 80% with < 10k parameters |
| `residual-depth` | Hard | Accuracy > 75% on a deep-net-hostile dataset |

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | LLM model identifier |
| `HF_TOKEN` / `API_KEY` | — | Auth key for LLM API |
| `MY_ENV_V4_TASK` | `arch-foundations` | Task name passed to evaluator |
| `MY_ENV_V4_BENCHMARK` | `neural-arch-bench` | Benchmark name |
| `IMAGE_NAME` | — | Docker image name (triggers `from_docker_image` init) |

---

## Inference Loop

**File:** `inference.py`

The evaluator runs `MAX_CYCLES = 2` full DPI cycles (6 total `env.step()` calls).

```
[START] task=arch-foundations env=neural-arch-bench model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action="..." reward=0.70 done=false error=null
[STEP]  step=2 action="..." reward=0.60 done=false error=null
[STEP]  step=3 action="..." reward=9.85 done=false error=null
[STEP]  step=4 action="..." reward=0.70 done=false error=null
[STEP]  step=5 action="..." reward=0.60 done=false error=null
[STEP]  step=6 action="..." reward=1.20 done=false error=null
[END]   success=true steps=6 score=0.597 rewards=0.70,0.60,9.85,0.70,0.60,1.20
```

### LLM System Prompts

Each phase uses a dedicated system prompt that includes the reward signals to enable reward-aware generation:

- **DIAGNOSE prompt:** instructs the model to analyze architecture, accuracy, and loss curve; identifies reward criteria
- **PLAN prompt:** instructs the model to describe a specific architectural change; references diagnosis
- **IMPLEMENT prompt:** instructs the model to write the complete `ArchModel` class; strict tabular constraints

---

## File Structure

```
scaler-hack/
├── core/
│   ├── arch_library.py          # 5 starting architectures (source strings)
│   ├── dataset_library.py       # Dataset configs (iris, wine, breast_cancer)
│   ├── models.py                # NeuralArchAction + NeuralArchObservation schemas
│   ├── trainer.py               # Subprocess training script; writes results.json
│   └── template_model.py        # Initial model template
├── server/
│   ├── app.py                   # FastAPI create_app() entrypoint
│   └── neural_arch_environment.py  # reset() / step() / reward logic
├── data/
│   ├── iris.csv
│   ├── wine.csv
│   └── breast_cancer.csv
├── docs/
│   ├── CLAUDE.md
│   └── project.md
├── scripts/
│   └── pre_validation.sh
├── inference.py                 # LLM agent evaluation loop
├── sample_inference.py          # Reference inference template
├── Dockerfile
├── openenv.yaml
├── requirements.txt
└── pyproject.toml
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 3-phase cycle instead of single-action steps | Decomposes the task into reasoning (diagnose/plan) and execution (implement); enables partial credit and GRPO-friendly dense rewards |
| Subprocess isolation for training | Prevents the server process from crashing on invalid agent code (infinite loops, shape mismatches, syntax errors) |
| Tabular datasets instead of image datasets | Trains in seconds (30 epochs on ≤600 rows), enabling multiple iterations within a single episode |
| Heuristic rewards for diagnose/plan | Provides signal proportional to reasoning quality without requiring ground-truth labels |
| `NUM_FEATURES` / `NUM_CLASSES` injection | Allows the agent to write dataset-agnostic code; the trainer patches the constants before loading |
| Random arch + dataset selection per episode | Prevents memorization; forces generalization across problem setups |
