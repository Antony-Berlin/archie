"""
Inference script for NeuralArch-Bench (hackathon evaluator).

Runs a 3-phase cycle: DIAGNOSE → PLAN → IMPLEMENT, repeated MAX_CYCLES times.
Each phase calls the LLM with a phase-specific prompt and submits one env.step().

Required environment variables:
    API_BASE_URL       LLM endpoint   (default: https://router.huggingface.co/v1)
    MODEL_NAME         Model ID       (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN / API_KEY Auth key
    MY_ENV_V4_TASK     Task name      (default: arch-foundations)
    MY_ENV_V4_BENCHMARK Benchmark    (default: neural-arch-bench)

Stdout format (mandatory):
    [START] task=<name> env=<bench> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<bool> error=<msg|null>
    [END]   success=<bool> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

try:
    from my_env_v4 import MyEnvV4Action, MyEnvV4Env  # type: ignore[import]
    _Action = MyEnvV4Action
    _Env = MyEnvV4Env
except ImportError:
    from core.models import NeuralArchAction as _Action  # type: ignore[assignment]
    from server.neural_arch_environment import NeuralArchEnvironment as _Env  # type: ignore[assignment]

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "arch-foundations")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "neural-arch-bench")

MAX_CYCLES = 2           # 2 full DPI cycles = 6 total env.step() calls
MAX_STEPS = MAX_CYCLES * 3
SUCCESS_SCORE_THRESHOLD = 0.5

# Max reward: per cycle = 0.7 (diagnose) + 0.7 (plan) + 10.1 (implement)
_MAX_REWARD = MAX_CYCLES * (0.7 + 0.7 + 10.1)

# ── dataset hint lookup ────────────────────────────────────────────────────────
_DATASET_INFO = {
    "iris":          {"num_features": 4,  "num_classes": 3},
    "wine":          {"num_features": 13, "num_classes": 3},
    "breast_cancer": {"num_features": 30, "num_classes": 2},
}

# ── system prompts ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_DIAGNOSE = textwrap.dedent("""
    You are a deep learning researcher diagnosing a PyTorch model for tabular classification.
    Analyze the current model architecture, accuracy, and loss curve.
    Identify the primary reason the model is not performing optimally.

    Two module-level constants are available in the model at runtime:
        NUM_FEATURES  — number of input features
        NUM_CLASSES   — number of output classes

    Reward signals for this phase:
      +0.3  mention accuracy or performance issues
      +0.2  mention overfitting, underfitting, or generalization
      +0.1  mention specific layer names from the current code
      +0.1  write more than 100 characters

    Reply with ONLY your diagnosis as plain text. No code, no preamble.
""").strip()

SYSTEM_PROMPT_PLAN = textwrap.dedent("""
    You are a deep learning researcher planning an architectural improvement.
    Based on your diagnosis, describe the specific change you will make to the model.

    Reward signals for this phase:
      +0.3  mention specific layer types (BatchNorm, Dropout, Linear, ReLU, etc.)
      +0.2  explain why you expect improvement
      +0.1  reference your diagnosis (shared words)
      +0.1  write more than 80 characters

    Reply with ONLY your plan as plain text. No code, no preamble.
""").strip()

SYSTEM_PROMPT_IMPLEMENT = textwrap.dedent("""
    You are a deep learning researcher implementing a PyTorch model for tabular data.
    Write the complete updated Python module defining ArchModel.

    Module-level constants injected at runtime:
        NUM_FEATURES  — number of input features
        NUM_CLASSES   — number of output classes

    Rules:
      - Class named ArchModel inheriting from nn.Module
      - Implement __init__ and forward
      - Use NUM_FEATURES as input dim, NUM_CLASSES as output dim
      - Tabular data only: use 1D layers (Linear, BatchNorm1d, Dropout, etc.)
      - No Conv2d or image-processing layers

    Reward signals:
      +0.1   valid code that trains without error
      +10.0 × accuracy improvement over previous run
      -0.01 × (param_count / 1000) efficiency penalty
      -1.0   RuntimeError during training

    Reply with ONLY the Python code. No markdown fences, no explanation.
""").strip()


# ── stdout helpers ─────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action!r:.80} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM helpers ────────────────────────────────────────────────────────────────

def _call_llm(client: OpenAI, system: str, user: str, fallback: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.7,
            max_tokens=1024,
            stream=False,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return fallback


def _dataset_hint(obs) -> str:
    name = getattr(obs, "dataset_name", "iris")
    info = _DATASET_INFO.get(name, {})
    return f"NUM_FEATURES={info.get('num_features', '?')}, NUM_CLASSES={info.get('num_classes', '?')}"


def get_diagnosis(client: OpenAI, obs) -> str:
    user = textwrap.dedent(f"""
        Architecture : {getattr(obs, 'architecture_name', 'unknown')}
        Dataset      : {getattr(obs, 'dataset_name', 'iris')} ({_dataset_hint(obs)})
        Last accuracy: {obs.last_accuracy:.4f}
        Param count  : {obs.param_count}
        Loss curve   : {obs.loss_curve}
        Errors       : {obs.error_logs or 'none'}
        Phase rewards so far: {getattr(obs, 'phase_rewards', [])}

        Current model code:
        {obs.current_code}

        Diagnose why this model is not performing optimally.
    """).strip()
    return _call_llm(client, SYSTEM_PROMPT_DIAGNOSE, user,
                     fallback="The model may be underfitting due to insufficient capacity or missing regularization.")


def get_plan(client: OpenAI, obs) -> str:
    user = textwrap.dedent(f"""
        Dataset      : {getattr(obs, 'dataset_name', 'iris')} ({_dataset_hint(obs)})
        Last accuracy: {obs.last_accuracy:.4f}
        Your diagnosis: {getattr(obs, 'last_diagnosis', 'N/A')}

        Current model code:
        {obs.current_code}

        Describe the specific architectural change you will implement.
    """).strip()
    return _call_llm(client, SYSTEM_PROMPT_PLAN, user,
                     fallback="Add BatchNorm1d layers after each Linear layer to stabilize training and improve generalization.")


def get_model_code(client: OpenAI, obs) -> str:
    user = textwrap.dedent(f"""
        Dataset      : {getattr(obs, 'dataset_name', 'iris')} ({_dataset_hint(obs)})
        Last accuracy: {obs.last_accuracy:.4f}
        Your diagnosis: {getattr(obs, 'last_diagnosis', 'N/A')}
        Your plan    : {getattr(obs, 'last_plan', 'N/A')}

        Current model code:
        {obs.current_code}

        Write the improved ArchModel now.
    """).strip()
    raw = _call_llm(client, SYSTEM_PROMPT_IMPLEMENT, user, fallback=_fallback_code())
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    return raw if raw.strip() else _fallback_code()


def _fallback_code() -> str:
    return textwrap.dedent("""
        import torch.nn as nn

        NUM_FEATURES = 4
        NUM_CLASSES = 3

        class ArchModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(NUM_FEATURES, 64)
                self.bn1 = nn.BatchNorm1d(64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, NUM_CLASSES)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.bn1(self.fc1(x)))
                x = self.relu(self.fc2(x))
                return self.fc3(x)
    """).strip()


# ── main loop ──────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if IMAGE_NAME:
        env = await _Env.from_docker_image(IMAGE_NAME)
    else:
        env = _Env()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset() if asyncio.iscoroutinefunction(env.reset) else env.reset()
        obs = getattr(result, "observation", result)

        for cycle in range(1, MAX_CYCLES + 1):

            # ── Phase 1: DIAGNOSE ──────────────────────────────────────
            step_num = (cycle - 1) * 3 + 1
            diagnosis = get_diagnosis(client, obs)
            result = env.step(_Action(diagnosis=diagnosis))
            obs, reward, done = _unpack(result)
            rewards.append(reward)
            steps_taken = step_num
            log_step(step=step_num, action=diagnosis[:60], reward=reward, done=done,
                     error=getattr(obs, "error_logs", None))
            if done:
                break

            # ── Phase 2: PLAN ──────────────────────────────────────────
            step_num = (cycle - 1) * 3 + 2
            plan = get_plan(client, obs)
            result = env.step(_Action(change_plan=plan))
            obs, reward, done = _unpack(result)
            rewards.append(reward)
            steps_taken = step_num
            log_step(step=step_num, action=plan[:60], reward=reward, done=done,
                     error=getattr(obs, "error_logs", None))
            if done:
                break

            # ── Phase 3: IMPLEMENT ─────────────────────────────────────
            step_num = (cycle - 1) * 3 + 3
            code = get_model_code(client, obs)
            result = env.step(_Action(new_model_code=code))
            obs, reward, done = _unpack(result)
            rewards.append(reward)
            steps_taken = step_num
            log_step(step=step_num, action=code[:60], reward=reward, done=done,
                     error=getattr(obs, "error_logs", None))
            if done:
                break

        score = min(max(sum(rewards) / _MAX_REWARD, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            if asyncio.iscoroutinefunction(env.close):
                await env.close()
            else:
                env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def _unpack(result):
    obs = getattr(result, "observation", result)
    reward = float(getattr(result, "reward", 0) or getattr(obs, "reward", 0) or 0)
    done = bool(getattr(result, "done", False) or getattr(obs, "done", False))
    return obs, reward, done


if __name__ == "__main__":
    asyncio.run(main())
