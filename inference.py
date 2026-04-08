"""
Inference script for NeuralArch-Bench (hackathon evaluator).

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

# The evaluator imports from my_env_v4; we expose our env under that alias.
# For direct execution the imports below are used instead.
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
MAX_STEPS = 6
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
    You are a deep learning researcher improving a PyTorch model architecture.
    Each turn you receive the current model source code, training metrics, and the
    dataset being used this episode.

    You must reply with ONLY a complete, valid Python module containing a class named
    ArchModel that inherits from nn.Module and implements __init__ and forward.
    Do not add any explanation — output only the Python code.

    Two module-level constants are available in your model file at runtime:
        INPUT_CHANNELS  — 1 for grayscale datasets, 3 for RGB (e.g. CIFAR-10)
        INPUT_SIZE      — image side length in pixels (28 or 32)
    Use these constants in your architecture to stay dataset-agnostic.

    Reward signals:
      +0.1  for valid, runnable code
      +10.0 × accuracy improvement over the previous run
      -0.01 × (param_count / 1000) efficiency penalty
      -1.0  if code causes a RuntimeError

    Maximise accuracy while keeping parameter count low.
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


# ── LLM helper ─────────────────────────────────────────────────────────────────

def build_user_prompt(obs) -> str:
    arch_name = getattr(obs, "architecture_name", "unknown")
    dataset = getattr(obs, "dataset_name", "fashion_mnist")
    return textwrap.dedent(f"""
        Architecture : {arch_name}
        Dataset      : {dataset}
        Last accuracy: {obs.last_accuracy:.4f}
        Param count  : {obs.param_count}
        Loss curve   : {obs.loss_curve}
        Errors       : {obs.error_logs or 'none'}

        Current model code:
        ```python
        {obs.current_code}
        ```

        Provide the improved model code now.
    """).strip()


def get_new_model_code(client: OpenAI, obs) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(obs)},
            ],
            temperature=0.7,
            max_tokens=1024,
            stream=False,
        )
        code = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if code.startswith("```"):
            lines = code.splitlines()
            code = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        return code if code else _fallback_code()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return _fallback_code()


def _fallback_code() -> str:
    return textwrap.dedent("""
        import torch.nn as nn
        import torch.nn.functional as F

        class ArchModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(784, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 10)

            def forward(self, x):
                x = x.view(-1, 784)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
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

        # Normalise: handle both raw Observation and ResetResponse wrappers
        obs = getattr(result, "observation", result)

        for step in range(1, MAX_STEPS + 1):
            new_code = get_new_model_code(client, obs)
            thought = f"Step {step}: improving architecture"

            action = _Action(new_model_code=new_code, thought_process=thought)

            if asyncio.iscoroutinefunction(env.step):
                result = await env.step(action)
            else:
                result = env.step(action)

            obs = getattr(result, "observation", result)
            reward = float(getattr(result, "reward", 0) or getattr(obs, "reward", 0) or 0)
            done = bool(getattr(result, "done", False) or getattr(obs, "done", False))
            error = getattr(obs, "error_logs", None)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=new_code[:60], reward=reward, done=done, error=error)

            if done:
                break

        # Score = cumulative reward normalised to [0, 1] using max possible reward
        max_reward = MAX_STEPS * (0.1 + 10.0)  # compile + 100 % accuracy gain each step
        score = min(max(sum(rewards) / max_reward, 0.0), 1.0)
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


if __name__ == "__main__":
    asyncio.run(main())
