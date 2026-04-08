"""
Run inference for ALL tasks in NeuralArch-Bench with their respective graders.

Tasks:
    arch-foundations  (Easy)   — accuracy > 85% within 2 implement steps
    efficient-net     (Medium) — accuracy > 80% with param_count < 10,000
    residual-depth    (Hard)   — accuracy > 75% on breast_cancer dataset

Usage:
    python run_all_tasks.py
"""

import asyncio
import os
import textwrap
from pathlib import Path
from typing import List, Optional

# Load .env file from repo root if present
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from openai import OpenAI

from core.models import NeuralArchAction as _Action
from core.task_graders import TASK_IDS, get_grader, EpisodeResult
from server.neural_arch_environment import NeuralArchEnvironment as _Env

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "neural-arch-bench")

MAX_CYCLES = 4
MAX_STEPS = MAX_CYCLES * 3
SUCCESS_SCORE_THRESHOLD = 0.5
_MAX_REWARD = MAX_CYCLES * (0.7 + 0.7 + 10.1)

_DATASET_INFO = {
    "iris":          {"num_features": 4,  "num_classes": 3},
    "wine":          {"num_features": 13, "num_classes": 3},
    "breast_cancer": {"num_features": 30, "num_classes": 2},
}

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


def _build_task_system_prompt_implement(task_id: str) -> str:
    """Augment the implement prompt with task-specific guidance."""
    grader = get_grader(task_id)
    task_hint = {
        "arch-foundations": (
            "TASK GOAL: Reach accuracy > 85% as fast as possible (within 2 implement steps). "
            "Prioritize a well-regularized, reliable architecture over novelty."
        ),
        "efficient-net": (
            "TASK GOAL: Achieve accuracy > 80% while keeping param_count UNDER 10,000. "
            "Use small hidden dims (e.g., 32–64 neurons max). "
            "Avoid unnecessary layers. Compact BatchNorm + Dropout MLP preferred."
        ),
        "residual-depth": (
            "TASK GOAL: Achieve accuracy > 75% on breast_cancer (NUM_FEATURES=30, NUM_CLASSES=2). "
            "Use sufficient capacity for 30 input features. Residual/skip connections are allowed. "
            "BatchNorm and Dropout help with generalization on this dataset."
        ),
    }.get(task_id, "")

    return SYSTEM_PROMPT_IMPLEMENT + (f"\n\n{task_hint}" if task_hint else "")


# ── stdout helpers ─────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"\n{'='*70}", flush=True)
    print(f"[START] task={task} env={env} model={model}", flush=True)
    print(f"{'='*70}", flush=True)


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


def log_grader_result(task_id: str, episode_result: EpisodeResult, grader_score: float) -> None:
    grader = get_grader(task_id)
    print(f"\n[GRADER] task={task_id} difficulty={grader.difficulty}", flush=True)
    print(f"[GRADER] accuracy={episode_result.accuracy:.4f} "
          f"param_count={episode_result.param_count} "
          f"implement_steps={episode_result.implement_steps} "
          f"dataset={episode_result.dataset_name}", flush=True)
    print(f"[GRADER] score={grader_score:.4f} "
          f"success_criteria='{grader.success_criteria}'", flush=True)


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


def get_plan(client: OpenAI, obs, task_id: str) -> str:
    grader = get_grader(task_id)
    task_hint = {
        "arch-foundations": "Focus on fast accuracy improvement. You have at most 2 implement steps.",
        "efficient-net": "Keep param_count UNDER 10,000. Use small hidden dims (32–64 max).",
        "residual-depth": "The dataset has 30 features. Ensure sufficient model capacity with residual connections or deep MLP.",
    }.get(task_id, "")

    user = textwrap.dedent(f"""
        Task         : {task_id} ({grader.difficulty}) — {grader.success_criteria}
        {f'Task hint    : {task_hint}' if task_hint else ''}
        Dataset      : {getattr(obs, 'dataset_name', 'iris')} ({_dataset_hint(obs)})
        Last accuracy: {obs.last_accuracy:.4f}
        Param count  : {obs.param_count}
        Your diagnosis: {getattr(obs, 'last_diagnosis', 'N/A')}

        Current model code:
        {obs.current_code}

        Describe the specific architectural change you will implement.
    """).strip()
    return _call_llm(client, SYSTEM_PROMPT_PLAN, user,
                     fallback="Add BatchNorm1d layers after each Linear layer to stabilize training and improve generalization.")


def get_model_code(client: OpenAI, obs, task_id: str) -> str:
    system = _build_task_system_prompt_implement(task_id)
    user = textwrap.dedent(f"""
        Dataset      : {getattr(obs, 'dataset_name', 'iris')} ({_dataset_hint(obs)})
        Last accuracy: {obs.last_accuracy:.4f}
        Param count  : {obs.param_count}
        Your diagnosis: {getattr(obs, 'last_diagnosis', 'N/A')}
        Your plan    : {getattr(obs, 'last_plan', 'N/A')}

        Current model code:
        {obs.current_code}

        Write the improved ArchModel now.
    """).strip()
    raw = _call_llm(client, system, user, fallback=_fallback_code())
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


def _unpack(result):
    obs = getattr(result, "observation", result)
    reward = float(getattr(result, "reward", 0) or getattr(obs, "reward", 0) or 0)
    done = bool(getattr(result, "done", False) or getattr(obs, "done", False))
    return obs, reward, done


# ── single task runner ─────────────────────────────────────────────────────────

async def run_task(task_id: str, client: OpenAI) -> dict:
    """Run a full episode for one task. Returns summary dict."""
    # Each task gets a fresh environment instance with the correct task_id set
    os.environ["MY_ENV_V4_TASK"] = task_id
    env = _Env()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    implement_steps = 0
    final_obs = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset() if asyncio.iscoroutinefunction(env.reset) else env.reset()
        obs = getattr(result, "observation", result)
        final_obs = obs

        for cycle in range(1, MAX_CYCLES + 1):

            # ── Phase 1: DIAGNOSE ──────────────────────────────────────
            step_num = (cycle - 1) * 3 + 1
            diagnosis = get_diagnosis(client, obs)
            result = env.step(_Action(diagnosis=diagnosis))
            obs, reward, done = _unpack(result)
            final_obs = obs
            rewards.append(reward)
            steps_taken = step_num
            log_step(step=step_num, action=diagnosis[:60], reward=reward, done=done,
                     error=getattr(obs, "error_logs", None))
            if done:
                break

            # ── Phase 2: PLAN ──────────────────────────────────────────
            step_num = (cycle - 1) * 3 + 2
            plan = get_plan(client, obs, task_id)
            result = env.step(_Action(change_plan=plan))
            obs, reward, done = _unpack(result)
            final_obs = obs
            rewards.append(reward)
            steps_taken = step_num
            log_step(step=step_num, action=plan[:60], reward=reward, done=done,
                     error=getattr(obs, "error_logs", None))
            if done:
                break

            # ── Phase 3: IMPLEMENT ─────────────────────────────────────
            step_num = (cycle - 1) * 3 + 3
            implement_steps += 1
            code = get_model_code(client, obs, task_id)
            result = env.step(_Action(new_model_code=code))
            obs, reward, done = _unpack(result)
            final_obs = obs
            rewards.append(reward)
            steps_taken = step_num
            log_step(step=step_num, action=code[:60], reward=reward, done=done,
                     error=getattr(obs, "error_logs", None))
            if done:
                break

        score = min(max(sum(rewards) / _MAX_REWARD, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

        # ── Grader evaluation ──────────────────────────────────────────
        if final_obs is not None:
            episode_result = EpisodeResult(
                accuracy=getattr(final_obs, "last_accuracy", 0.0),
                param_count=getattr(final_obs, "param_count", 0),
                implement_steps=implement_steps,
                dataset_name=getattr(final_obs, "dataset_name", "iris"),
                rewards=rewards,
                error_logs=getattr(final_obs, "error_logs", None),
            )
            grader = get_grader(task_id)
            grader_score = grader.grade(episode_result)
            log_grader_result(task_id, episode_result, grader_score)
        else:
            grader_score = 0.0
            episode_result = None

    finally:
        try:
            if asyncio.iscoroutinefunction(env.close):
                await env.close()
            else:
                env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "grader_score": grader_score if final_obs is not None else 0.0,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
    }


# ── main: run all tasks ────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"\n{'#'*70}", flush=True)
    print(f"# NeuralArch-Bench — Running ALL tasks", flush=True)
    print(f"# Tasks: {TASK_IDS}", flush=True)
    print(f"# Model: {MODEL_NAME}", flush=True)
    print(f"{'#'*70}\n", flush=True)

    all_results = []
    for task_id in TASK_IDS:
        result = await run_task(task_id, client)
        all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'#'*70}", flush=True)
    print(f"# SUMMARY", flush=True)
    print(f"{'#'*70}", flush=True)
    print(f"{'Task':<20} {'Difficulty':<12} {'Score':>8} {'GraderScore':>12} {'Success':>8} {'Steps':>6}", flush=True)
    print(f"{'-'*70}", flush=True)

    difficulties = {"arch-foundations": "easy", "efficient-net": "medium", "residual-depth": "hard"}
    for r in all_results:
        diff = difficulties.get(r["task_id"], "?")
        print(
            f"{r['task_id']:<20} {diff:<12} {r['score']:>8.3f} {r['grader_score']:>12.4f} "
            f"{'yes' if r['success'] else 'no':>8} {r['steps']:>6}",
            flush=True,
        )

    overall = sum(r["grader_score"] for r in all_results) / len(all_results)
    print(f"{'-'*70}", flush=True)
    print(f"{'OVERALL AVERAGE':<20} {'':12} {'':>8} {overall:>12.4f}", flush=True)
    print(f"{'#'*70}\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
