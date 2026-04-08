"""
NeuralArch-Bench Environment Implementation.

The agent runs a 3-phase cycle per turn:
  1. DIAGNOSE  — explains what is wrong with the current model (heuristic reward)
  2. PLAN      — describes the architectural change to make (heuristic reward)
  3. IMPLEMENT — writes new model code; training runs (accuracy-based reward)

Reward formulas:
  DIAGNOSE:  0–0.7 heuristic score based on diagnosis quality
  PLAN:      0–0.7 heuristic score based on plan specificity and coherence
  IMPLEMENT: +0.1 execution  +10.0×Δacc  −0.01×(params/1000)  −1.0 on error
"""

import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..core.models import NeuralArchAction, NeuralArchObservation
    from ..core.arch_library import ARCH_LIBRARY
    from ..core.dataset_library import DATASET_LIBRARY
    from ..core.task_graders import EpisodeResult, get_grader, TASK_IDS
except ImportError:
    from core.models import NeuralArchAction, NeuralArchObservation
    from core.arch_library import ARCH_LIBRARY
    from core.dataset_library import DATASET_LIBRARY
    from core.task_graders import EpisodeResult, get_grader, TASK_IDS

_REPO_ROOT = Path(__file__).parent.parent
_TRAINER = _REPO_ROOT / "core" / "trainer.py"


class NeuralArchEnvironment(Environment):
    """Neural Architecture Search environment with 3-phase step protocol."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._workdir: Optional[Path] = None
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._last_accuracy: float = 0.0
        self._current_code: str = ""
        self._current_arch_name: str = "unknown"
        self._current_dataset_name: str = "iris"
        # Phase state
        self._current_phase: str = "diagnose"
        self._last_diagnosis: Optional[str] = None
        self._last_plan: Optional[str] = None
        self._phase_rewards: list[float] = []
        # Task grading
        _task_id = os.getenv("MY_ENV_V4_TASK", "arch-foundations")
        if _task_id not in TASK_IDS:
            _task_id = "arch-foundations"
        self._task_id: str = _task_id
        self._grader = get_grader(self._task_id)
        self._implement_steps: int = 0
        self._task_score: float = 0.0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, episode_id=None, **kwargs) -> NeuralArchObservation:
        """Randomly pick an architecture and dataset; return starting observation."""
        self._cleanup()

        self._workdir = Path(tempfile.mkdtemp(prefix="neural_arch_"))
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._last_accuracy = 0.0

        rng = random.Random(seed)
        self._current_arch_name = rng.choice(list(ARCH_LIBRARY.keys()))
        self._current_dataset_name = rng.choice(list(DATASET_LIBRARY.keys()))
        self._current_code = ARCH_LIBRARY[self._current_arch_name]
        (self._workdir / "model_to_edit.py").write_text(self._current_code)

        # Reset phase state
        self._current_phase = "diagnose"
        self._last_diagnosis = None
        self._last_plan = None
        self._phase_rewards = []
        # Reset task grading state
        self._implement_steps = 0
        self._task_score = 0.0

        return self._build_obs(reward=0.0)

    def step(self, action: NeuralArchAction, timeout_s=None, **kwargs) -> NeuralArchObservation:
        """Dispatch to the appropriate phase handler."""
        if self._workdir is None:
            raise RuntimeError("Call reset() before step().")
        self._state.step_count += 1

        if self._current_phase == "diagnose":
            return self._step_diagnose(action)
        elif self._current_phase == "plan":
            return self._step_plan(action)
        else:
            return self._step_implement(action)

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        self._cleanup()

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _step_diagnose(self, action: NeuralArchAction) -> NeuralArchObservation:
        diagnosis = action.diagnosis or ""
        reward = self._score_diagnosis(diagnosis, self._current_code)
        self._last_diagnosis = diagnosis
        self._phase_rewards = [reward]
        self._current_phase = "plan"
        return self._build_obs(reward=reward)

    def _step_plan(self, action: NeuralArchAction) -> NeuralArchObservation:
        plan = action.change_plan or ""
        reward = self._score_plan(plan, self._last_diagnosis or "")
        self._last_plan = plan
        self._phase_rewards.append(reward)
        self._current_phase = "implement"
        return self._build_obs(reward=reward)

    def _step_implement(self, action: NeuralArchAction) -> NeuralArchObservation:
        new_code = action.new_model_code or ""
        if not new_code:
            reward = -1.0
            self._phase_rewards.append(reward)
            self._reset_cycle()
            return self._build_obs(reward=reward, error_logs="No model code provided in implement phase.")

        model_file = self._workdir / "model_to_edit.py"
        results_file = self._workdir / "results.json"
        if results_file.exists():
            results_file.unlink()

        model_file.write_text(new_code)
        self._current_code = new_code
        self._current_arch_name = "custom"

        proc = subprocess.run(
            [sys.executable, str(_TRAINER),
             "--model-file", str(model_file),
             "--results-file", str(results_file),
             "--dataset", self._current_dataset_name],
            timeout=60,
            capture_output=True,
            text=True,
        )

        accuracy, param_count, loss_curve, error_logs, runtime_error = self._parse_results(
            results_file, proc
        )
        reward = self._compute_reward(accuracy, param_count, runtime_error)
        self._last_accuracy = accuracy
        self._phase_rewards.append(reward)
        self._implement_steps += 1

        # Grade against the active task
        ep_result = EpisodeResult(
            accuracy=accuracy,
            param_count=param_count,
            implement_steps=self._implement_steps,
            dataset_name=self._current_dataset_name,
            rewards=list(self._phase_rewards),
            error_logs=error_logs,
        )
        self._task_score = self._grader.grade(ep_result)
        task_done = self._grader.is_success(ep_result)

        cycle_rewards = list(self._phase_rewards)
        self._reset_cycle()

        return self._build_obs(
            reward=reward,
            error_logs=error_logs,
            accuracy=accuracy,
            param_count=param_count,
            loss_curve=loss_curve,
            phase_rewards_override=cycle_rewards,
            done=task_done,
        )

    # ------------------------------------------------------------------
    # Reward scorers
    # ------------------------------------------------------------------

    def _score_diagnosis(self, diagnosis: str, code: str) -> float:
        if len(diagnosis) < 10:
            return 0.0
        score = 0.0
        low = diagnosis.lower()
        if "accuracy" in low or "performance" in low:
            score += 0.3
        if any(w in low for w in ("overfitting", "underfitting", "generalization", "generalisation")):
            score += 0.2
        layer_names = re.findall(r'self\.(\w+)', code)
        if any(name in diagnosis for name in layer_names):
            score += 0.1
        if len(diagnosis) > 100:
            score += 0.1
        return round(score, 4)

    def _score_plan(self, plan: str, diagnosis: str) -> float:
        if not plan:
            return 0.0
        score = 0.0
        low = plan.lower()
        layer_keywords = ("batchnorm", "dropout", "linear", "relu", "sigmoid",
                          "tanh", "layernorm", "gelu", "conv", "embedding")
        if any(kw in low for kw in layer_keywords):
            score += 0.3
        reasoning_words = ("improve", "expect", "because", "reduce", "increase",
                           "prevent", "help", "avoid", "allow", "should", "will")
        if any(w in low for w in reasoning_words):
            score += 0.2
        if diagnosis:
            stopwords = {"the", "a", "an", "is", "to", "of", "and", "in", "it", "this"}
            diag_words = set(diagnosis.lower().split()) - stopwords
            plan_words = set(plan.lower().split()) - stopwords
            if len(diag_words & plan_words) >= 3:
                score += 0.1
        if len(plan) > 80:
            score += 0.1
        return round(score, 4)

    def _compute_reward(self, new_acc: float, param_count: int, runtime_error: bool) -> float:
        if runtime_error:
            return -1.0
        reward = 0.1
        reward += 10.0 * (new_acc - self._last_accuracy)
        reward -= 0.01 * (param_count / 1000)
        return round(reward, 4)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_results(
        self, results_file: Path, proc
    ) -> tuple[float, int, list, Optional[str], bool]:
        accuracy, param_count, loss_curve = 0.0, 0, []
        error_logs: Optional[str] = None
        runtime_error = False

        if results_file.exists():
            try:
                data = json.loads(results_file.read_text())
                accuracy = float(data.get("accuracy", 0.0))
                param_count = int(data.get("param_count", 0))
                loss_curve = data.get("loss_curve", [])
                error_raw = data.get("error")
                if error_raw:
                    error_logs = error_raw
                    runtime_error = "RuntimeError" in error_raw
            except (json.JSONDecodeError, ValueError):
                error_logs = "Failed to parse results.json"
                runtime_error = True
        else:
            error_logs = proc.stderr or "Trainer produced no output"
            runtime_error = True

        return accuracy, param_count, loss_curve, error_logs, runtime_error

    def _build_obs(
        self,
        reward: float = 0.0,
        error_logs: Optional[str] = None,
        accuracy: Optional[float] = None,
        param_count: int = 0,
        loss_curve: Optional[list] = None,
        phase_rewards_override: Optional[list] = None,
        done: bool = False,
    ) -> NeuralArchObservation:
        return NeuralArchObservation(
            current_code=self._current_code,
            architecture_name=self._current_arch_name,
            dataset_name=self._current_dataset_name,
            last_accuracy=accuracy if accuracy is not None else self._last_accuracy,
            param_count=param_count,
            loss_curve=loss_curve or [],
            error_logs=error_logs,
            done=done,
            reward=reward,
            current_phase=self._current_phase,
            last_diagnosis=self._last_diagnosis,
            last_plan=self._last_plan,
            phase_rewards=phase_rewards_override if phase_rewards_override is not None else list(self._phase_rewards),
            task_id=self._task_id,
            task_score=self._task_score,
        )

    def _reset_cycle(self) -> None:
        """Advance back to diagnose phase for the next cycle."""
        self._current_phase = "diagnose"
        self._last_diagnosis = None
        self._last_plan = None
        self._phase_rewards = []

    def _cleanup(self) -> None:
        if self._workdir and self._workdir.exists():
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = None
