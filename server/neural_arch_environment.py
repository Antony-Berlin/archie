"""
NeuralArch-Bench Environment Implementation.

The agent (LLM) acts as a deep learning researcher:
  1. reset()  — randomly picks an architecture and dataset from the libraries,
                loads that architecture, returns starting observation
  2. step()   — accepts either a full model replacement (new_model_code) or an
                incremental layer modification (layer_modification), runs
                trainer.py in a subprocess (timeout=40 s), computes reward,
                returns observation

Reward formula (per project.md):
  +0.1   code compiles and trainer runs without RuntimeError
  +10.0 × (new_acc − old_acc)   performance improvement
  −0.01 × (param_count / 1000)  efficiency penalty
  −1.0   trainer subprocess exited with RuntimeError / timeout
"""

import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Support both package-import and direct-run contexts
try:
    from ..core.models import NeuralArchAction, NeuralArchObservation
    from ..core.arch_library import ARCH_LIBRARY
    from ..core.dataset_library import DATASET_LIBRARY
    from ..core.layer_modifier import apply_layer_modification
except ImportError:
    from core.models import NeuralArchAction, NeuralArchObservation
    from core.arch_library import ARCH_LIBRARY
    from core.dataset_library import DATASET_LIBRARY
    from core.layer_modifier import apply_layer_modification

# Resolve paths relative to the *repo root* (one level up from server/)
_REPO_ROOT = Path(__file__).parent.parent
_TRAINER = _REPO_ROOT / "core" / "trainer.py"


class NeuralArchEnvironment(Environment):
    """
    Neural Architecture Search environment for the NeuralArch-Bench hackathon.

    Each episode randomly picks an architecture from the library and a dataset
    to train on. The agent improves the architecture step-by-step; each step
    triggers a real training run and produces a reward based on accuracy delta
    and parameter efficiency.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._workdir: Path | None = None
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._last_accuracy: float = 0.0
        self._current_code: str = ""
        self._current_arch_name: str = "unknown"
        self._current_dataset_name: str = "fashion_mnist"

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, episode_id=None, **kwargs) -> NeuralArchObservation:
        """Randomly pick an architecture and dataset; return starting observation."""
        self._cleanup()

        self._workdir = Path(tempfile.mkdtemp(prefix="neural_arch_"))
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._last_accuracy = 0.0

        # Reproducible random selection when seed is provided
        rng = random.Random(seed)
        self._current_arch_name = rng.choice(list(ARCH_LIBRARY.keys()))
        self._current_dataset_name = rng.choice(list(DATASET_LIBRARY.keys()))

        self._current_code = ARCH_LIBRARY[self._current_arch_name]
        model_dest = self._workdir / "model_to_edit.py"
        model_dest.write_text(self._current_code)

        return NeuralArchObservation(
            current_code=self._current_code,
            architecture_name=self._current_arch_name,
            dataset_name=self._current_dataset_name,
            last_accuracy=0.0,
            param_count=0,
            loss_curve=[],
            error_logs=None,
            done=False,
            reward=0.0,
        )

    def step(self, action: NeuralArchAction, timeout_s=None, **kwargs) -> NeuralArchObservation:
        """Apply action, run trainer subprocess, return observation + reward."""
        if self._workdir is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")

        self._state.step_count += 1
        model_file = self._workdir / "model_to_edit.py"
        results_file = self._workdir / "results.json"

        # Determine new architecture code and update arch name
        modifier_error: str | None = None

        if action.new_model_code:
            new_code = action.new_model_code
            self._current_arch_name = "custom"
        elif action.layer_modification:
            new_code, success = apply_layer_modification(
                self._current_code, action.layer_modification
            )
            if not success:
                modifier_error = (
                    f"[layer_modifier] Could not parse instruction: "
                    f"{action.layer_modification!r}. Proceeding with unchanged code."
                )
        else:
            # Validator prevents this, but be defensive
            new_code = self._current_code

        model_file.write_text(new_code)
        self._current_code = new_code

        # Run trainer in isolation, passing the chosen dataset
        proc = subprocess.run(
            [
                sys.executable, str(_TRAINER),
                "--model-file", str(model_file),
                "--results-file", str(results_file),
                "--dataset", self._current_dataset_name,
            ],
            timeout=40,
            capture_output=True,
            text=True,
        )

        # Parse results
        accuracy = 0.0
        param_count = 0
        loss_curve: list[float] = []
        error_logs: str | None = modifier_error
        runtime_error = False

        if results_file.exists():
            try:
                data = json.loads(results_file.read_text())
                accuracy = float(data.get("accuracy", 0.0))
                param_count = int(data.get("param_count", 0))
                loss_curve = data.get("loss_curve", [])
                error_raw = data.get("error")
                if error_raw:
                    trainer_error = error_raw
                    runtime_error = "RuntimeError" in error_raw
                    error_logs = (
                        (error_logs + "\n" + trainer_error) if error_logs else trainer_error
                    )
            except (json.JSONDecodeError, ValueError):
                trainer_error = "Failed to parse results.json"
                runtime_error = True
                error_logs = (error_logs + "\n" + trainer_error) if error_logs else trainer_error
        else:
            trainer_error = proc.stderr or "Trainer produced no output"
            runtime_error = True
            error_logs = (error_logs + "\n" + trainer_error) if error_logs else trainer_error

        reward = self._compute_reward(accuracy, param_count, runtime_error)
        self._last_accuracy = accuracy

        return NeuralArchObservation(
            current_code=self._current_code,
            architecture_name=self._current_arch_name,
            dataset_name=self._current_dataset_name,
            last_accuracy=accuracy,
            param_count=param_count,
            loss_curve=loss_curve,
            error_logs=error_logs,
            done=False,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        self._cleanup()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_reward(self, new_acc: float, param_count: int, runtime_error: bool) -> float:
        if runtime_error:
            return -1.0
        reward = 0.1  # execution reward
        reward += 10.0 * (new_acc - self._last_accuracy)  # performance delta
        reward -= 0.01 * (param_count / 1000)              # efficiency penalty
        return round(reward, 4)

    def _cleanup(self) -> None:
        if self._workdir and self._workdir.exists():
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = None
