"""
Task Graders for NeuralArch-Bench.

Each grader class evaluates a completed episode against task-specific success
criteria and returns a normalized score in [0.0, 1.0].

Tasks:
    arch-foundations  (Easy)   — Reach accuracy > 85% within 2 implement steps
    efficient-net     (Medium) — Reach accuracy > 80% with param_count < 10,000
    residual-depth    (Hard)   — Reach accuracy > 75% on breast_cancer (30 features)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ── episode result snapshot passed to each grader ─────────────────────────────

@dataclass
class EpisodeResult:
    """Summary of a completed episode, extracted from the final observation."""
    accuracy: float
    param_count: int
    implement_steps: int          # number of IMPLEMENT phases executed
    dataset_name: str
    rewards: List[float] = field(default_factory=list)
    error_logs: Optional[str] = None


# ── base class ─────────────────────────────────────────────────────────────────

class BaseTaskGrader:
    """
    Abstract grader. Subclasses implement grade() and expose task metadata.
    All scores are in [0.0, 1.0].
    """

    task_id: str = ""
    difficulty: str = ""
    description: str = ""
    success_criteria: str = ""

    def grade(self, result: EpisodeResult) -> float:
        """Return a score in [0.0, 1.0] for the episode."""
        raise NotImplementedError

    def is_success(self, result: EpisodeResult) -> bool:
        """Convenience wrapper: True if the graded score meets the pass threshold."""
        return self.grade(result) >= 1.0


# ── Task 1: arch-foundations (Easy) ───────────────────────────────────────────

class ArchFoundationsGrader(BaseTaskGrader):
    """
    Easy task: demonstrate that the agent can quickly improve any architecture.

    Success: final accuracy > 85% AND achieved within the first 2 implement steps.
    Partial credit: proportional to accuracy, capped if step budget is exceeded.
    """

    task_id = "arch-foundations"
    difficulty = "easy"
    description = (
        "Improve any starting architecture to achieve > 85% accuracy "
        "within 2 implementation attempts."
    )
    success_criteria = "accuracy > 0.85 AND implement_steps <= 2"

    ACC_TARGET = 0.85
    STEP_BUDGET = 2

    def grade(self, result: EpisodeResult) -> float:
        acc_score = min(result.accuracy / self.ACC_TARGET, 1.0)

        # Full credit only if within step budget
        if result.implement_steps <= self.STEP_BUDGET:
            return round(acc_score, 4)

        # Exceeded budget: penalise by 50%
        return round(acc_score * 0.5, 4)


# ── Task 2: efficient-net (Medium) ────────────────────────────────────────────

class EfficientNetGrader(BaseTaskGrader):
    """
    Medium task: build an accurate yet compact model.

    Success: accuracy > 80% AND param_count < 10,000.
    Partial credit: weighted combination of accuracy and efficiency sub-scores.
    """

    task_id = "efficient-net"
    difficulty = "medium"
    description = (
        "Achieve > 80% accuracy while keeping the model under 10,000 parameters."
    )
    success_criteria = "accuracy > 0.80 AND param_count < 10,000"

    ACC_TARGET = 0.80
    PARAM_BUDGET = 10_000

    def grade(self, result: EpisodeResult) -> float:
        acc_score = min(result.accuracy / self.ACC_TARGET, 1.0)

        if result.param_count <= 0:
            param_score = 0.0
        elif result.param_count <= self.PARAM_BUDGET:
            # Reward smaller models more: score = 1 - (params / budget)
            param_score = 1.0 - (result.param_count / self.PARAM_BUDGET)
            param_score = max(param_score, 0.1)   # floor at 0.1 for meeting budget
        else:
            # Over budget: partial credit proportional to how close to the limit
            param_score = max(0.0, 1.0 - (result.param_count / self.PARAM_BUDGET) * 0.5)

        # 70% accuracy, 30% efficiency
        score = 0.7 * acc_score + 0.3 * param_score
        return round(score, 4)


# ── Task 3: residual-depth (Hard) ─────────────────────────────────────────────

class ResidualDepthGrader(BaseTaskGrader):
    """
    Hard task: handle a high-dimensional dataset (breast_cancer, 30 features).

    Success: accuracy > 75% on the breast_cancer dataset.
    Partial credit: proportional to accuracy achievement.
    Penalty: if wrong dataset is used, score is halved.
    """

    task_id = "residual-depth"
    difficulty = "hard"
    description = (
        "Achieve > 75% accuracy on the breast_cancer dataset "
        "(30 input features, binary classification)."
    )
    success_criteria = "accuracy > 0.75 AND dataset_name == 'breast_cancer'"

    ACC_TARGET = 0.75
    REQUIRED_DATASET = "breast_cancer"

    def grade(self, result: EpisodeResult) -> float:
        acc_score = min(result.accuracy / self.ACC_TARGET, 1.0)

        # If the environment landed on the wrong dataset, reduce credit
        if result.dataset_name != self.REQUIRED_DATASET:
            acc_score *= 0.5

        return round(acc_score, 4)


# ── Registry ───────────────────────────────────────────────────────────────────

_GRADERS: dict[str, BaseTaskGrader] = {
    g.task_id: g
    for g in [
        ArchFoundationsGrader(),
        EfficientNetGrader(),
        ResidualDepthGrader(),
    ]
}

TASK_IDS: list[str] = list(_GRADERS)


def get_grader(task_id: str) -> BaseTaskGrader:
    """Return the grader for *task_id*, or raise KeyError."""
    if task_id not in _GRADERS:
        raise KeyError(
            f"Unknown task {task_id!r}. Available: {TASK_IDS}"
        )
    return _GRADERS[task_id]


def grade_episode(task_id: str, result: EpisodeResult) -> float:
    """Convenience function: grade *result* against *task_id*. Returns [0, 1]."""
    return get_grader(task_id).grade(result)
