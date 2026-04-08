"""
Data models for the NeuralArch-Bench environment.

The environment runs a 3-phase cycle per turn:
  1. DIAGNOSE  — agent explains what is wrong with the current model
  2. PLAN      — agent describes the specific change it will make
  3. IMPLEMENT — agent writes the new model code; training runs

Each phase uses a different action field and returns a different reward signal.
"""

from typing import List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class NeuralArchAction(Action):
    """
    Agent action. Exactly one field should be set per step, matching the
    current phase reported in the observation's current_phase field.

    Phase 1 (diagnose)  → set diagnosis
    Phase 2 (plan)      → set change_plan
    Phase 3 (implement) → set new_model_code
    """

    diagnosis: Optional[str] = Field(
        default=None,
        description="Phase 1 (DIAGNOSE): Explain what is wrong with the current model and why.",
    )
    change_plan: Optional[str] = Field(
        default=None,
        description="Phase 2 (PLAN): Describe the specific architectural change you will make.",
    )
    new_model_code: Optional[str] = Field(
        default=None,
        description="Phase 3 (IMPLEMENT): Full Python source defining ArchModel(nn.Module).",
    )


class NeuralArchObservation(Observation):
    """Current state of the neural architecture search episode."""

    # ── training metrics ──────────────────────────────────────────────
    current_code: str = Field(default="", description="Source code of the current PyTorch model")
    architecture_name: str = Field(default="unknown", description="Name from arch library, or 'custom'")
    dataset_name: str = Field(default="iris", description="Dataset being used this episode")
    last_accuracy: float = Field(default=0.0, description="Test accuracy from the most recent training run (0–1)")
    param_count: int = Field(default=0, description="Total trainable parameter count")
    loss_curve: List[float] = Field(default_factory=list, description="Loss values from last 5 epochs")
    error_logs: Optional[str] = Field(default=None, description="Errors from the last training run")

    # ── phase state ───────────────────────────────────────────────────
    current_phase: str = Field(
        default="diagnose",
        description="Current phase: 'diagnose' | 'plan' | 'implement'",
    )
    last_diagnosis: Optional[str] = Field(
        default=None,
        description="Agent's diagnosis from phase 1 of the current cycle",
    )
    last_plan: Optional[str] = Field(
        default=None,
        description="Agent's change plan from phase 2 of the current cycle",
    )
    phase_rewards: List[float] = Field(
        default_factory=list,
        description="Rewards earned so far in the current cycle [diag_reward, plan_reward, impl_reward]",
    )

    # ── task grading ──────────────────────────────────────────────────
    task_id: str = Field(
        default="arch-foundations",
        description="Active task ID: 'arch-foundations' | 'efficient-net' | 'residual-depth'",
    )
    task_score: float = Field(
        default=0.0,
        description="Normalized task score in [0.0, 1.0] from the task grader after each implement step",
    )
