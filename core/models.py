"""
Data models for the NeuralArch-Bench environment.

Action:      agent provides full updated PyTorch model source code OR a
             natural-language layer modification instruction
Observation: current model code + latest training metrics + episode context
"""

from typing import List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, model_validator


class NeuralArchAction(Action):
    """Agent either replaces the full model code or applies a layer modification."""

    new_model_code: str = Field(
        default="",
        description="Full updated Python source for the PyTorch model class. "
                    "If provided, replaces the current architecture entirely.",
    )
    layer_modification: Optional[str] = Field(
        default=None,
        description="Natural-language instruction to modify the current architecture, "
                    "e.g. 'add BatchNorm1d(256) after fc1' or 'remove drop1'. "
                    "Applied on top of the existing code; ignored if new_model_code is also set.",
    )
    thought_process: str = Field(default="", description="Agent's reasoning behind the architectural change")

    @model_validator(mode="after")
    def at_least_one_action(self) -> "NeuralArchAction":
        if not self.new_model_code and not self.layer_modification:
            raise ValueError("Either new_model_code or layer_modification must be provided.")
        return self


class NeuralArchObservation(Observation):
    """Current state of the neural architecture search episode."""

    current_code: str = Field(default="", description="Raw source code of the current PyTorch model class")
    architecture_name: str = Field(
        default="unknown",
        description="Name of the architecture from the library, or 'custom' if agent-provided",
    )
    dataset_name: str = Field(
        default="fashion_mnist",
        description="Name of the dataset being used for this episode",
    )
    last_accuracy: float = Field(default=0.0, description="Training accuracy from the most recent run (0.0–1.0)")
    param_count: int = Field(default=0, description="Total trainable parameter count")
    loss_curve: List[float] = Field(default_factory=list, description="Loss values from the last 5 epochs")
    error_logs: Optional[str] = Field(default=None, description="Compiler or runtime errors from the last training run")
