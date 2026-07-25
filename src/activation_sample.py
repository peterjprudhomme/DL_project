from dataclasses import dataclass
import torch


@dataclass
class ActivationSample:
    """
    One prompt/response pair and extracted activations.
    """

    prompt: str
    response: str

    # Ground truth / metadata
    harmful: bool | None = None
    refused: bool | None = None

    # Layer -> activation vector
    activations: dict[int, torch.Tensor] | None = None