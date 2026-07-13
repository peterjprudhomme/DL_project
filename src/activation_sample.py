from dataclasses import dataclass
import torch


@dataclass
class ActivationSample:
    """
    One prompt/response pair and extracted activations.
    """

    prompt: str
    response: str

    # harmful: bool
    # refused: bool 

    # # key = transformer layer
    # # value = activation vector
    activations: dict[int, torch.Tensor]