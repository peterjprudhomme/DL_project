import re

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# "Qwen/Qwen2.5-7B-Instruct"  <- this is the bigger model
MAX_NEW_TOKENS = 150

# Transformer layers to extract activations from.
LAYERS = [5, 10, 15, 20]


def model_slug(model_name: str = MODEL_NAME) -> str:
    """Return a filesystem-safe slug for a model name.

    Replaces any run of characters outside [A-Za-z0-9._-] with a single
    underscore and strips leading/trailing underscores.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")
