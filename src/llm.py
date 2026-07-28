from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

import torch

from .config import MODEL_NAME
from .config import MAX_NEW_TOKENS


class LLMInterface:

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=dtype,
        )

        self.model.to(device)
        self.device = device


    def generate(self, prompt):

        inputs = self._prepare_inputs(prompt)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS
        )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
    
    def tokenize(self, prompt):

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return self.tokenizer(
            text,
            return_tensors="pt"
        )

    def _prepare_inputs(self, prompt):

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.device)

        return inputs
    

    def get_hidden_states(self, prompt):

        inputs = self._prepare_inputs(prompt)

        with torch.no_grad():

            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )

        return outputs.hidden_states
    
    def get_activations(self, prompt, layers, token_position=-1):

        activations = {}
        hidden_states = self.get_hidden_states(prompt)
        for layer in layers:

            if layer < 0 or layer >= len(hidden_states):
                raise ValueError(
                    f"Layer {layer} does not exist. "
                    f"Model has {len(hidden_states)} hidden-state tensors."
                )

            activation = hidden_states[layer][0, token_position, :]
            activations[layer] = activation.detach().cpu()
        
        return activations