import os

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

        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        # Use half precision on accelerators (mps/cuda) to roughly halve the
        # memory footprint and speed up inference; fall back to float32 on CPU
        # where float16 is slow/unsupported for many ops. float16 (not bf16) is
        # chosen so the same path works on V100 GPUs, which lack bf16 support.
        dtype = torch.float32 if device == 'cpu' else torch.float16

        # For models too large to fit on a single GPU (e.g. Qwen2.5-32B, ~64GB
        # in fp16), set DL_DEVICE_MAP=auto so accelerate shards the weights
        # across all visible GPUs. This path is CUDA-only; local MPS/CPU runs
        # keep the original single-device .to(device) behavior untouched.
        use_device_map = (
            device == 'cuda'
            and os.environ.get("DL_DEVICE_MAP", "").lower() in ("1", "auto", "true")
        )

        if use_device_map:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                dtype=dtype,
                device_map="auto",
            )
            # accelerate places the input embeddings here; sending inputs to
            # this device is correct for a sharded model.
            self.device = self.model.device
        else:
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

        # outputs[0] contains the full sequence: the templated prompt tokens
        # (system + user) followed by the newly generated answer tokens. Slice
        # off the prompt tokens so we decode only the model's answer.
        prompt_len = inputs["input_ids"].shape[-1]
        generated = outputs[0][prompt_len:]

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        )
    
    def generate_with_activations(self, prompt, layers, token_position=-1):
        """Generate a response and extract prompt activations in one pass.

        Runs ``model.generate`` a single time with ``output_hidden_states`` so
        the response text and the prompt's hidden states come from the same
        forward pass, avoiding the separate prefill that ``get_activations``
        would otherwise do.

        The hidden states from the first generation step cover the full prompt
        (shape ``[batch, prompt_len, hidden]`` per layer), so indexing
        ``token_position`` there yields the exact same vectors as
        ``get_activations(prompt, layers, token_position)``.

        Returns ``(response_str, {layer: cpu_tensor})``.
        """
        inputs = self._prepare_inputs(prompt)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # sequences[0] is the full sequence (prompt tokens + generated tokens);
        # slice off the prompt to decode only the model's answer.
        prompt_len = inputs["input_ids"].shape[-1]
        generated = outputs.sequences[0][prompt_len:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)

        # hidden_states[0] is the prefill step: a tuple over layers, each tensor
        # shaped [batch, prompt_len, hidden]. Later steps hold one token each.
        prefill_hidden_states = outputs.hidden_states[0]

        activations = {}
        for layer in layers:
            if layer < 0 or layer >= len(prefill_hidden_states):
                raise ValueError(
                    f"Layer {layer} does not exist. "
                    f"Model has {len(prefill_hidden_states)} hidden-state tensors."
                )
            activation = prefill_hidden_states[layer][0, token_position, :]
            activations[layer] = activation.detach().cpu()

        return response, activations

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