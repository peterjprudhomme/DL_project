from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Completely ungated - no approval required
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
)


# Test prompt using the standard chat template
messages = [
    {"role": "user", "content": "Write a quick Python function to check if a number is prime."}
]

inputs = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True, 
    return_tensors="pt"
).to(model.device)

outputs = model.generate(inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))