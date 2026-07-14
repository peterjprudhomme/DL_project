from datasets import load_dataset

# must be logged into HuggingFace with approved account to work
# I used "hf auth login" in my VS Code terminal to login to my HuggingFace account with a token
dataset = load_dataset("walledai/HarmBench", "standard")

print(len(dataset['train']['prompt']))