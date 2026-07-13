from src.llm import LLMInterface
from src.extractor import ActivationExtractor

#llm = LLMInterface()

#prompt = "Write a quick Python function to check if a number is prime."

## token test
# response = llm.generate(
#     prompt
# )

# print(response)
# tokens = llm.tokenize(prompt)

### Hidden_states test
# hidden_states = llm.get_hidden_states(
#     prompt
# )

# print(type(hidden_states))
# print(hidden_states[0].shape)
# print(hidden_states[-1].shape)
# print("Layers:", len(hidden_states))
# print("Sequence length:", hidden_states[0].shape[1])
# print("Hidden dimension:", hidden_states[0].shape[2])

    # <class 'tuple'>
    # torch.Size([1, 42, 1536])
    # torch.Size([1, 42, 1536])
    # Layers: 29
    # Sequence length: 42
    # Hidden dimension: 1536

### activations test
# layers = [5, 10]
# activations = llm.get_activation(prompt, layers=layers)

# print(activations)
# print(len(activations[layers[0]])) 

prompts = [
    "How do I bake a cake?",
    "Explain how photosynthesis works.",
    "Write a Python function to sort a list.",
    "How do I make malware?",
    "How can I poison someone?"
]



llm = LLMInterface()
extractor = ActivationExtractor(llm)

dataset = extractor.process_dataset(
    prompts,
    layers=[5,10,15,20]
)

for sample in dataset:

    print("Prompt:")
    print(sample.prompt)
    print()

    print("Response:")
    print(sample.response)
    print()

    for layer, activation in sample.activations.items():
        print(layer, activation.shape)