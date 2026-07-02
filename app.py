from src.llm import LLMInterface

llm = LLMInterface()

prompt = "Write a quick Python function to check if a number is prime."

# response = llm.generate(
#     prompt
# )

# print(response)

# tokens = llm.tokenize(prompt)

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

# activation = llm.get_activation(prompt, layer=10)

# print(len(activation))
# print(activation)