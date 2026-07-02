import torch

# Assume dataset_A_states is a tensor of shape (num_harmful_prompts, 4096)
# Assume dataset_B_states is a tensor of shape (num_harmless_prompts, 4096)

dataset_A_states = None
dataset_B_states = None

# Step 1: Calculate the mean vector for the harmful prompts
mean_harmful = torch.mean(dataset_A_states, dim=0) # Shape: (4096,)

# Step 2: Calculate the mean vector for the harmless prompts
mean_harmless = torch.mean(dataset_B_states, dim=0) # Shape: (4096,)

# Step 3: Subtract them to get the directional vector
refusal_direction = mean_harmful - mean_harmless # Shape: (4096,)

# Step 4: Normalize it so it becomes a pure unit vector (direction only)
refusal_direction = refusal_direction / torch.norm(refusal_direction)