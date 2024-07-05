# example.py

import numpy as np
from attention import AttentionMechanism

# Example input data
inputs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Initialize attention mechanism
attention = AttentionMechanism(input_dim=3, attention_units=2)

# Calculate attention
context_vector, attention_weights = attention.calculate_attention(inputs)

# Print results
print("Inputs:\n", inputs)
print("\nAttention Weights:\n", attention_weights)
print("\nContext Vector:\n", context_vector)
