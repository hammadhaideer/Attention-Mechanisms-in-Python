# attention.py

import numpy as np

class AttentionMechanism:
    def __init__(self, input_dim, attention_units):
        self.input_dim = input_dim
        self.attention_units = attention_units
        self.W = np.random.randn(input_dim, attention_units)

    def calculate_attention(self, inputs):
        # Calculate attention weights
        u = np.dot(inputs, self.W)  # Dot product with trainable weights
        attention_weights = np.exp(u) / np.sum(np.exp(u), axis=1, keepdims=True)

        # Calculate context vector
        context_vector = np.sum(inputs * attention_weights, axis=0)

        return context_vector, attention_weights

# Example usage
if __name__ == "__main__":
    # Example input
    inputs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Initialize attention mechanism
    attention = AttentionMechanism(input_dim=3, attention_units=2)

    # Calculate attention
    context_vector, attention_weights = attention.calculate_attention(inputs)

    print("Inputs:\n", inputs)
    print("\nAttention Weights:\n", attention_weights)
    print("\nContext Vector:\n", context_vector)
