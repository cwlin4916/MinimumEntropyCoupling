import numpy as np
import torch
import os

def generate_matrices(k, seed):
    """Generate 2x2 matrices with uniform marginal distributions."""
    np.random.seed(seed)
    matrices = []
    for _ in range(k):
        # Sample 'a' uniformly from 0 to 0.5
        a = np.random.uniform(0, 0.5)
        # Calculate 'b', 'c', and 'd'
        b = 0.5 - a
        c = 0.5 - a
        d = 1 - b - c - a
        # Construct the matrix
        matrix = np.array([[a, b], [c, d]])
        matrices.append(matrix)
    
    return matrices

def save_data(k, seed):
    """Save the generated matrices into structured directories."""
    directory = "data/data_gen_m2n2_alg"
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"2_2_{k}_{seed}.pth")
    matrices = generate_matrices(k, seed)
    torch.save(matrices, filename)
    print(f"Data saved to {filename}")

    # Optional: Displaying a few elements of the data
    print("A few elements from the saved data:")
    for matrix in matrices[:min(15, len(matrices))]:
        print(matrix)

# Example usage
if __name__ == "__main__":
    save_data(k=10000, seed=1)
