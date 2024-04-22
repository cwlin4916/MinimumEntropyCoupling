import numpy as np
import torch
import os

def generate_and_adjust_matrices(m, n, k, seed, upper_bound=100000, lower_bound=1):
    """Generate matrices and adjust them to have uniform marginal distributions."""
    np.random.seed(seed)
    matrices = []
    for _ in range(k):
        # Generate a random matrix
        matrix = np.random.randint(lower_bound, upper_bound, size=(m, n))
        matrix = matrix.astype(float)
        matrix /= matrix.sum()  # Normalize to make the sum of all elements 1

        # Adjust columns
        for j in range(n-1):
            column_sum = matrix[:, j].sum()
            correction = (1/n - column_sum) / m
            matrix[:, j] += correction
            matrix[:, j+1] -= correction
        
        # Adjust rows
        for i in range(m-1):
            row_sum = matrix[i, :].sum()
            correction = (1/m - row_sum) / n
            matrix[i, :] += correction
            matrix[i+1, :] -= correction
        
        matrices.append(matrix)
    
    return matrices

def save_data(m, n, k, seed):
    """Save the adjusted matrices into structured directories."""
    directory = f"data/algorithm_1"
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"{m}_{n}_{k}_{seed}.pth")
    matrices = generate_and_adjust_matrices(m, n, k, seed)
    torch.save(matrices, filename)
    print(f"Data saved to {filename}")

    # Optional: Displaying a few elements of the data
    print("A few elements from the saved data:")
    for i in range(min(15, len(matrices))):
        print(matrices[i])

# Example usage
if __name__ == "__main__":
    save_data(m=2, n=2, k=10000, seed=1)