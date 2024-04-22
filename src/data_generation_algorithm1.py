import numpy as np
import torch

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

def save_data(m, n, k, filename, seed):
    """Save the adjusted matrices."""
    matrices = generate_and_adjust_matrices(m, n, k, seed)
    torch.save(matrices, filename)
    print(f"Data saved to {filename}")
    #let us check a few elements 
    print("few elements in the data")
    for i in range(15):
        print(matrices[i])

# Example usage
if __name__ == "__main__":
    save_data(m=2, n=2, k=10000, filename="data_matrices_10000.pth", seed=0)
