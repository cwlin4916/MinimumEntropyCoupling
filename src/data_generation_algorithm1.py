"""The goal of the project is to learn joint distribution of data given marginal distribution of each variable. 
This file generates data for the project via backward generation. 
this code has two parts 
1. generate the random matrix of size (m,n), m and n are the sizes of the two marginals, which has poistive entries.
2. adjust the random matrix by normalizing and adjusting each column. """

import numpy as np
 
 # function for generating random matrix, which has positive entries. 
def generate_random_positive_matrices(m,n,seed,k,upper_bound=100,lower_bound=1): #k is the number of matrices to generate
    np.random.seed(seed)
    matrices = []
    for i in range(k):
        matrix = np.random.randint(lower_bound, upper_bound, size=(m,n))
        matrices.append(matrix)
    return matrices


def adjust_matrices(matrices):
    """ Adjust matrices to have uniform marginal distributions.
    Input: list of matrices. 
    Output: list of adjusted matrices with uniform  marginal distributions. 
    """
    adjusted_matrices = []
    
    for matrix in matrices:
        matrix = matrix.astype(float)   # Convert matrix to float to allow for division and multiplication
        adjusted_matrices.append(matrix.copy())  # Use copy to ensure original is not modified
        print("adjusted_matrices", adjusted_matrices)
        
        m, n = matrix.shape
        total_sum = matrix.sum()
        print("sum of current matrix is ", total_sum)
        
        for j in range(n-1):
            column_sum = matrix[:, j].sum()
            print("sum/n of current matrix is ", total_sum / n)
            print("column sum of jth column is ", column_sum)
            
            x = (total_sum / n - column_sum) / m  # Adjustment for each row in column
            print("The amount of x to add to each row of the column j is ", x)
            
            matrix[:, j] += x
            matrix[:, j+1] -= x  # Adjust the next column to keep total sum constant
            
            print(f"column {j} sum: {matrix[:, j].sum()}")
            assert np.isclose(matrix[:, j].sum(), total_sum / n)
        # we will similarly adjust the rows 
        for i in range(m-1):
            row_sum = matrix[i,:].sum()
            print("row sum of ith row is ", row_sum)
            y = (total_sum / m - row_sum) / n
            print("The amount of y to add to each column of the row i is ", y)
            matrix[i,:] += y
            # now we need to guarantee that we do not change the column 
            matrix[i+1,:] -= y
            print(f"row {i} sum: {matrix[i,:].sum()}")
            assert np.isclose(matrix[i,:].sum(), total_sum / m)
            
        
        matrix /= total_sum  # Normalize the final matrix
        print("final matrix sum", matrix.sum())
        adjusted_matrices[-1] = matrix  # Replace the original with the adjusted
        
    return adjusted_matrices


# below code is for testing the functions 
def test_alg():
    m, n = 2,2
    seed = 0
    upper_bound = 100
    lower_bound = 1
    k = 10
    matrices = generate_random_positive_matrices(m, n, seed, k, upper_bound, lower_bound)
    print("matrices", matrices)
    adjusted_matrices = adjust_matrices(matrices)
    print("adjusted_matrices", [mat.tolist() for mat in adjusted_matrices])
    # we will print out the 5th matrix to check if the columns sum to 1
    # the index of matrix which we want to look 
    index = 4
    print(f"the matrix of {index} is : \n  {adjusted_matrices[index]}")
    print("sum of the columns of the matrix", adjusted_matrices[index].sum(axis=0))
    print("sum of the rows of the matrix", adjusted_matrices[index].sum(axis=1))
    
    
if __name__ == "__main__":
    test_alg()
