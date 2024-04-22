"""
We will define the trainer class. This class will be responsible for training the model.
"""

#import necessary libraries
from torch.utils.data import DataLoader, TensorDataset 
import torch 
import torch.nn as nn
from torch.optim import Adam
from model import DistributionTransformerModel
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# load the data for training 
def load_data(filename):
    """Load data and prepare DataLoader where:
    - Input: Two sequences (row sums and column sums of the matrix).
    - Target: The original matrix.
    """
    matrices = torch.load(filename)
    # Convert list of NumPy arrays to tensors
    matrices = [torch.tensor(matrix, dtype=torch.float32) for matrix in matrices]
    
    # Prepare row and column sums as additional inputs
    row_sums = [matrix.sum(dim=1) for matrix in matrices]  # Shape: [batch_size, m]
    col_sums = [matrix.sum(dim=0) for matrix in matrices]  # Shape: [batch_size, n]

    # Prepare row and column sums as additional inputs
    row_sums = [matrix.sum(dim=1, keepdim=True) for matrix in matrices]  # Keep dimensions
    col_sums = [matrix.sum(dim=0, keepdim=True) for matrix in matrices]  # Keep dimensions

    # Stack matrices and sums into tensors
    matrices = torch.stack(matrices)  # Shape: [batch_size, m, n]
    row_sums = torch.stack(row_sums)  # Shape: [batch_size, m, 1]
    # Column sums are transposed to match the shape of row sums 
    col_sums = torch.stack(col_sums).transpose(1, 2)  # Shape: [batch_size, n,1])
    dataset = TensorDataset(row_sums, col_sums, matrices)

    # Create a DataLoader
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Optionally, print some dataset elements to verify structure
    print("Sample data from DataLoader:")
    for i, (row, col, mat) in enumerate(data_loader):
        print(f"Batch {i+1}")
        print("Row Sums Shape:", row.shape)
        print("Row Sums:", row[0])
        print("Column Sums Shape:", col.shape)
        print("Column Sums:", col[0])
        print("Matrix Shape:", mat.shape)
        print("Matrix:", mat[0])
        if i == 0:  # Only print the first batch to check
            break

    return data_loader


class Trainer:
    def __init__(self, model, data_loader):
        # Initialize the model on the appropriate device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.data_loader = data_loader
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()  # Using MSE Loss for matrix reconstruction

    def train(self, epochs):
        self.model.train()  # Set the model to training mode
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (row_sums, col_sums, matrices) in enumerate(self.data_loader):
                # Move data to the same device as the model
                row_sums = row_sums.to(self.device)
                col_sums = col_sums.to(self.device)
                matrices = matrices.to(self.device)
                
                # Perform a forward pass through the model
                self.optimizer.zero_grad()
                outputs = self.model(row_sums, col_sums)
                loss = self.criterion(outputs, matrices)
                
                # Backpropagation
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                if batch_idx % 10 == 0:  # Log every 10 batches
                    logging.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

            # Log the average loss for each epoch
            average_loss = total_loss / len(self.data_loader)
            logging.info(f'Epoch {epoch} completed, Average Loss: {average_loss}')


# If you have specific training configuration needs, adjust the arguments here
if __name__ == "__main__":
    filename = "data_matrices.pth"
    dataloader = load_data(filename)
    input_dim = 1  # each row/column sum element is treated as a 1-dimensional input
    model_dim = 512
    num_heads = 8
    num_layers = 3
    output_dim = 4  # m * n for matrix of size m=2, n=2
    m, n = 2, 2
    dropout = 0.1
    
    # Assuming your model is already initialized
    model = DistributionTransformerModel(input_dim=1, model_dim=512, num_heads=8, num_layers=3, output_dim=4*4, m=2, n=2, dropout=0.1).to("cpu")
    # Send data to the same device as model
    
    # # Load a single batch from the DataLoader
    # data_iter = iter(dataloader)
    # row_sums, col_sums, matrices = next(data_iter)
    # row_sums, col_sums, matrices = row_sums.to(model.device), col_sums.to(model.device), matrices.to(model.device)
    # outputs = model(row_sums, col_sums)
    # print("Output shape:", outputs.shape)
    # print("Sample output:", outputs[0])

    trainer = Trainer(model, dataloader)
    trainer.train(epochs=1)
    
	