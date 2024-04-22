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
import matplotlib.pyplot as plt
import time 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(filename, test_size=0.2):
    matrices = torch.load(filename)
    matrices = [torch.tensor(matrix, dtype=torch.float32) for matrix in matrices]
    row_sums = [matrix.sum(dim=1, keepdim=True) for matrix in matrices]
    col_sums = [matrix.sum(dim=0, keepdim=True) for matrix in matrices]

    matrices = torch.stack(matrices)
    row_sums = torch.stack(row_sums)
    col_sums = torch.stack(col_sums).transpose(1, 2)

    # Split the datasets
    train_idx, test_idx = train_test_split(range(len(matrices)), test_size=test_size, random_state=42)
    
    # Create datasets
    train_dataset = TensorDataset(row_sums[train_idx], col_sums[train_idx], matrices[train_idx])
    test_dataset = TensorDataset(row_sums[test_idx], col_sums[test_idx], matrices[test_idx])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader


class Trainer:
    def __init__(self, model, train_loader, test_loader):
        self.model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.test_losses = []

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            epoch_start_time = time.time()
            train_loss = 0
            for batch_idx, (row_sums, col_sums, matrices) in enumerate(self.train_loader):
                start_time = time.time()
                row_sums, col_sums, matrices = row_sums.to(self.model.device), col_sums.to(self.model.device), matrices.to(self.model.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(row_sums, col_sums)
                loss = self.criterion(outputs, matrices)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                batch_time = time.time() - start_time
                
                
                
                if batch_idx % 10 == 0:
                    logging.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Batch Time: {batch_time:.2f}s')
                    
            
            avg_train_loss = train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            epoch_time = time.time() - epoch_start_time
            
            test_loss = self.evaluate()
            self.test_losses.append(test_loss)
            
            logging.info(f'End of Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, Epoch Time: {epoch_time:.2f}s')

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        eval_start_time = time.time()
        with torch.no_grad():
            for row_sums, col_sums, matrices in self.test_loader:
                row_sums, col_sums, matrices = row_sums.to(self.model.device), col_sums.to(self.model.device), matrices.to(self.model.device)
                outputs = self.model(row_sums, col_sums)
                loss = self.criterion(outputs, matrices)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.test_loader)
        eval_time = time.time() - eval_start_time
        logging.info(f'Evaluation done, Average Loss: {avg_loss:.4f}, Eval Time: {eval_time:.2f}s')
        return avg_loss


# After training
if __name__ == "__main__":
    file="data_matrices_10000.pth"
    dataloader_train, dataloader_test = load_data(file)
    model = DistributionTransformerModel(input_dim=1, model_dim=512, num_heads=8, num_layers=3, output_dim=4, m=2, n=2)
    trainer = Trainer(model, dataloader_train, dataloader_test)
    trainer.train(epochs=1)