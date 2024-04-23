#import necessary libraries
import os
import logging
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import DistributionTransformerModel

# Configure logging and directories
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
figures_directory = "figures"
checkpoints_directory = "checkpoints"
os.makedirs(figures_directory, exist_ok=True)
os.makedirs(checkpoints_directory, exist_ok=True)


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



# Define the Trainer class
class Trainer:
    def __init__(self, model, train_loader, test_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.test_losses = []

    def train(self, epochs, checkpoint_intervals=1):
        start_epoch = 0
        latest_checkpoint_path = os.path.join(checkpoints_directory, "latest_checkpoint.pth")
        if os.path.exists(latest_checkpoint_path):
            start_epoch = self.load_checkpoint(latest_checkpoint_path)
            logging.info(f"Checkpoint found at {latest_checkpoint_path}. Resuming training from epoch {start_epoch}")
        
        for epoch in range(start_epoch, epochs):
            self.model.train()
            train_loss = []
            for batch_idx, (row_sums, col_sums, matrices) in enumerate(self.train_loader):
                row_sums, col_sums, matrices = row_sums.to(self.device), col_sums.to(self.device), matrices.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(row_sums, col_sums)
                loss = self.criterion(outputs, matrices)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

            avg_train_loss = sum(train_loss) / len(train_loss)
            test_loss = self.evaluate()
            self.train_losses.append(avg_train_loss)
            self.test_losses.append(test_loss)
            logging.info(f'End of Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}')
            
            if (epoch + 1) % checkpoint_intervals == 0:
                self.save_checkpoint(epoch)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for row_sums, col_sums, matrices in self.test_loader:
                row_sums, col_sums, matrices = row_sums.to(self.device), col_sums.to(self.device), matrices.to(self.device)
                outputs = self.model(row_sums, col_sums)
                loss = self.criterion(outputs, matrices)
                total_loss += loss.item()
        return total_loss / len(self.test_loader)
    
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses
        }
        checkpoint_path = os.path.join(checkpoints_directory, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        return checkpoint['epoch'] + 1

# Plotting function for epoch losses
def plot_within_epoch_loss(epoch_losses, epoch_number):
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label=f'Loss for Epoch {epoch_number}')
    plt.title(f'Training Loss over Batches for Epoch {epoch_number}')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    fig_path = os.path.join(figures_directory, f"epoch_{epoch_number}_loss.png")
    plt.savefig(fig_path)
    plt.show()
    logging.info(f"Figure saved as {fig_path}")
    
def plot_epoch_losses(train_losses, test_losses, data_label, model_params):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.xlabel('Epoch Number')
    #average y value is 0.006 so we will use log scale to see the difference
    plt.yscale('log')
    plt.ylabel('Loss')
    plt.legend()
    fig_name = f"{data_label}_model_{model_params}_loss.png"
    fig_path = os.path.join(figures_directory, fig_name)
    plt.savefig(fig_path)
    plt.show()
    logging.info(f"Figure saved as {fig_path}")

# Main block to initiate training and testing
if __name__ == "__main__":
    data_file = "data/algorithm_1/2_2_10000_0.pth"
    # we will extract data_label from the  data_file 
    data_label = data_file.split("/")[-1].split(".")[0]
    dataloader_train, dataloader_test = load_data(data_file)
    model = DistributionTransformerModel(input_dim=1, model_dim=512, num_heads=8, num_layers=3, output_dim=4, m=2, n=2)
    # we will extract model params, which are m,n from the model 
    model_params = f"{model.m}_{model.n}"
    trainer = Trainer(model, dataloader_train, dataloader_test)
    trainer.train(epochs=100)  # Example for training 10 epochs
    plot_epoch_losses(trainer.train_losses, trainer.test_losses, data_label, model_params)
    
    
