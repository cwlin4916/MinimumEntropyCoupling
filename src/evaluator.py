import torch
from torch.utils.data import DataLoader, TensorDataset
from model import DistributionTransformerModel

# Function to evaluate the model
def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for row_sums, col_sums, matrices in test_loader:
            row_sums, col_sums, matrices = row_sums.to(device), col_sums.to(device), matrices.to(device)
            outputs = model(row_sums, col_sums)
            loss = criterion(outputs, matrices)
            total_loss += loss.item()
    return total_loss / len(test_loader)


def load_test_data(filename):
    # Load matrices from the file
    matrices = torch.load(filename)
    matrices = [torch.tensor(matrix, dtype=torch.float32) for matrix in matrices]
    row_sums = [matrix.sum(dim=1, keepdim=True) for matrix in matrices]
    col_sums = [matrix.sum(dim=0, keepdim=True) for matrix in matrices]

    # Stack and transpose the data correctly
    matrices = torch.stack(matrices)
    row_sums = torch.stack(row_sums)
    col_sums = torch.stack(col_sums).transpose(1, 2)

    # Create a dataset
    dataset = TensorDataset(row_sums, col_sums, matrices)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    return loader

def print_example_outputs(model, loader, device, num_examples=3):
    model.eval()
    with torch.no_grad():
        for i, (row_sums, col_sums, matrices) in enumerate(loader):
            if i >= num_examples:
                break
            row_sums, col_sums, matrices = row_sums.to(device), col_sums.to(device), matrices.to(device)
            outputs = model(row_sums, col_sums)
            outputs = outputs[0]  # Assuming the batch size could be greater than 1, we take the first example.

            # Calculate the row and column sums of the predicted matrix
            predicted_row_sums = outputs.sum(dim=1).unsqueeze(1)  # Ensure it keeps the same dimensionality
            predicted_col_sums = outputs.sum(dim=0).unsqueeze(0)

            print(f"Example {i+1}:")
            print("Predicted Matrix:\n", outputs.cpu().numpy())
            print("Target Matrix:\n", matrices[0].cpu().numpy())
            print("Input Row Sums:\n", row_sums[0].cpu().numpy())
            print("Input Column Sums:\n", col_sums[0].cpu().numpy())
            print("Predicted Row Sums:\n", predicted_row_sums.cpu().numpy())
            print("Predicted Column Sums:\n", predicted_col_sums.cpu().numpy())
            print("\n")



# Now modify the main block to use this new function:
if __name__ == "__main__":
    # File paths
    test_data_path = "/Users/miltonlin/Documents/GitHub/MinimumEntropyCoupling/src/data/data_gen_m2n2_alg/2x2_10000_1.pth"
    checkpoint_path = "checkpoints/checkpoint_epoch_9.pth"  # Path to your model checkpoint

    # Load the test data
    test_loader = load_test_data(test_data_path)

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistributionTransformerModel(input_dim=1, model_dim=512, num_heads=8, num_layers=3, output_dim=4, m=2, n=2)
    model = model.to(device)

    # Load model state
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate the model
    test_loss = evaluate_model(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")
    
    #print examples of the model output
    print_example_outputs(model, test_loader, device, num_examples=3) 