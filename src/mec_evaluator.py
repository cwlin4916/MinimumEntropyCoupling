import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from model2 import DistributionTransformerModel  # Ensure the correct import path
import numpy as np
# load in entropy and kl2 functions 

# entropy function 
def entropy2(q, prec=18):
    res = q * np.log2(q)# calculate the entropy of q 
    res[q == 0] = 0
    ressum = res.sum()
    return -np.around(ressum, decimals=prec)


def load_checkpoint(filename, model, optimizer):
    print(f"Loading checkpoint '{filename}'")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print("Checkpoint loaded successfully.")

def load_data(filename):
    print(f"Loading test data from {filename}")
    data = torch.load(filename)
    p_list = [torch.tensor(item[0], dtype=torch.float32).unsqueeze(1) for item in data]
    q_list = [torch.tensor(item[1], dtype=torch.float32).unsqueeze(1) for item in data]
    M_list = [torch.tensor(item[2], dtype=torch.float32) for item in data]
    
    p_tensor = torch.stack(p_list)
    q_tensor = torch.stack(q_list)
    M_tensor = torch.stack(M_list)
    
    dataset = TensorDataset(p_tensor, q_tensor, M_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    return loader

def evaluate(model, test_loader, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for p, q, M in test_loader:
            p, q, M = p.to(device), q.to(device), M.to(device)
            outputs = model(p, q)
            val_loss = F.mse_loss(outputs, M)
            total_val_loss += val_loss.item()
            print("Sample Input (p):", p[0].cpu().numpy())  # Print first item of the batch
            print("Sample Input (q):", q[0].cpu().numpy())
            print("Algorithm Output (M):", M[0].cpu().numpy())
            # calculate entropy  of M 
            print("Entropy of algorithm output M:", entropy2(M[0].cpu().numpy()))
            
            print("Model Prediction :", outputs[0].cpu().numpy())
            # check that the sum of output is 1 
            print("Sum of output:", outputs[0].sum().cpu().numpy())
            # also calculate entropy of prediction  
            print("Entropy of Model Prediction:", entropy2(outputs[0].cpu().numpy())) 
    avg_val_loss = total_val_loss / len(test_loader)
    print(f"Test Loss: {avg_val_loss:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistributionTransformerModel(input_dim=1, model_dim=512, num_heads=8, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    test_data_path = "/Users/miltonlin/Documents/GitHub/MinimumEntropyCoupling/src/data/mec_generation/mec_data_3_10^4_test.pth"
    model_checkpoint_path = "/Users/miltonlin/Documents/GitHub/MinimumEntropyCoupling/src/output/model_checkpoint_3_3_700.pth"

    test_loader = load_data(test_data_path)
    load_checkpoint(model_checkpoint_path, model, optimizer)
    evaluate(model, test_loader, device)
