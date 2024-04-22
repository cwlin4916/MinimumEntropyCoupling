import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Input: two marginal  X and Y distributiosn of not necessarly same shape, say size (1,m) and (1,n) respectively. 
So these are discrete distributions on m and n variables. 
Output: a matrix of size (m,n) representing the joint distribution of the two marginals. 
"""

class TransformerFFN(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, dropout):
        super().__init__()
        self.dropout = dropout
        self.act = F.relu
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, out_dim)

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.ffn = TransformerFFN(model_dim, model_dim * 4, model_dim, dropout) # Adjusted hidden dimension 
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention part
        x= x.permute(1, 0, 2) # Adjusted permutation because of batch_first=False default 
        attn_output, _ = self.self_attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)
        x=x.permute(1,0,2) 

        # Feedforward part
        ffn_output = self.ffn(x) 
        x = x + self.dropout(ffn_output) #add dropout to the output 
        x = self.layer_norm2(x) 
        return x


"""this process the two input marginals independently and outputs the joint distribution 
"""
class DistributionTransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, m,n, dropout=0.1, device=None):
        super().__init__()
         # Set the device explicitly, default to CPU if no device is specified and no GPU is available
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.to(self.device)  # Move the model to the specified device
        
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(model_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        # Update this layer to match the doubled features from concatenation
        # the input would be of dimendsion (1,(m+n)*model_dim) where m,n are the sizes of the two marginals
        self.final_layer = nn.Linear((m+n) * model_dim, m*n)
    def forward(self, dist1, dist2):
        # Concatenate distributions along the sequence dimension
        
        concatenated_input = torch.cat((dist1, dist2), dim=1)
        # print("concatenated_input", concatenated_input.size())
        # this should be of shape (1, m+n, 1) or (batch_size, seq_len, input_dim) where seq_len = m+n 
        embedded_input = self.embedding(concatenated_input)
        # print("embedded_input", embedded_input.size()) 
        
        # Process through transformer layers
        output = embedded_input
        for layer in self.transformer_layers:
            output = layer(output)
        
        # print("output", output.size()) 
        # Flatten the output correctly, maintaining the batch dimension
        output_flat = output.view(output.size(0), -1)  # Reshape to [batch_size, -1]
        # print("output_flat size", output_flat.size())
        # Final transformation to output dimensions and reshaping to matrix
        #check output dimension
        final_output = self.final_layer(output_flat)
        # print("final_output size", final_output.size())
        # set m and n to the sizes of the marginals
        m,n = dist1.size(1), dist2.size(1)
        # print("m,n", m,n)
        return final_output.view(-1, m, n)  # Reshape to [batch_size, m, n]

# Example instantiation and usage
if __name__ == "__main__":
    m, n = 3, 4
    print("m, n:", m, n)
    model = DistributionTransformerModel(model_dim=512, num_heads=8, num_layers=3, input_dim=1,m=m, n=n, output_dim=m*n)
    dist1 = torch.rand(1, m, 1)  # distribution over m elements
    print("dist1 has size", dist1.size())    
    print("dist1", dist1)
    dist2 = torch.rand(1, n, 1)  # distribution over n elements
    output_matrix = model(dist1, dist2)
    print("Size of output matrix:", output_matrix.size())
    print("Output matrix of dimension m*n:", output_matrix)