import torch
import torch.nn as nn
import torch.nn.functional as F
from nanoGPT.nano_GPT_model import Block, GPTConfig

"""
Input: two marginal  X and Y distributiosn of not necessarly same shape, say size (1,m) and (1,n) respectively. 
So these are discrete distributions on m and n variables. 
Output: a matrix of size (m,n) representing the joint distribution of the two marginals. 
"""

## need to input noramlize output 



class TransformerLayer(Block):
    """Inherits directly from nanoGPT's Block which includes attention and feed-forward layers"""
    def __init__(self, config):
        super().__init__(config)

"""this process the two input marginals independently and outputs the joint distribution 
"""


class DistributionTransformerModel(nn.Module):
    def __init__(self, model_dim, input_dim, num_heads, num_layers, dropout=0.1, device=None):
        super().__init__()
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        # Initialize the embedding layer to convert input dimension to model dimension
        self.embedding = nn.Linear(input_dim, model_dim)
        # Setup transformer layers as a list of transformer layers
        # Each transformer layer is a Block from nanoGPT
        # We can use the Block class from nanoGPT, which includes attention and feed-forward layers, here we can use the transformer layer as a block
        self.transformer_layers = nn.ModuleList([
            Block(GPTConfig(n_embd=model_dim, n_head=num_heads, dropout=dropout, block_size=25, bias=True))
            for _ in range(num_layers)
        ]) 
        # for _ in range (num_layers) means we are creating num_layers of transformer layers 
        # Final layer to map the transformer output back to the desired output shape
        

    def forward(self, dist1, dist2):
        # first two elements are m and n 
        # we first get the m and n values these are length of the marginals
        m, n = dist1.size(1), dist2.size(1)
        # Embedding both distributions
        dist1 = self.embedding(dist1)
        dist2 = self.embedding(dist2)
        # Concatenate along the sequence length dimension (dim=1)
        concatenated_input = torch.cat((dist1, dist2), dim=1)
        # Process through transformer layers
        output = concatenated_input

        for layer in self.transformer_layers:
            output = layer(output)

        output_flat = output.view(output.size(0), -1)
        final_output = nn.Linear(output_flat.size(1), m * n)(output_flat)
        final_output = final_output.view(-1, m, n)
        # this has shape (batch_size, m, n) 
        # We can apply softmax to get the joint distribution 
        # we first flatten the output to apply softmax 
        final_output = F.softmax(final_output.view(-1,m*n), dim =1).view(-1,m,n)
        return final_output # Reshape or process output as needed


# Example instantiation and usage
if __name__ == "__main__":
    m, n = 3, 4
    model = DistributionTransformerModel(input_dim=1, model_dim=64, num_heads=8, num_layers=4)
    dist1 = torch.rand(1, m, 1)  # Random distribution over m elements
    dist2 = torch.rand(1, n, 1)  # Random distribution over n elements
    print("The input are two distributions of sizes", dist1.size(), "and", dist2.size())
    output = model(dist1, dist2)
    print("Output:", output)
    print("Output size:", output.size())
    print("Sum of output:", output.sum())  # Check that the sum of the output is 1