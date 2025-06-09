import torch
import torch.nn as nn

class fast_ann(nn.Module):
    def __init__(self, layer_dims, dropout_rate=0.5):
        """
        Args:
            layer_dims (list): A list containing the number of neurons in each layer.
                               The first element should correspond to the input size,
                               and the last element should be the output size.
            dropout_rate (float): The dropout rate to use for regularization. Default is 0.5.
        """
        super(fast_ann, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout_rate = dropout_rate

        # Create each layer in sequence based on the provided dimensions
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:  # Apply ReLU and Dropout to all but the last layer
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(p=dropout_rate))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.sigmoid(x)
        x = x.squeeze()

        # Add an additional column of zeros at the fourth position
        return torch.cat([x[:, :3], torch.zeros([x.shape[0], 1]).to(x.device), x[:, 3:]], axis=1)