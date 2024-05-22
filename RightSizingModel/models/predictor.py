from models import Enblock

from torch import nn

class GraphPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, edge_index, down_transform, K, **kwargs):
        super(GraphPredictor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_index = edge_index
        self.down_transform = down_transform
        # self.num_vert used in the last and the first layer of encoder and decoder
        self.num_vert = self.down_transform[-1].size(0)

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    Enblock(in_channels, out_channels[idx], K, **kwargs))
            else:
                self.en_layers.append(
                    Enblock(out_channels[idx - 1], out_channels[idx], K,
                            **kwargs))
        
        # The attributes include gender, height, weight, arm_length, crotch_height,
        #                        chest_circumference, hip_circumference, waist_circumference

        self.en_layers.append(nn.Linear(self.num_vert * out_channels[-1], 6))

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.edge_index[i], self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                predicted_val = layer(x)

        return predicted_val