import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.hetero import HeteroGCLSTM

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, metadata):
        super(RecurrentGCN, self).__init__()
        self.l1 = HeteroGCLSTM(in_channels_dict=node_features, out_channels=128, metadata=metadata)
        self.l2 = HeteroGCLSTM(in_channels_dict={"location": 128, "patient": 128}, out_channels=64, metadata=metadata)
        self.linear = torch.nn.Linear(64, 6)

    def forward(self, x, edge_index):
        h,c = self.l1(x, edge_index)
        h['patient'] = F.relu(h['patient'])
        h['location'] = F.relu(h['location'])
        h,c = self.l2(h, edge_index)
        h = F.relu(h['patient'])
        h = self.linear(h)
        return h