# Create RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, gnn):
        super(RNNModel, self).__init__()
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim

        #GNN
        self.gnn = gnn
        
        # RNN
        self.rnn = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.2)
        self.ln = nn.LayerNorm(hidden_dim)
        # Readout layer
        #self.fc = nn.Linear(2*hidden_dim, output_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.bn = nn.BatchNorm1d(15)
        self.relu = nn.ReLU()

    def get_temporal_embeddings(self,pai, batch_size):
        l = []
        k = {}
        for ai in pai:
            for sai in ai:
                k[int(sai[0].item())] = []
        for ts in k.keys():
            inf_nodes = statistics_per_timestamp[ts]
            inf_edges = links
            inf_weights = weights

            h = gnn.recurrent(torch.LongTensor(inf_nodes).float().to(device), torch.LongTensor(inf_edges).T.long().to(device), torch.LongTensor(inf_weights).T.float().to(device))
            h = F.relu(h)
            k[ts] = h  
        for ai in pai:
            for sai in ai:
                l.append(k[int(sai[0].item())][int(sai[1].item())])        
        return torch.stack(l).reshape(batch_size,15,128)
    
    
    def forward(self, x, pai):
        
        # Initialize hidden state with zeros
        h0 = torch.autograd.Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(device)
  
        B,T, C = x.shape
        # One time step
        x = self.bn(x)
        out, hn = self.rnn(x)
        #out = self.ln(out)
        #out = self.relu(out)
        h0 = h0.detach()
        if self.gnn != None:
            temporal_embedding = self.get_temporal_embeddings( pai, B)
        else:
            temporal_embedding = compute_location_embeddings(B, pai)
        out = torch.add(out, temporal_embedding)
        
        #out=self.ln(out)
        out = torch.mean(out, dim=1)
        out = self.fc(out) 
        #B,T,C = out.shape
        #out = torch.reshape(out, [B,C,T])
        return out