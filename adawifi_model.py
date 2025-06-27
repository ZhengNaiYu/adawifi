import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


class Client_Encoder(nn.Module):
    def __init__(self, n_clients=None, n_feature=224, n_hidden=128, n_rnn_layers=1, p_dropout=0.1):
        super(Client_Encoder, self).__init__()
        
        self.n_clients = n_clients
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.p_dropout = p_dropout

        self.dropout1 = nn.Dropout(p_dropout)
        self.dropout2 = nn.Dropout(p_dropout)

        self.bn = nn.BatchNorm1d(num_features=n_feature)
        
        # RNN to encode each receiver
        self.linear1 = nn.Linear(n_feature, n_hidden)
        self.rnn = nn.GRU(input_size=n_hidden, hidden_size=n_hidden, batch_first=True, num_layers=n_rnn_layers)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
  
    def forward(self, inp):
        if self.n_clients:
            inp = inp[:, :self.n_clients, :, :]            # [batch, n_clients, seq, feature]
        inp = inp.swapaxes(2, 3)
        batch_sz = inp.size(0)
        n_clients = inp.size(1)
        seq_len = inp.size(2)
        feature_sz = inp.size(3)
        x = inp.contiguous().view(-1, seq_len, feature_sz) # [batch*n_clients, seq, feature]
        # x = F.normalize(x, dim=2)
        x = self.bn(x.swapaxes(1, 2)).swapaxes(1, 2)

        x = F.relu(self.linear1(x))                        # [batch*n_clients, seq, hidden]
        x = self.dropout1(x)
        x, _ = self.rnn(x)                                 # [batch*n_clients, seq, hidden]
        x = self.dropout2(x)
        x = x.view(batch_sz, n_clients, seq_len, x.size(-1))   # [batch, n_clients, seq, hidden]
        # print(x.shape)
        x = x[:, :, -1, :]                                 # [batch, n_clients, hidden]
        x = self.linear2(x)
        enc = F.normalize(x, dim=2)
        return x

    
class Client_Transformer(nn.Module):
    def __init__(self, n_activities=6, n_head=4, n_encoder_layers=1, n_hidden=128, dim_feedforward=128, **kwargs):
        super(Client_Transformer, self).__init__()

        self.client_encoder = Client_Encoder(n_hidden=n_hidden, **kwargs)
        d_model = n_hidden
        self.pos_encoder = nn.Parameter(torch.randn(10, 1, d_model) * 1e-1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers, encoder_norm)
        self.linear1 = nn.Linear(d_model, n_activities)
  
    def forward(self, inp):
        enc = self.client_encoder(inp)
        x = enc.swapaxes(0, 1)                             # [n_clients, batch, hidden]
        n_clients = x.size(0)
        x = x + self.pos_encoder[:n_clients]
        x = self.transformer_encoder(x)                    # [n_clients, batch, d_model]
        # print(x.shape)
        x = x.mean(dim=0)                                  # [batch, d_model]
        x = self.linear1(x)                                # [batch, n_activities]
        return x, enc

    
class Client_Set_Transformer(nn.Module):
    def __init__(self, n_activities=6, n_head=4, n_encoder_layers=1, n_hidden=64, dim_feedforward=64, **kwargs):
        super(Client_Set_Transformer, self).__init__()

        self.client_encoder = Client_Encoder(n_hidden=n_hidden, **kwargs)
        d_model = n_hidden
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers, encoder_norm)
        self.linear1 = nn.Linear(d_model, n_activities)
  
    def forward(self, inp):
        enc = self.client_encoder(inp)
        x = enc.swapaxes(0, 1)                             # [n_clients, batch, hidden]
        x = self.transformer_encoder(x)                    # [n_clients, batch, d_model]
        # print(x.shape)
        x = x.mean(dim=0)                                  # [batch, d_model]
        x = self.linear1(x)                                # [batch, n_activities]
        return x, enc

    
class Client_Mean(nn.Module):
    # aggregrate the receiver embeddings with average pooling
    def __init__(self, n_activities=6, **kwargs):
        super(Client_Mean, self).__init__()
        self.client_encoder = Client_Encoder(**kwargs)
        n_hidden = self.client_encoder.n_hidden
        self.linear1 = nn.Linear(n_hidden, n_activities)
  
    def forward(self, inp):
        enc = self.client_encoder(inp)                     # [batch, n_clients, hidden]
        x = enc.mean(dim=1)                                # [batch, hidden]
        x = F.normalize(x, dim=1)
        x = self.linear1(x)                                # [batch, n_activities]
        return x, enc

    
class Client_Max(nn.Module):
    # aggregrate the receiver embeddings with max pooling
    def __init__(self, n_activities=6, **kwargs):
        super(Client_Max, self).__init__()
        self.client_encoder = Client_Encoder(**kwargs)
        n_hidden = self.client_encoder.n_hidden
        self.linear1 = nn.Linear(n_hidden, n_activities)
  
    def forward(self, inp):
        enc = self.client_encoder(inp)                     # [batch, n_clients, hidden]
        x, _ = enc.max(dim=1)                              # [batch, hidden]
        x = F.normalize(x, dim=1)
        x = self.linear1(x)                                # [batch, n_activities]
        return x, enc
    

class MixRx_Transformer(nn.Module):
    def __init__(self, n_activities=6, n_head=4, n_encoder_layers=1, n_hidden=128, max_pos_idx=300, **kwargs):
        super(MixRx_Transformer, self).__init__()
        self.n_activities = n_activities
        self.n_hidden = n_hidden
        dim_feedforward = n_hidden

        self.client_encoder = Client_Encoder(n_hidden=n_hidden, **kwargs)
        d_model = n_hidden
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers, encoder_norm)
        # FIXME
        # self.pos_emb = nn.Parameter(torch.zeros(assign_rx_pos.max_pos_idx, n_hidden))
        # self.pos_weight = nn.Parameter(torch.ones(assign_rx_pos.max_pos_idx, 1))
        self.pos_emb = nn.Parameter(torch.zeros(max_pos_idx, n_hidden))
        self.pos_weight = nn.Parameter(torch.ones(max_pos_idx, 1))
        self.cls_emb = nn.Parameter(torch.randn(1, 1, n_hidden) * 1e-1)
        self.classifier = nn.Linear(d_model, n_activities)
        # self.pos_encoder = nn.Embedding(num_embeddings=assign_rx_pos.max_pos_idx, embedding_dim=d_model)    # one embedding for each receiver
        # nn.init.constant_(self.pos_encoder.weight, 1)

    def forward_client_emb(self, inp, data_T=None, pos=None):
        emb = self.client_encoder(inp, data_T)
        if pos is not None:
            pos_weight = self.pos_weight[pos]  # [batch, n_clients, hidden]
            pos_weight = F.softmax(pos_weight, dim=1)
            # print(pos_emb.shape)
            pos_emb = self.pos_emb[pos]
            emb = emb + pos_emb
            emb = emb * pos_weight  # [batch, n_clients, hidden]
        return emb

    # def forward_emb_agg(self, emb):
    #     emb_agg = emb.mean(dim=1)
    #     return emb_agg

    def forward_emb_agg(self, emb):
        cls_emb = self.cls_emb.repeat(emb.size(0), 1, 1)
        # cls_pad = torch.ones(emb.size(0), 1).long()
        # cls_emb = self.pos_encoder(cls_pad)                # [batch, 1, hidden]
        x = torch.cat([cls_emb, emb], dim=1)  # [batch, n_clients+1, hidden]
        x = x.swapaxes(0, 1)  # [n_clients+1, batch, hidden]
        x = self.transformer_encoder(x)  # [n_clients+1, batch, d_model]
        emb_agg = x[0]  # [batch, d_model]
        return emb_agg

    def forward(self, inp, data_T=None, pos=None):
        emb = self.forward_client_emb(inp, data_T, pos)
        emb_agg = self.forward_emb_agg(emb)
        out = self.classifier(emb_agg)
        return out