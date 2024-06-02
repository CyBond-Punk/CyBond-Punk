import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import softmax
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class bond_agg_layer(MessagePassing, nn.Module):
    def __init__(self, input_dim,output_dim, dropout):
        super(bond_agg_layer, self).__init__()
        self.dropout = dropout
        self.bond_atten = Linear(input_dim,1,bias=False)        
        self.bond_lin = Linear(input_dim,output_dim,bias=False)
        self.bond_lin2 = Linear(output_dim,output_dim,bias=False)
        self.bn = torch.nn.BatchNorm1d(output_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.bond_atten.weight,a=0,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_uniform_(self.bond_lin.weight,a=0,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_uniform_(self.bond_lin2.weight,a=0,mode='fan_in',nonlinearity='relu')

    def forward(self, x, edge_index):  
        bond_embedding = self.bond_agg(x, edge_index)
        return bond_embedding
    
    def bond_agg(self, x, edge_index):
        edge_id = edge_index[1]
        edge_atten = self.bond_atten(x)
        edge_atten = F.leaky_relu(edge_atten, 0.01)
        edge_atten = softmax(edge_atten, edge_id)
        x = edge_atten*x
        bond_embedding = scatter(x, edge_id, dim=0, reduce="mean")
        bond_embedding = self.bond_lin(bond_embedding)
        bond_embedding = self.bond_lin2(bond_embedding)
        bond_embedding = self.bn(bond_embedding)
        bond_embedding = F.gelu(bond_embedding)
        bond_embedding =  F.dropout(bond_embedding, p=self.dropout, training=self.training) 
        return bond_embedding

class cycle_agg_layer(MessagePassing, nn.Module):
    def __init__(self, input_dim,output_dim, dropout):
        super(cycle_agg_layer, self).__init__()
        self.dropout = dropout
        self.cycle_atten = Linear(input_dim,1,bias=False)
        self.cycle_lin = Linear(input_dim,output_dim,bias=False)
        self.cycle_lin2 = Linear(output_dim,output_dim,bias=False)
        self.bn = torch.nn.BatchNorm1d(output_dim,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.cycle_atten.weight,a=0,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_uniform_(self.cycle_lin.weight,a=0,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_uniform_(self.cycle_lin2.weight,a=0,mode='fan_in',nonlinearity='relu')
        
    def forward(self, x, cycle_vertex_matrix):
        cycle_embedding = self.cycle_agg(x, cycle_vertex_matrix)
        return cycle_embedding

    def cycle_agg(self, x, cycle_info):
        edge_id = torch.tensor(cycle_info[2]).to(device)
        edge_atten = self.cycle_atten(x[cycle_info[3]])
        edge_atten = F.leaky_relu(edge_atten, 0.01)
        edge_atten = softmax(edge_atten, edge_id)
        x = x[cycle_info[3]]
        x = edge_atten*x
        cycle_embedding = scatter(x, edge_id, dim=0, reduce="mean")
        cycle_embedding = self.cycle_lin(cycle_embedding)
        cycle_embedding = self.cycle_lin2(cycle_embedding)
        cycle_embedding = self.bn(cycle_embedding)
        cycle_embedding = F.gelu(cycle_embedding)
        cycle_embedding =  F.dropout(cycle_embedding, p=self.dropout, training=self.training)
        return cycle_embedding
    
class edge_agg_layer(MessagePassing, nn.Module):
    def __init__(self, input_dim,output_dim, dropout):
        super(edge_agg_layer, self).__init__()
        self.dropout = dropout
        self.edge_lin = Linear(input_dim,output_dim,bias=False)
        self.edge_lin2 = Linear(output_dim,output_dim,bias=False)     
        self.lin = Linear(output_dim,output_dim,bias=False)
        self.lin2 = Linear(output_dim,output_dim,bias=False)
        self.bn = torch.nn.BatchNorm1d(output_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.edge_lin.weight,a=0,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_uniform_(self.lin.weight,a=0,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_uniform_(self.edge_lin2.weight,a=0,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_uniform_(self.lin2.weight,a=0,mode='fan_in',nonlinearity='relu')
    
    def forward(self, x, bond, cycle, edge_index, cycle_info):  
        bond_embedding = self.edge_agg(x, bond, cycle, edge_index, cycle_info)
        return bond_embedding
    
    def edge_agg(self, x, bond, cycle, edge_index, cycle_info):
        x = self.edge_lin(x)
        x = self.edge_lin2(x)
        x_out = bond[edge_index[0]]
        x_in = bond[edge_index[1]]
        x_cycle = scatter(cycle[cycle_info[2]], torch.tensor(cycle_info[3]).to(device), dim=0, dim_size = x.shape[0], reduce="sum")
        x = self.lin(x + x_out + x_in + x_cycle)
        x = self.lin2(x)
        x = self.bn(x)
        x = F.gelu(x)
        x =  F.dropout(x, p=self.dropout, training=self.training)
        return x

class graph_agg_layer(MessagePassing, nn.Module):
    def __init__(self, hidden_channels, out_channels,dropout, out_dim):
        super(graph_agg_layer, self).__init__()
        self.dropout = dropout
        self.graph_lin = Linear(hidden_channels, hidden_channels, bias=False)
        self.graph_lin2 = Linear(hidden_channels, out_channels, bias=False)
        self.lin = Linear(out_channels, out_dim, bias=False)
        self.bn = torch.nn.BatchNorm1d(out_channels,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.graph_lin.weight,a=0,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_uniform_(self.graph_lin2.weight,a=0,mode='fan_in',nonlinearity='relu')
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, edge, batch,edge_index):
        edge_batch = batch[edge_index[0]]
        G_embedding = scatter(edge, edge_batch, dim=0, reduce='sum')
        G_embedding = self.graph_lin(G_embedding)
        G_embedding = self.graph_lin2(G_embedding)
        G_embedding = F.gelu(self.bn(G_embedding))
        G_embedding =  F.dropout(G_embedding, p=self.dropout, training=self.training)
        G_embedding = self.lin(G_embedding)
        return G_embedding

class Cybond_Punk_Network(MessagePassing, nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, name, dropout, num_classes):
        super(Cybond_Punk_Network, self).__init__()
        self.name = name
        if self.name == "IMDB-BINARY":
            self.embedding = nn.Embedding(136+1, in_channels)
        elif self.name == "PTC_FR":
            self.embedding = nn.Embedding(19+1, in_channels)
        elif self.name == "MUTAG":
            self.embedding = nn.Embedding(7+1, in_channels)
        elif self.name == "PROTEINS":
            self.embedding = nn.Embedding(3+1, in_channels)
        elif self.name == "ENZYMES":
            self.embedding = nn.Embedding(3+1, in_channels)
        self.dropout = dropout
        self.mlp_feature =  nn.Sequential(
            nn.Linear(2*in_channels, 2*in_channels), 
            nn.Linear(2*in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        output_dim = num_classes

        self.bond_agg_layer = bond_agg_layer(hidden_channels, out_channels, self.dropout)
        self.cycle_agg_layer = cycle_agg_layer(hidden_channels, out_channels, self.dropout)
        self.edge_agg_layer = edge_agg_layer(hidden_channels, out_channels, self.dropout)
        self.graph_agg_layer = graph_agg_layer(out_channels, out_channels, self.dropout, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.mlp_feature[0].weight, a=0,mode='fan_in',nonlinearity='relu')

    def forward(self, sub_x, sub_edge_index, batch,cycle_info):
        if self.name == "IMDB-BINARY":
            sub_x = torch.argmax(sub_x, dim=1, keepdim=True).to(torch.float)
            x = self.embedding(sub_x.int()).squeeze(1)
        elif self.name == "PTC_FR":
            x = self.embedding(sub_x.int()).squeeze(1) 
        elif self.name == "MUTAG":
            x = self.embedding(sub_x.int()).squeeze(1)
        elif self.name == "PROTEINS":
            x = self.embedding(sub_x.int()).squeeze(1)  
        elif self.name == "ENZYMES":
            x = self.embedding(sub_x.int()).squeeze(1)
        x = torch.cat([x[sub_edge_index[0]],x[sub_edge_index[1]]],dim=1)
        x = self.mlp_feature(x)
        bond = self.bond_agg_layer(x, sub_edge_index)
        cycle = self.cycle_agg_layer(x, cycle_info)
        x = self.edge_agg_layer(x, bond, cycle, sub_edge_index, cycle_info)
        output =  self.graph_agg_layer(x, batch,sub_edge_index)
        return output