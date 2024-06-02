import networkx as nx
import torch
from torch_geometric.utils import to_networkx


def find_cycles_in_graph(data):

    G = to_networkx(data, to_undirected=True)

    cycles = nx.cycle_basis(G)

    cycle_vertex_dict = {i: cycle for i, cycle in enumerate(cycles)}

    return cycle_vertex_dict

def data_cycle(data, edge_indexs):
    out_list = []
    in_list = []
    edge_index = []
    index_list = []
    for key, value in data.items():
        out_list = out_list + value
        in_list = in_list + value[1:] + [value[0]]
        out_tensor = torch.tensor(value).unsqueeze(1).to('cuda')
        in_tensor = torch.tensor(value[1:]+[value[0]]).unsqueeze(1).to('cuda')
        edge_id = ((edge_indexs[0]==out_tensor)&(edge_indexs[1]==in_tensor))
        edge_index = edge_index + edge_id.nonzero(as_tuple=False)[:,1].to('cpu').tolist()
        index_list = index_list + len(value)*[key]
    
    return [out_list,in_list,index_list,edge_index]