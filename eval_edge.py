import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from torch_geometric.datasets import ZINC, LRGBDataset
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch_geometric.loader import DataLoader
from model_edge import Bond_Cycle_Network
from utils import find_cycles_in_graph,data_cycle
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


data_name = 'ZINC' #'ZINC'  #'Peptides-func' 'Peptides-func'

scaler_node = StandardScaler()
class scale(object):
    def __call__(self, data):
        scaler_node = StandardScaler()
        data.x = scaler_node.fit_transform(data.x)
        scaler_edge = StandardScaler()
        data.edge_attr = scaler_edge.fit_transform(data.edge_attr)
        return data
    
if data_name == 'ZINC':
    task_type = 'regression'
    nclass = 1
    transformer_pep = scale()
    test_dataset = ZINC(root='/home/zzh/cycle/data/ZINC', split='test',subset=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

elif data_name == 'Peptides-func':
    task_type = 'classification'
    nclass = 10
    transformer_pep = scale()
    test_dataset = LRGBDataset(root='/home/zzh/cycle/data/LRGB',name = data_name, split='test',transform=transformer_pep)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

elif data_name == 'Peptides-struct':
    task_type = 'regression'
    nclass = 11
    transformer_pep = scale()
    test_dataset = LRGBDataset(root='/home/zzh/cycle/data/LRGB',name = data_name, split='test',transform=transformer_pep)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


in_channels = 64
hidden_channels = 128
out_channels = 64
dropout = 0.0

model = Bond_Cycle_Network(in_channels, hidden_channels, out_channels, data_name, dropout, task_type, nclass).to(device)
model.load_state_dict(torch.load('/home/zzh/2024Neurips/save_model/model_parameters.pth'))
if task_type == 'regression':
    criterion = torch.nn.L1Loss(reduction='mean')


def test(loader,data_name):
    model.eval()
    total_loss = 0
    if task_type == 'classification':
        total_loss = 0
        all_preds = []
        all_targets = []
    
    with torch.no_grad():
        N = 0
        for data in loader:
            data = data.to(device)
            if data_name != 'ZINC':
                data.x = torch.tensor(np.concatenate(data.x,axis=0)).to(torch.float32).to(device)
                data.edge_attr = torch.tensor(np.concatenate(data.edge_attr,axis=0)).to(torch.float32).to(device)              
            cycle_info = data_cycle(find_cycles_in_graph(data),data.edge_index)
            out = model(data.x, data.edge_index, data.batch, cycle_info, data.edge_attr) 
        
            if task_type == 'regression':
                loss = criterion(out, data.y)
            elif task_type == 'classification':
                loss = F.binary_cross_entropy_with_logits(out, data.y.to(device))
            total_loss += loss.item() * len(data)
            if task_type == 'classification':
                all_preds.append(out)
                all_targets.append(data.y)
            N += len(data.y) 
        if task_type == 'classification':
            all_preds = torch.cat(all_preds, dim=0).cpu().detach().numpy()
            all_targets = torch.cat(all_targets, dim=0).cpu().detach().numpy()
            ap_scores = []
            for i in range(all_targets.shape[1]):
                ap = average_precision_score(all_targets[:, i], all_preds[:, i])
                ap_scores.append(ap)
            mean_ap = np.mean(ap_scores)
            return total_loss / N, mean_ap
        return total_loss / N

if task_type == 'classification':
    test_loss, test_accuracy = test(test_loader,data_name)
    print(f'Test Loss: {test_loss:.4f}, Test AP: {test_accuracy:.4f}')
else:
    test_loss = test(test_loader,data_name)
    print(f'Test Loss: {test_loss:.4f}')