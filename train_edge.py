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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.autograd.set_detect_anomaly(True)
setup_seed(42)


avg_acc = []
for kth in range(5):
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
        train_dataset = ZINC(root='/home/zzh/cycle/data/ZINC', split='train', subset=True)
        test_dataset = ZINC(root='/home/zzh/cycle/data/ZINC', split='val',subset=True)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
        epoch_num = 2000
        warm_num = 50
    elif data_name == 'Peptides-func':
        task_type = 'classification'
        nclass = 10
        transformer_pep = scale()
        train_dataset = LRGBDataset(root='/home/zzh/cycle/data/LRGB',name = data_name, split='train',transform=transformer_pep)
        test_dataset = LRGBDataset(root='/home/zzh/cycle/data/LRGB',name = data_name, split='val',transform=transformer_pep)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
        epoch_num = 200
        warm_num = 5
    elif data_name == 'Peptides-struct':
        task_type = 'regression'
        nclass = 11
        transformer_pep = scale()
        train_dataset = LRGBDataset(root='/home/zzh/cycle/data/LRGB',name = data_name, split='train',transform=transformer_pep)
        test_dataset = LRGBDataset(root='/home/zzh/cycle/data/LRGB',name = data_name, split='val',transform=transformer_pep)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
        epoch_num = 200
        warm_num = 5


    in_channels = 64
    hidden_channels = 128
    out_channels = 64
    dropout = 0.0

    model = Bond_Cycle_Network(in_channels, hidden_channels, out_channels, data_name, dropout, task_type, nclass).to(device)
    params = list(model.parameters())
    num_params = sum([p.numel() for p in params])
    print("模型参数量：{}个".format(num_params))
    if task_type == 'regression':
        criterion = torch.nn.L1Loss(reduction='mean')

    class CombinedLRScheduler:
        def __init__(self, warmup_scheduler, cosine_scheduler, warmup_steps):
            self.warmup_scheduler = warmup_scheduler
            self.cosine_scheduler = cosine_scheduler
            self.warmup_steps = warmup_steps
            self.current_step = 0

        def step(self):
            if self.current_step < self.warmup_steps:
                self.warmup_scheduler.step()
            else:
                self.cosine_scheduler.step()
            self.current_step += 1
    
    warmup_steps = len(train_loader)*warm_num
    total_steps = len(train_loader)*epoch_num
    cosine_steps = total_steps - warmup_steps
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


    def warmup_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps,eta_min=1e-6)

    combined_scheduler = CombinedLRScheduler(warmup_scheduler, cosine_scheduler, warmup_steps)

    def train(loader,data_name):
        model.train()
        total_loss = 0
        if task_type == 'classification':
            total_loss = 0
            all_preds = []
            all_targets = []
        N = 0
        for data in loader:
            data = data.to(device)
            if data_name != 'ZINC':
                data.x = torch.tensor(np.concatenate(data.x,axis=0)).to(torch.float32).to(device)
                data.edge_attr = torch.tensor(np.concatenate(data.edge_attr,axis=0)).to(torch.float32).to(device)
            cycle_info = data_cycle(find_cycles_in_graph(data),data.edge_index)
            out = model(data.x, data.edge_index, data.batch, cycle_info,data.edge_attr)
            
            if task_type == 'regression':
                loss = criterion(out, data.y.to(device))
            elif task_type == 'classification':
                loss = F.binary_cross_entropy_with_logits(out, data.y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            combined_scheduler.step()
            total_loss += loss.item() * len(data)
            N += len(data.y) 
            if task_type == 'classification':
                all_preds.append(out)
                all_targets.append(data.y)
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

    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    best_acc = 0
    best_mae = 10000
    for epoch in range(epoch_num):
        if task_type == 'classification':
            train_loss, train_accuracy = train(train_loader,data_name)
            train_losses.append(train_loss)
        else:
            train_loss = train(train_loader,data_name)
            train_losses.append(train_loss)
        if task_type == 'classification':
            test_loss, test_accuracy = test(test_loader,data_name)
            if best_acc < test_accuracy:
                best_acc = test_accuracy
                torch.save(model.state_dict(), '/home/zzh/2024Neurips/save_model/model_parameters.pth')
            test_losses.append(test_loss)
            test_acc.append( test_accuracy)
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train AP: {train_accuracy:.4f},Test Loss: {test_loss:.4f}, Test AP: {test_accuracy:.4f}')
        else:
            test_loss = test(test_loader,data_name)
            test_losses.append(test_loss)
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    print('k',kth)
    if task_type == 'classification':
        print("best acc:", max(test_acc))
        avg_acc.append(max(test_acc))
    else:
        print("best mae:", min(test_losses))
        avg_acc.append(min(test_losses))
print(avg_acc,np.mean(avg_acc),np.std(avg_acc))