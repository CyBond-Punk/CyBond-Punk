import os
import torch
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from sklearn.model_selection import KFold
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from model_noedge import Cybond_Punk_Network
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
for kth in range(10):
    data_name = 'MUTAG' #'PTC_FR' 'IMDB-BINARY' 'PROTEINS' 'ENZYMES'

    def Split_10_Fold(datasets,kthfold):  
        pos_train_index = []
        pos_test_index = []
        neg_train_index = []
        neg_test_index = []
        kf = KFold(n_splits=10, shuffle=True, random_state=True)
        for pos_train, pos_test in kf.split(datasets[torch.where(datasets.y==1)[0]]):
            pos_train_index.append(pos_train)
            pos_test_index.append(pos_test)  
        for neg_train, neg_test in kf.split(datasets[torch.where(datasets.y==0)[0]]):
            neg_train_index.append(neg_train)
            neg_test_index.append(neg_test)
        
        pos_train_id = torch.where(datasets.y==1)[0][pos_train_index[kthfold]]
        neg_train_id = torch.where(datasets.y==0)[0][neg_train_index[kthfold]]

        pos_test_id = torch.where(datasets.y==1)[0][pos_test_index[kthfold]]
        neg_test_id = torch.where(datasets.y==0)[0][neg_test_index[kthfold]]

        train_id = torch.cat([pos_train_id,neg_train_id])
        test_id = torch.cat([pos_test_id,neg_test_id])

        trainset = datasets[train_id]
        testset = datasets[test_id]

        return trainset,testset

    class OneHotToIndex(object):
            def __call__(self, data):
                if data.x is None:
                    raise ValueError("Data object does not have node features (x).")
                data.x = torch.argmax(data.x, dim=1, keepdim=True).to(torch.float)
                return data

    class NormalizedDegree:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, data):
            deg = degree(data.edge_index[0], dtype=torch.float)
            deg = (deg - self.mean) / self.std
            data.x = deg.view(-1, 1)
            return data
    
    def initializeNodes(dataset):
        if dataset.data.x is None:
            max_degree = 0
            degs = []
            for data in dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            if max_degree < 1000:
                dataset.transform = T.OneHotDegree(max_degree)
            else:
                deg = torch.cat(degs, dim=0).to(torch.float)
                mean, std = deg.mean().item(), deg.std().item()
                dataset.transform = NormalizedDegree(mean, std)
        
    if data_name == 'IMDB-BINARY':
        datasets = TUDataset(root='data/TUDataset', name=data_name)
        initializeNodes(datasets)
        nclass = 2
    
    elif data_name == 'ENZYMES':
        transform_PROT = OneHotToIndex()
        datasets = TUDataset(root='data/TUDataset', name=data_name, transform=transform_PROT)
        nclass = 6

    elif data_name == 'PROTEINS':
        transform_PROT = OneHotToIndex()
        datasets = TUDataset(root='data/TUDataset', name=data_name, transform=transform_PROT)
        nclass = 2

    elif data_name == 'PTC_FR':
        transform_PTC = OneHotToIndex()
        datasets = TUDataset(root='data/TUDataset', name=data_name, transform=transform_PTC)
        nclass = 2

    elif data_name == 'MUTAG':
        transform_MUTAG = OneHotToIndex()
        datasets = TUDataset(root='data/TUDataset', name=data_name, transform=transform_MUTAG)
        nclass = 2

    train_dataset, test_dataset= Split_10_Fold(datasets,kth)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    epoch_num = 300
    in_channels = 32
    hidden_channels = 128
    out_channels = 32
    task_type = 'classification'
    dropout = 0.0
    lr = 0.01
    weightdecay = 0

    model = Cybond_Punk_Network(in_channels, hidden_channels, out_channels, data_name, dropout, nclass).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay= weightdecay)

    def train(loader):
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for data in loader:
            data = data.to(device)
            cycle_info = data_cycle(find_cycles_in_graph(data),data.edge_index)
            out = model(data.x, data.edge_index, data.batch, cycle_info)

            loss = criterion(out, data.y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(data)
            preds = out.argmax(dim=1)
            correct += (preds == data.y.squeeze()).sum().item()
            total += data.y.size(0)
        return total_loss / total, correct/total

    def test(loader):
        model.eval()
        total_loss = 0
        total = 0
        correct = 0
        
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                cycle_info = data_cycle(find_cycles_in_graph(data),data.edge_index)
                out = model(data.x, data.edge_index, data.batch, cycle_info)

                loss = criterion(out, data.y.to(device))
                total_loss += loss.item() * len(data)
                preds = out.argmax(dim=1)
                correct += (preds == data.y.squeeze()).sum().item()
                total += data.y.size(0)
            return total_loss / total, correct/total

    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    for epoch in range(epoch_num):
        train_loss, train_accuracy = train(train_loader)
        test_loss, test_accuracy = test(test_loader)
        test_acc.append( test_accuracy)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f},Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    print('k',kth)
    print("best acc:", max(test_acc))
    avg_acc.append(max(test_acc))
print(avg_acc,np.mean(avg_acc),np.std(avg_acc))