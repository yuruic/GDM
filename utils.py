from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, DenseDataLoader
import torch
import torch.nn.functional as F
import os.path as osp
# from torch_geometric.datasets import MNISTSuperpixels
# from torch.utils.data import random_split, Subset
import numpy as np
from torch_geometric.data import Data
# from dgl.data import BA2MotifDataset
import load_data
import torch
# from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import random

class Complete(object):
    def __call__(self, data):
        if data.x is None:
            if hasattr(data, 'adj'):
                data.x = data.adj.sum(1).view(-1, 1)
            else:
                adj = to_scipy_sparse_matrix(data.edge_index).sum(1)
                data.x = torch.FloatTensor(adj.sum(1)).view(-1, 1)
        return data

class RemoveEdgeAttr(object):
    def __call__(self, data):
        if data.edge_attr is not None:
            data.edge_attr = None
        if data.x is None:
            if hasattr(data, 'adj'):
                data.x = data.adj.sum(1).view(-1, 1)
            else:
                adj = to_scipy_sparse_matrix(data.edge_index).sum(1)
                data.x = torch.FloatTensor(adj.sum(1)).view(-1, 1)

        data.y = data.y.squeeze(0)
        data.x = data.x.float()
        return data

class ConcatPos(object):
    def __call__(self, data):
        if data.edge_attr is not None:
            data.edge_attr = None
        data.x = torch.cat([data.x, data.pos], dim=1)
        data.pos = None
        return data

class Dataset:

    def __init__(self, args):
        # random seed setting
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        name = args.dataset
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'datasets', f'{name}')

        if name in ['DD', 'MUTAG', 'NCI1']:
            dataset = TUDataset(path, name=name, transform=T.Compose([Complete()]), use_node_attr=True)
            dataset = dataset.shuffle()
            n = (len(dataset) + 9) // 10
            test_dataset = dataset[:n]
            val_dataset = dataset[n:2 * n]
            train_dataset = dataset[2 * n:]
            nnodes = [x.num_nodes for x in dataset]
            print('mean #nodes:', np.mean(nnodes), 'max #nodes:', np.max(nnodes))

            y_final = [g.y.item() for g in test_dataset]
            from collections import Counter; counter=Counter(y_final); print(counter)
            print("#Majority guessing:", sorted(counter.items())[-1][1]/len(y_final))

            test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
            train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

            train_datalist = np.ndarray((len(train_dataset),), dtype=object)
            for ii in range(len(train_dataset)):
                train_datalist[ii] = train_dataset[ii]
            self.packed_data = [train_dataset, train_loader, val_loader, test_loader, train_datalist]


        if name in ['BA']:
            #dataset = load_data.get_dataset(dataset_dir='./datasets', dataset_name='BA_2Motifs')
            #dataset.process()
            dataset = load_data.get_dataset(dataset_dir='./datasets', dataset_name='BA_2Motifs')
            train_dataset,train_datalist, train_loader, val_loader, test_loader = load_data.get_dataloader(dataset,512,True,[0.7,0.2,0.1],42)
            self.packed_data = [train_dataset, train_loader, val_loader, test_loader, train_datalist]

        if name in ['Graph-SST5']:
            #dataset = load_data.get_dataset(dataset_dir='./datasets', dataset_name='Graph-SST5')
            #dataset.process()
            dataset = load_data.get_dataset(dataset_dir='./datasets', dataset_name='Graph-SST5', task=None)
            train_dataset,train_datalist, train_loader, val_loader, test_loader = load_data.get_dataloader(dataset,512,True,[0.7,0.2,0.1],42)
            self.packed_data = [train_dataset, train_loader, val_loader, test_loader, train_datalist]

        if name in ['Graph-SST2']:
            #dataset = load_data.get_dataset(dataset_dir='./datasets', dataset_name='Graph-SST5')
            #dataset.process()
            dataset = load_data.get_dataset(dataset_dir='./datasets', dataset_name='Graph-SST2', task=None)
            train_dataset,train_datalist, train_loader, val_loader, test_loader = load_data.get_dataloader(dataset,512,True,[0.7,0.2,0.1],42)
            self.packed_data = [train_dataset, train_loader, val_loader, test_loader, train_datalist]

        if name in ['Graph-Twitter']:
            #dataset = load_data.get_dataset(dataset_dir='./datasets', dataset_name='Graph-SST5')
            #dataset.process()
            dataset = load_data.get_dataset(dataset_dir='./datasets', dataset_name='Graph-Twitter', task=None)
            train_dataset,train_datalist, train_loader, val_loader, test_loader = load_data.get_dataloader(dataset,512,True,[0.7,0.2,0.1],42)
            self.packed_data = [train_dataset, train_loader, val_loader, test_loader, train_datalist]

        if name in ['ba_lrp']:
            #dataset = load_data.get_dataset(dataset_dir='./datasets', dataset_name='ba_lrp')
            #dataset.process()
            dataset = load_data.get_dataset(dataset_dir='./datasets', dataset_name='ba_lrp')
            train_dataset,train_datalist, train_loader, val_loader, test_loader = load_data.get_dataloader(dataset,512,True,[0.7,0.2,0.1],42)
            self.packed_data = [train_dataset, train_loader, val_loader, test_loader, train_datalist] 

        if name in ['BA_Community']:
            dataset = load_data.get_dataset(dataset_dir='./datasets', dataset_name='BA_Community')
            dataset.process()
            #dataset = load_data.get_dataset(dataset_dir='./datasets', dataset_name='BA_Community')
            print(dataset,len(dataset))
            train_dataset,train_datalist, train_loader, val_loader, test_loader = load_data.get_dataloader(dataset,512,True,[0.7,0.2,0.1],42)
            self.packed_data = [train_dataset, train_loader, val_loader, test_loader, train_datalist]           


        if name in ['acyclic']:
            class Dataset(torch.utils.data.Dataset):
                def __init__(self, data, labels):
                    'Initialization'
                    self.y = labels
                    self.data = data

                def __len__(self):
                    'Denotes the total number of samples'
                    return len(self.data)

                def __getitem__(self, index):
                    'Generates one sample of data'
                    # Select sample
                    X = self.data[index]
                    y = self.y[index]
                    return X, y

            import networkx as nx
            from torch_geometric.utils import from_networkx
            dataset = []
            labels = []
            for i in range(5,10):
                for j in range(5,10):
                    ## 25
                    edge_index = from_networkx(nx.grid_2d_graph(i, j)).edge_index
                    source = F.one_hot(torch.arange(0, i*j) % i*j)
                    x = F.pad(input=source, pad=(0, 128-(i-1)*j-1, 1, 0), mode='constant', value=0)
                    x = x.float()
                    dataset.append(Data(edge_index=edge_index, x=x,y=0))
                    labels.append(0)
            for i in range(5,55):
                ## 60
                edge_index = from_networkx(nx.cycle_graph(i)).edge_index
                source = F.one_hot(torch.arange(0, i) % i)
                x = F.pad(input=source, pad=(0, 128-i, 1, 0), mode='constant', value=0)
                x = x.float()
                dataset.append(Data(edge_index=edge_index,x=x,y=0))
                labels.append(0)
            for i in range(15):
                ## 15
                edge_index = from_networkx(nx.cycle_graph(3)).edge_index
                source = F.one_hot(torch.arange(0, 3) % 3)
                x = F.pad(input=source, pad=(0, 128-3, 1, 0), mode='constant', value=0)
                x = x.float()
                dataset.append(Data(edge_index=edge_index,x=x,y=0))
                labels.append(0)
            for i in range(5,15):
                ## 10
                edge_index = from_networkx(nx.wheel_graph(i)).edge_index
                source = F.one_hot(torch.arange(0, i) % i)
                x = F.pad(input=source, pad=(0, 128-i, 1, 0), mode='constant', value=0)
                x = x.float()
                dataset.append(Data(edge_index=edge_index, x=x,y=0))
                labels.append(0)
            for i in range(5,15):
                ## 10
                edge_index = from_networkx(nx.circular_ladder_graph(i)).edge_index
                source = F.one_hot(torch.arange(0, 2*(i)) % 2*i)
                x = F.pad(input=source, pad=(0, 128-i-1, 1, 0), mode='constant', value=0)
                x = x.float()
                dataset.append(Data(edge_index=edge_index, x=x,y=0))
                labels.append(0)
            for i in range(3,7):
                ## 4
                edge_index = from_networkx(nx.binomial_tree(i)).edge_index
                source = F.one_hot(torch.arange(0, 2**i) % 2**i)
                x = F.pad(input=source, pad=(0, 128-2**i, 1, 0), mode='constant', value=0)
                x = x.float()
                dataset.append(Data(edge_index=edge_index, x=x,y=1))
                labels.append(1)
            for r in range(5,8):
                ## 3*25
                for i in range(5,30):
                    edge_index = from_networkx(nx.full_rary_tree(r, i, create_using=None)).edge_index
                    source = F.one_hot(torch.arange(0, i) % i)
                    x = F.pad(input=source, pad=(0, 128-i, 1, 0), mode='constant', value=0)
                    x = x.float()
                    dataset.append(Data(edge_index=edge_index, x=x,y=1))
                    labels.append(1)

            n = (len(dataset) + 9) // 10
            np.random.seed(42)
            random.shuffle(dataset)
            test= (dataset[:n])
            eval = (dataset[n:2 * n])
            train = (dataset[2 * n:])
            test_loader = DataLoader(test, batch_size=1024, shuffle=False)
            val_loader = DataLoader(eval, batch_size=1024, shuffle=False)
            train_loader = DataLoader(train, batch_size=512, shuffle=True)
            train_datalist = np.ndarray((len(train),),dtype=object)
            for ii in range(len(train)):
                train_datalist[ii] = train[ii]
            self.packed_data = [train, train_loader, val_loader, test_loader, train_datalist]


        if name in ['shape']:
            class Dataset(torch.utils.data.Dataset):
                def __init__(self, data, labels):
                    'Initialization'
                    self.y = labels
                    self.data = data

                def __len__(self):
                    'Denotes the total number of samples'
                    return len(self.data)

                def __getitem__(self, index):
                    'Generates one sample of data'
                    # Select sample
                    X = self.data[index]
                    y = self.y[index]
                    return X, y

            import networkx as nx
            from torch_geometric.utils import from_networkx
            dataset = []
            labels = []
            for i in range(3,8):
                for j in range(3,8):
                    ## 25
                    edge_index = from_networkx(nx.grid_2d_graph(i, j)).edge_index
                    source = F.one_hot(torch.arange(0, i*j+1) % i*j)
                    x = F.pad(input=source, pad=(0, 128-(i-1)*j-1, 0, 0), mode='constant', value=0)
                    x = x.float()
                    dataset.append(Data(edge_index=edge_index, x=x,y=0))
                    labels.append(0)
            for _ in range(5):
                for i in range(30,35):
                ## 25
                    edge_index = from_networkx(nx.wheel_graph(i)).edge_index
                    source = F.one_hot(torch.arange(0, i+1) % i)
                    x = F.pad(input=source, pad=(0, 128-i, 0, 0), mode='constant', value=0)
                    x = x.float()
                    dataset.append(Data(edge_index=edge_index, x=x,y=1))
                    labels.append(0)
            for j in range(5):
                for i in range(30,35):
                ## 25
                    edge_index = from_networkx(nx.lollipop_graph(i,j)).edge_index
                    source = F.one_hot(torch.arange(0, i+j+1) % (i+j))
                    x = F.pad(input=source, pad=(0, 128-i-j, 0, 0), mode='constant', value=0)
                    x = x.float()
                    dataset.append(Data(edge_index=edge_index, x=x,y=2))
                    labels.append(0)

            for j in range(5):
                for i in range(30,35):
                ## 25
                    edge_index = from_networkx(nx.star_graph(i)).edge_index
                    source = F.one_hot(torch.arange(0, i+2) % (i+1))
                    x = F.pad(input=source, pad=(0, 128-i-1, 0, 0), mode='constant', value=0)
                    x = x.float()
                    dataset.append(Data(edge_index=edge_index, x=x,y=3))
                    labels.append(0)



            n = (len(dataset) + 9) // 10
            np.random.seed(42)
            random.shuffle(dataset)
            test= (dataset[:n])
            eval = (dataset[n:2 * n])
            train = (dataset[2 * n:])
            test_loader = DataLoader(test, batch_size=1024, shuffle=False)
            val_loader = DataLoader(eval, batch_size=1024, shuffle=False)
            train_loader = DataLoader(train, batch_size=512, shuffle=True)
            train_datalist = np.ndarray((len(train),),dtype=object)
            for ii in range(len(train)):
                train_datalist[ii] = train[ii]
            self.packed_data = [train, train_loader, val_loader, test_loader, train_datalist]

class TensorDataset(Dataset):
    def __init__(self, feat, adj, labels): # images: n x c x h x w tensor
        self.x = feat.detach()
        self.adj = adj.detach()
        self.y = labels.detach()

    def __getitem__(self, index):
        return self.x[index], self.adj[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

class SparseTensorDataset(Dataset):
    def __init__(self, data): # images: n x c x h x w tensor
        self.data  = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_max_nodes(args):
    if args.dataset == 'CIFAR10':
        return 150
    if args.dataset == 'DD':
        return 5748
    if args.dataset == 'MUTAG':
        return 28
    if args.dataset == 'NCI1':
        return 111
    if args.dataset == 'ogbg-molhiv':
        return 222
    raise NotImplementedError

def get_mean_nodes(args):
    if args.dataset == 'CIFAR10':
        return 118
    if args.dataset == 'DD':
        return 285
    if args.dataset == 'MUTAG':
        return 18
    if args.dataset == 'NCI1':
        return 30
    if args.dataset == 'ogbg-molhiv':
        return 26
    if args.dataset == 'ogbg-molbbbp':
        return 24
    if args.dataset == 'ogbg-molbace':
        return 34

    raise NotImplementedError


def match_loss(gw_syn, gw_real, args, device):
    dis = torch.tensor(0.0).to(device)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2) / torch.sum((gw_real_vec)**2) # I used this a lot

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s'%args.dis_metric)

    return dis


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


def save_pyg_graphs(graphs, args):
    memory_dict = {}
    for d in graphs:
        y = d.y.item()
        if y not in memory_dict:
            memory_dict[y] = [d]
        else:
            memory_dict[y].append(d)

    for k, v in memory_dict.items():
        graph_dict = {}
        d, slices = InMemoryDataset.collate(v)
        graph_dict['x'] = d.x
        graph_dict['edge_index'] = d.edge_index
        graph_dict['y'] = d.y
        memory_dict[k] = (graph_dict, slices)

    torch.save(memory_dict, f'saved/memory/{args.dataset}_ours_{args.seed}_ipc{args.ipc}.pt')

