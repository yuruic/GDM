import torch
import torch.nn.functional as F
from utils import *
import argparse
from models_gcn import GCN
from models import DenseGCN
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.data import Batch, Data
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import json

# =====================
# Config Loading
# =====================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# =====================
# Argument Parsing
# =====================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUTAG', help='Dataset name (MUTAG, BA, shape, etc.)')
parser.add_argument('--ipc', type=int, default=10, help='Images per class')
parser.add_argument('--dis_metric', type=str, default='mse', help='distance metric')
parser.add_argument('--pooling', type=str, default='mean')
parser.add_argument('--init', type=str, default='real')
parser.add_argument('--beta', type=float, help='coefficient for the regularization term sparsity loss')
parser.add_argument('--gama', type=float, help='coefficient for the regularization term feature loss')
parser.add_argument('--theta', type=float, default=.1, help='coefficient for the regularization term edge loss')
parser.add_argument('--lr_adj', type=float, help='Learning rate for adjacency matrix')
parser.add_argument('--lr_feat', type=float, help='Learning rate for features')
parser.add_argument('--net_norm', type=str, default='none')
parser.add_argument('--nconvs', type=int, default=3)
parser.add_argument('--nfeat', type=int, help='Number of features')
parser.add_argument('--bs_cond', type=int, default=256)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--nclass', type=int, help='Number of classes')
parser.add_argument('--nnodes_syn', type=int, help='Number of synthetic nodes')
parser.add_argument('--inner_loop', type=int, help='Inner loop iterations')
parser.add_argument('--outer_loop', type=int, help='Outer loop iterations')
parser.add_argument('--epochs_last', type=int, help='Last training epochs')
parser.add_argument('--epochs_test', type=int, help='Test epochs')
parser.add_argument('--alpha_last', type=int, help='Alpha for last training')
parser.add_argument('--epoch_intrain', type=int, default=20)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--lr_model', type=float, help='Model learning rate')
parser.add_argument('--big_graph', type=float, help='big graph or small(compared to mean')
parser.add_argument('--stru_discrete', type=int, default=1)
parser.add_argument('--reduction_rate_vis', type=float, help='Reduction rate for visualization')

args = parser.parse_args()

# Set default values from config if not provided
if args.beta is None:
    args.beta = config[args.dataset]['beta']
if args.gama is None:
    args.gama = config[args.dataset]['gama']
if args.lr_adj is None:
    args.lr_adj = config[args.dataset]['lr_adj']
if args.lr_feat is None:
    args.lr_feat = config[args.dataset]['lr_feat']
if args.nfeat is None:
    args.nfeat = config[args.dataset]['nfeat']
if args.nclass is None:
    args.nclass = config[args.dataset]['nclass']
if args.nnodes_syn is None:
    args.nnodes_syn = config[args.dataset]['nnodes_syn']
if args.inner_loop is None:
    args.inner_loop = config[args.dataset]['inner_loop']
if args.outer_loop is None:
    args.outer_loop = config[args.dataset]['outer_loop']
if args.epochs_last is None:
    args.epochs_last = config[args.dataset]['epochs_last']
if args.epochs_test is None:
    args.epochs_test = config[args.dataset]['epochs_test']
if args.alpha_last is None:
    args.alpha_last = config[args.dataset]['alpha_last']
if args.lr_model is None:
    args.lr_model = config[args.dataset]['lr_model']
if args.big_graph is None:
    args.big_graph = config[args.dataset]['big_graph']
if args.reduction_rate_vis is None:
    args.reduction_rate_vis = config[args.dataset]['reduction_rate_vis']

if args.dataset == 'MUTAG' and args.ipc == 50:
    args.ipc = 20

# =====================
# Data Loading
# =====================
data = Dataset(args)
packed_data = data.packed_data

# =====================
# Utility Functions
# =====================
def prepare_train_indices(data, args):
    set = data[0]
    indices_class = {}
    nnodes_all = []
    for ix, single in enumerate(set):
        if args.dataset == 'acyclic' or args.dataset == 'shape':
            c = single.y
        else:
            c = single.y.item()
        if c not in indices_class:
            indices_class[c] = [ix]
        else:
            indices_class[c].append(ix)
        nnodes_all.append(single.num_nodes)
    nnodes_all = np.array(nnodes_all)
    real_indices_class = indices_class
    return nnodes_all, real_indices_class

def sample_graphs(data, real_indices_class, nnodes_all, c, batch_size, max_node_size=None, to_dense=False, idx_selected=None):
    """sample random batch_size images from class c"""
    device = 'cuda'
    if idx_selected is None:
        if max_node_size is None:
            idx_shuffle = np.random.permutation(real_indices_class[c])[:batch_size]
            sampled = data[4][idx_shuffle]
        else:
            indices = np.array(real_indices_class[c])[nnodes_all[real_indices_class[c]] <= max_node_size]
            idx_shuffle = np.random.permutation(indices)[:batch_size]
            sampled = data[4][idx_shuffle]
    else:
        sampled = data[4][idx_selected]
    data = Batch.from_data_list(sampled)
    if to_dense:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x, mask = to_dense_batch(x, batch=batch, max_num_nodes=max_node_size)
        adj = to_dense_adj(edge_index, batch=batch, max_num_nodes=max_node_size)
        y = data.y
        return x.to(device), adj.to(device), y.to(device)
    else:
        return data.to(device)

def get_discrete_graphs(adj, init, args, inference, device='cuda'):
    cnt = 0
    adj = (adj.transpose(1, 2) + adj) / 2
    if not inference:
        N = adj.size()[1]
        vals = torch.rand(adj.size(0) * N * (N + 1) // 2)
        vals = vals.view(adj.size(0), -1).to(device)
        i, j = torch.triu_indices(N, N)
        epsilon = torch.zeros_like(adj)
        epsilon[:, i, j] = vals
        epsilon.transpose(1, 2)[:, i, j] = vals
        tmp = torch.log(epsilon) - torch.log(1 - epsilon)
        adj = tmp + adj
        t0 = 1
        tt = 0.01
        end_iter = 200
        t = t0 * (tt / t0) ** (cnt / end_iter)
        t = max(t, tt)
        adj = torch.sigmoid(adj / t)
        adj = adj * (1 - torch.eye(adj.size(1)).to(device))
        return adj
    else:
        adj_ori = torch.sigmoid(adj)
        adj_ori = adj_ori * (1 - torch.eye(adj_ori.size(1)).to(device))
        adj_ori[adj_ori > 0.5] = 1
        adj_ori[adj_ori <= 0.5] = 0
        res = torch.rand(size=(adj.size(0), adj.size(1), adj.size(2)), dtype=torch.float, device=device)
        reduction_rate = args.reduction_rate_vis
        index_len = int(adj.size(1) * adj.size(2) * reduction_rate)
        counter = 0
        for i in adj:
            i = torch.sigmoid(i)
            i = i * (1 - torch.eye(i.size(1)).to(device))
            if args.reduction_rate_vis or init == False:
                temp = np.sort(i.cpu().detach().numpy(), axis=None)[::-1][index_len]
            else:
                temp = 0.5
            i[i > temp] = 1
            i[i <= temp] = 0
            res[counter, :, :] = i
            counter += 1
        adj = res
        adj = adj * (1 - torch.eye(adj.size(1)).to(device))
        return adj_ori, adj

def data_convert(feat_syn, adj_syn, labels_syn):
    sampled = np.ndarray((adj_syn.size(0),), dtype=object)
    for i in range(len(feat_syn)):
        x = feat_syn[i]
        adj = adj_syn[i]
        y = labels_syn[i]
        g = adj.nonzero().T
        single_data = Data(x=x, edge_index=g, y=y)
        sampled[i] = single_data
    return sampled

# =====================
# Training and Evaluation Functions
# =====================

def train_inner(real_samples, model_real, args, optimizer, epochs=None, save=False, verbose=False):
    if epochs is None:
        epochs = args.epoch_intrain
    device = 'cuda'
    real_samples_all = []
    for c in range(args.nclass):
        real_samples_all += Batch.to_data_list(real_samples[str(c)])
    dst_real_train = SparseTensorDataset(real_samples_all)
    train_loader_real = DataLoader(dst_real_train, batch_size=128, shuffle=True, num_workers=0)
    for _ in range(epochs):
        model_real.train()
        for data_real in train_loader_real:
            optimizer.zero_grad()
            output_real = model_real(data_real, mask=None)
            y_real = data_real.y.type(torch.LongTensor).to(device)
            loss = F.nll_loss(output_real, y_real.view(-1))
            loss.backward()
            optimizer.step()

def GDM_training(packed_data, nnodes_all, real_indices_class, args, device='cuda'):
    n = args.ipc * args.nclass
    adj_syn = torch.rand(size=(n, args.nnodes_syn, args.nnodes_syn), dtype=torch.float, requires_grad=True, device=device)
    feat_syn = torch.rand(size=(n, args.nnodes_syn, args.nfeat), dtype=torch.float, requires_grad=True, device=device)
    syn_class_indices = {i: [i*args.ipc, (i+1)*args.ipc] for i in range(args.nclass)}
    labels_syn = torch.LongTensor([[i]*args.ipc for i in range(args.nclass)]).to(device).view(-1)
    if args.init == 'real':
        for c in range(args.nclass):
            ind = syn_class_indices[c]
            feat_real, adj_real, labels_real = sample_graphs(packed_data, real_indices_class, nnodes_all, c, batch_size=ind[1]-ind[0], max_node_size=None, to_dense=True)
            while feat_syn.shape[1] > feat_real.shape[1]:
                random.seed(1)
                feat_real, adj_real, labels_real = sample_graphs(packed_data, real_indices_class, nnodes_all, c, batch_size=ind[1]-ind[0], max_node_size=None, to_dense=True)
            feat_syn.data[ind[0]: ind[1]] = feat_real[:, :args.nnodes_syn].detach().data
            adj_syn.data[ind[0]: ind[1]] = adj_real[:, :args.nnodes_syn, :args.nnodes_syn].detach().data
        sparsity = adj_syn.mean().item()
        if args.stru_discrete:
            adj_syn.data.copy_(adj_syn*10-5)
    else:
        if args.stru_discrete:
            adj_init = torch.log(adj_syn) - torch.log(1-adj_syn)
            adj_init = adj_init.clamp(-10, 10)
            adj_syn.data.copy_(adj_init)
    optimizer_adj = torch.optim.Adam([adj_syn], lr=args.lr_adj)
    optimizer_feat = torch.optim.Adam([feat_syn], lr=args.lr_feat)
    for _ in range(args.outer_loop):
        model_syn = DenseGCN(nfeat=args.nfeat, nhid=args.hidden, net_norm=args.net_norm, pooling=args.pooling,
                             dropout=args.dropout, nclass=args.nclass, nconvs=args.nconvs, args=args).to(device)
        model_real = GCN(nfeat=args.nfeat, nhid=args.hidden, net_norm=args.net_norm, pooling=args.pooling,
                         dropout=args.dropout, nclass=args.nclass, nconvs=args.nconvs, args=args).to(device)
        optimizer = torch.optim.Adam(model_syn.parameters(), lr=args.lr_model)
        loss_r = torch.zeros(1, len(feat_syn[0][0]), device=device)
        sampled = packed_data[4]
        for i in sampled:
            temp = torch.sum(i.x, dim=0).to(device)
            loss_r += temp
        loss_r /= len(sampled)
        for _ in range(args.inner_loop):
            if args.stru_discrete:
                adj_syn = get_discrete_graphs(adj_syn, False, args, inference=False)
            model_syn.load_state_dict(model_real.state_dict())
            loss = 0
            real_samples = dict()
            ind_samples = dict(); feat_samples = dict(); adj_samples = dict()
            for c in range(args.nclass):
                data_real = sample_graphs(packed_data, real_indices_class, nnodes_all, c, batch_size=args.bs_cond)
                ind = syn_class_indices[c]
                feat_syn_c = feat_syn[ind[0]:ind[1]]
                adj_syn_c = adj_syn[ind[0]: ind[1]]
                real_samples[str(c)] = data_real
                ind_samples[str(c)] = ind; feat_samples[str(c)] = feat_syn_c; adj_samples[str(c)] = adj_syn_c
                output_real = model_real.embed(data_real)
                output_real = torch.mean(output_real, 0)
                output_syn = model_syn.embed(feat_syn_c, adj_syn_c)
                output_syn = torch.mean(output_syn, 0)
                loss += match_loss(output_syn, output_real, args, device)
            loss_sparsity = F.relu(torch.sigmoid(adj_syn).mean() - sparsity)
            loss_c = torch.zeros(1, len(feat_syn[0][0]), device='cuda')
            for i in feat_syn:
                loss_c += torch.sum(i, dim=0)
            loss_c /= len(feat_syn)
            loss_feat = match_loss(loss_r, loss_c, args, 'cuda')
            loss_edge = torch.norm(adj_syn.mean())
            loss = loss + args.beta*loss_sparsity + args.gama*loss_feat + args.theta*loss_edge
            optimizer_adj.zero_grad()
            optimizer_feat.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_adj.step()
            optimizer_feat.step()
            adj_real_t = torch.rand(size=(n, args.nnodes_syn, args.nnodes_syn), dtype=torch.float, requires_grad=True, device='cuda')
            feat_real_t = torch.rand(size=(n, args.nnodes_syn, args.nfeat), dtype=torch.float, requires_grad=True, device='cuda')
            labels_real_t = torch.rand(size=(n,), dtype=torch.float, requires_grad=True, device='cuda')
            for c in range(args.nclass):
                ind = syn_class_indices[c]
                if args.big_graph or args.dataset == 'MUTAG':
                    feat_real, adj_real, labels_real = sample_graphs(packed_data, real_indices_class, nnodes_all, c, batch_size=ind[1]-ind[0], max_node_size=args.nnodes_syn, to_dense=True)
                else:
                    feat_real, adj_real, labels_real = sample_graphs(packed_data, real_indices_class, nnodes_all, c, batch_size=ind[1]-ind[0], max_node_size=None, to_dense=True)
                feat_real_t.data[ind[0]: ind[1]] = feat_real[:, :args.nnodes_syn].detach().data
                adj_real_t.data[ind[0]: ind[1]] = adj_real[:, :args.nnodes_syn, :args.nnodes_syn].detach().data
                labels_real_t.data[ind[0]: ind[1]] = labels_real.detach().data.squeeze()
            real_train_all = TensorDataset(feat_real_t, adj_real_t, labels_real_t)
            train_loader_real_all = DataLoader(real_train_all, batch_size=128, shuffle=True, num_workers=0)
            for _ in range(args.epoch_intrain):
                model_syn.train()
                for data_all in train_loader_real_all:
                    x_real, adj_real, y_real = data_all
                    x_real, adj_real, y_real = x_real.to(device), adj_real.to(device), y_real.to(device)
                    optimizer.zero_grad()
                    output_all = model_syn(x_real, adj_real)
                    y_all = y_real.type(torch.LongTensor).to(device)
                    loss_train = F.cross_entropy(output_all, y_all.view(-1))
                    loss_train.backward()
                    optimizer.step()
            model_real.load_state_dict(model_syn.state_dict())
    return model_real, model_syn, feat_syn, adj_syn, labels_syn

def last_training(packed_data, model):
    nnodes_all, real_indices_class = prepare_train_indices(packed_data, args)
    labels_syn = torch.LongTensor([[i]*args.ipc for i in range(args.nclass)]).to('cuda').view(-1)
    syn_class_indices = {i: [i*args.ipc, (i+1)*args.ipc] for i in range(args.nclass)}
    n = args.ipc * args.nclass
    adj_syn = torch.rand(size=(n, args.nnodes_syn, args.nnodes_syn), dtype=torch.float, requires_grad=True, device='cuda')
    feat_syn = torch.rand(size=(n, args.nnodes_syn, args.nfeat), dtype=torch.float, requires_grad=True, device='cuda')
    for c in range(args.nclass):
        ind = syn_class_indices[c]
        if args.dataset in ['BA', 'shape', 'ba_lrp', 'MUTAG']:
            feat_real, adj_real, labels_real = sample_graphs(packed_data, real_indices_class, nnodes_all, c, batch_size=ind[1]-ind[0], max_node_size=None, to_dense=True)
            while feat_syn.shape[1] > feat_real.shape[1]:
                random.seed(1)
                feat_real, adj_real, labels_real = sample_graphs(packed_data, real_indices_class, nnodes_all, c, batch_size=ind[1]-ind[0], max_node_size=None, to_dense=True)
        elif args.big_graph:
            feat_real, adj_real, labels_real = sample_graphs(packed_data, real_indices_class, nnodes_all, c, batch_size=ind[1]-ind[0], max_node_size=args.nnodes_syn, to_dense=True)
            while feat_syn.shape[1] > feat_real.shape[1]:
                random.seed(1)
                feat_real, adj_real, labels_real = sample_graphs(packed_data, real_indices_class, nnodes_all, c, batch_size=ind[1]-ind[0], max_node_size=args.nnodes_syn, to_dense=True)
        feat_syn.data[ind[0]: ind[1]] = feat_real[:, :args.nnodes_syn].detach().data
        adj_syn.data[ind[0]: ind[1]] = adj_real[:, :args.nnodes_syn, :args.nnodes_syn].detach().data
    model_syn = DenseGCN(nfeat=args.nfeat, nhid=args.hidden, net_norm=args.net_norm, pooling=args.pooling,
                         dropout=args.dropout, nclass=args.nclass, nconvs=args.nconvs, args=args).to('cuda')
    optimizer_adj = torch.optim.Adam([adj_syn], lr=args.lr_adj*0.1)
    optimizer_feat = torch.optim.Adam([feat_syn], lr=args.lr_feat)
    for _ in range(args.epochs_last):
        for c in range(args.nclass):
            data_real_c = sample_graphs(packed_data, real_indices_class, nnodes_all, c, batch_size=args.bs_cond)
            ind = syn_class_indices[c]
            feat_syn_c = feat_syn[ind[0]:ind[1]]; adj_syn_c = adj_syn[ind[0]: ind[1]]
            model_syn.load_state_dict(model.state_dict())
            embed_real = model.embed(data_real_c)
            embed_syn = model_syn.embed(feat_syn_c, adj_syn_c)
            embed_real_mean = torch.mean(embed_real, axis=0)
            embed_syn_mean = torch.mean(embed_syn, axis=0)
            loss_dm = torch.linalg.norm(embed_syn_mean-embed_real_mean, ord=2)
            output_syn = model_syn(feat_syn_c, adj_syn_c)
            y = labels_syn[ind[0]:ind[1]]
            loss_ce = F.cross_entropy(output_syn, y)
            loss = args.alpha_last*loss_dm + loss_ce
            optimizer_adj.zero_grad()
            optimizer_feat.zero_grad()
            loss.backward()
            optimizer_adj.step()
            optimizer_feat.step()
    return feat_syn, adj_syn, labels_syn

def test_acc_fidelity(feat_syn, adj_syn, labels_syn, packed_data, model, args):
    model.eval()
    if args.stru_discrete:
        adj_syn, adj_vis = get_discrete_graphs(adj_syn, True, args, inference=True)
    sampled = data_convert(feat_syn, adj_syn, labels_syn)
    dst_syn_train = SparseTensorDataset(sampled)
    train_loader = DataLoader(dst_syn_train, batch_size=128, shuffle=True, num_workers=0)
    syn_class_indices = {i: [i*args.ipc, (i+1)*args.ipc] for i in range(args.nclass)}
    prob_list = []
    for c in range(args.nclass):
        ind = syn_class_indices[c]
        feat_syn_c = feat_syn[ind[0]:ind[1]]; adj_syn_c = adj_syn[ind[0]: ind[1]]; labels_syn_c = labels_syn[ind[0]:ind[1]]
        sampled_c = data_convert(feat_syn_c, adj_syn_c, labels_syn_c)
        dst_syn_train_c = SparseTensorDataset(sampled_c)
        train_loader_c = DataLoader(dst_syn_train_c, batch_size=128, shuffle=True, num_workers=0)
        for dat in train_loader_c:
            prob = torch.exp(model(dat)).mean(0)[c].item()
            prob_list.append(prob*100)
    print(args.dataset+' Pre_Acc [Mean, std];', [np.mean(prob_list), np.std(prob_list)])
    model_syn_new = GCN(nfeat=args.nfeat, nconvs=args.nconvs, nhid=args.hidden, nclass=args.nclass, net_norm=args.net_norm, pooling=args.pooling, dropout=args.dropout, args=args).to('cuda')
    lr = 0.001
    optimizer = torch.optim.Adam(model_syn_new.parameters(), lr=lr*10)
    epochs = 500
    for it in range(epochs):
        for dat in train_loader:
            dat = dat.to('cuda')
            y = dat.y
            optimizer.zero_grad()
            output = model_syn_new(dat)
            loss = F.cross_entropy(output, y.view(-1))
            loss.backward()
            optimizer.step()
    acc = (model_syn_new(dat).max(-1)[1]==dat.y).float().mean().item()
    print('ACC G_syn[syn]:', acc)
    fid_train = []; acc_train = []
    for dat in packed_data[1]:
        dat = dat.to('cuda')
        y_pred_real = model(dat).max(-1)[1]
        y_pred_syn = model_syn_new(dat).max(-1)[1].to('cuda')
        fidelity_train = (y_pred_syn==y_pred_real).float().mean()
        acc = (model_syn_new(dat).max(-1)[1]==dat.y).float().mean().item()
        fid_train.append(fidelity_train.item()*100)
        acc_train.append(acc)
    print('ACC G_syn[train]:', np.mean(acc_train))
    print(args.dataset+' Model Fidelity[train]:', np.mean(fid_train))
    fid_test = []; acc_test = []
    for dat in packed_data[3]:
        dat = dat.to('cuda')
        y_pred_real = model(dat).max(-1)[1]
        y_pred_syn = model_syn_new(dat).max(-1)[1].to('cuda')
        fidelity_test = (y_pred_syn==y_pred_real).float().mean()
        acc = (model_syn_new(dat).max(-1)[1]==dat.y).float().mean().item()
        fid_test.append(fidelity_test.item()*100)
        acc_test.append(acc)
    print('ACC G_syn[test]:', np.mean(acc_test))
    print(args.dataset+' Model Fidelity[test]:', np.mean(fid_test))
    return np.mean(prob_list), np.mean(acc_train), np.mean(acc_test), np.mean(fid_train), np.mean(fid_test)

def random_train(packed_data, real_indices_class, nnodes_all, model, runs):
    device = 'cuda'
    n = args.ipc * args.nclass
    syn_class_indices = {i: [i*args.ipc, (i+1)*args.ipc] for i in range(args.nclass)}
    adj_syn = torch.rand(size=(n, args.nnodes_syn, args.nnodes_syn), dtype=torch.float, requires_grad=True, device='cuda')
    feat_syn = torch.rand(size=(n, args.nnodes_syn, args.nfeat), dtype=torch.float, requires_grad=True, device='cuda')
    labels_syn = torch.rand(size=(n,), dtype=torch.float, requires_grad=True, device='cuda')
    for c in range(args.nclass):
        ind = syn_class_indices[c]
        if args.big_graph or args.dataset == 'MUTAG':
            feat_real, adj_real, labels_real = sample_graphs(packed_data, real_indices_class, nnodes_all, c, batch_size=ind[1]-ind[0], max_node_size=args.nnodes_syn, to_dense=True)
        else:
            feat_real, adj_real, labels_real = sample_graphs(packed_data, real_indices_class, nnodes_all, c, batch_size=ind[1]-ind[0], max_node_size=None, to_dense=True)
        feat_syn.data[ind[0]: ind[1]] = feat_real[:, :args.nnodes_syn].detach().data
        adj_syn.data[ind[0]: ind[1]] = adj_real[:, :args.nnodes_syn, :args.nnodes_syn].detach().data
        labels_syn.data[ind[0]: ind[1]] = labels_real.detach().data.squeeze()
    sampled = data_convert(feat_syn, adj_syn, labels_syn)
    dst_syn_train = SparseTensorDataset(sampled)
    train_loader = DataLoader(dst_syn_train, batch_size=128, shuffle=True, num_workers=0)
    syn_class_indices = {i: [i*args.ipc, (i+1)*args.ipc] for i in range(args.nclass)}
    prob_list = []
    for c in range(args.nclass):
        ind = syn_class_indices[c]
        feat_syn_c = feat_syn[ind[0]:ind[1]]; adj_syn_c = adj_syn[ind[0]: ind[1]]; labels_syn_c = labels_syn[ind[0]:ind[1]]
        sampled_c = data_convert(feat_syn_c, adj_syn_c, labels_syn_c)
        dst_syn_train_c = SparseTensorDataset(sampled_c)
        train_loader_c = DataLoader(dst_syn_train_c, batch_size=128, shuffle=True, num_workers=0)
        for dat in train_loader_c:
            prob = torch.exp(model(dat)).mean(0)[c].item()
            prob_list.append(prob*100)
    real_train_all = TensorDataset(feat_syn, adj_syn, labels_syn)
    train_loader_real_all = DataLoader(real_train_all, batch_size=128, shuffle=True, num_workers=0)
    model_syn_r = DenseGCN(nfeat=args.nfeat, nhid=args.hidden, net_norm=args.net_norm, pooling=args.pooling, dropout=args.dropout, nclass=args.nclass, nconvs=args.nconvs, args=args).to('cuda')
    model_real_r = GCN(nfeat=args.nfeat, nhid=args.hidden, net_norm=args.net_norm, pooling=args.pooling, dropout=args.dropout, nclass=args.nclass, nconvs=args.nconvs, args=args).to('cuda')
    optimizer = torch.optim.Adam(model_syn_r.parameters(), lr=args.lr_model)
    for _ in range(50):
        model_syn_r.train()
        for data_all in train_loader_real_all:
            x_real, adj_real, y_real = data_all
            x_real, adj_real, y_real = x_real.to(device), adj_real.to(device), y_real.to(device)
            optimizer.zero_grad()
            output_all = model_syn_r(x_real, adj_real)
            y_all = y_real.type(torch.LongTensor).to(device)
            loss_train = F.cross_entropy(output_all, y_all.view(-1))
            loss_train.backward()
            optimizer.step()
    model_real_r.load_state_dict(model_syn_r.state_dict())
    fid_train = []; fid_test = []
    for dat in packed_data[1]:
        dat = dat.to('cuda')
        y_pred_real = model(dat).max(-1)[1]
        y_pred_syn = model_real_r(dat).max(-1)[1].to('cuda')
        fidelity_train = (y_pred_syn==y_pred_real).float().mean()
        fid_train.append(fidelity_train.item()*100)
    for dat in packed_data[3]:
        dat = dat.to('cuda')
        y_pred_real = model(dat).max(-1)[1]
        y_pred_syn = model_real_r(dat).max(-1)[1].to('cuda')
        fidelity_test = (y_pred_syn==y_pred_real).float().mean()
        fid_test.append(fidelity_test.item()*100)
    return prob_list, fid_train, fid_test

# =====================
# Main Execution Logic (example usage)
# =====================
if __name__ == '__main__':
    torch.manual_seed(0)
    # Train the original model
    model = GCN(nfeat=args.nfeat, nconvs=args.nconvs, nhid=args.hidden, nclass=args.nclass, 
                net_norm=args.net_norm, pooling=args.pooling, dropout=args.dropout, args=args).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_model*10)
    epochs = 200
    for it in range(epochs):
        model.train()
        for dat in packed_data[1]:
            dat = dat.to('cuda')
            y = dat.y.type(torch.LongTensor).to('cuda')
            optimizer.zero_grad()
            output = model(dat).to('cuda')
            loss = F.cross_entropy(output, y.view(-1))
            loss.backward()
            optimizer.step()
    # Example: GDM training
    nnodes_all, real_indices_class = prepare_train_indices(packed_data, args)
    model_real, model_syn, feat_syn, adj_syn, labels_syn = GDM_training(packed_data, nnodes_all, real_indices_class, args)
    # Example: Evaluation
    pred_acc, acc_train, acc_test, fid_train, fid_test = test_acc_fidelity(feat_syn, adj_syn, labels_syn, packed_data, model, args) 