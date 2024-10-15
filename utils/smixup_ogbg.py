import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import numpy as np

from dig.auggraph.method.SMixup.model.GCN import GCN
from dig.auggraph.method.SMixup.model.GIN import GIN
from dig.auggraph.method.SMixup.model.GraphMatching import GraphMatching
from dig.auggraph.method.SMixup.utils.sinkhorn import Sinkhorn
from dig.auggraph.method.SMixup.utils.utils import NormalizedDegree, triplet_loss
from dig.auggraph.dataset.aug_dataset import TripleSet

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj, dense_to_sparse, degree
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from torch.nn.functional import softmax

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder

import copy
from torch_geometric.transforms import BaseTransform

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

    
class smixup():
    r"""
    The S-Mixup from the `"Graph Mixup with Soft Alignments" <https://icml.cc/virtual/2023/poster/24930>`_ paper.
    
    Args:
        data_root_path (string): Directory where datasets are saved. 
        dataset (string): Dataset Name.
        conf (dict): Hyperparameters of the graph matching network which is used to compute the soft alignments.
    """
    
    def __init__(self, data_root_path, dataset, GMNET_conf, SupCon_conf):
        self._get_dataset(data_root_path, dataset)
        self.GMNET_conf = GMNET_conf
        self.SupCon_conf = SupCon_conf
        self._get_GMNET(self.GMNET_conf['nlayers'], self.GMNET_conf['nhidden'], self.GMNET_conf['bs'], self.GMNET_conf['lr'], self.GMNET_conf['epochs'])

      
    def _get_GMNET(self, GMNET_nlayers, GMNET_hidden, GMNET_bs, GMNET_lr, GMNet_epochs):  
        conf = {}
        conf_dis_param = {}
        conf_dis_param['num_layers'] = GMNET_nlayers
        conf_dis_param['hidden'] = GMNET_hidden
        conf_dis_param['model_type'] = 'gmnet'
        conf_dis_param['pool_type'] = 'sum'
        conf_dis_param['fuse_type'] = 'abs_diff'
        conf['dis_param'] = conf_dis_param

        conf['batch_size'] = GMNET_bs
        conf['start_lr'] = GMNET_lr
        conf['factor'] = 0.5
        conf['patience'] = 5
        conf['min_lr'] = 0.0000001
        conf['pre_train_path'] = None
        conf['max_num_epochs'] = GMNet_epochs

        conf['dis_param']['in_dim'] = self.dataset[0].x.shape[1]
        
        self.GMNET = GraphMatching(**conf['dis_param'])
        
        
    def _get_dataset(self, data_root_path, dataset):
        if dataset == 'molhiv':
            dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=data_root_path)
            num_cls = 2
            
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
                
        self.dataset = dataset
        self.num_cls = num_cls
        
    def train_test(self, batch_size, cls_model, cls_nlayers, cls_hidden, cls_dropout, cls_lr, cls_epochs, alpha1, alpha2, ckpt_path, sim_method = 'cos'):
        r"""
        This method first train a GMNET and then use the GMNET to perform S-Mixup. 
        
        Args:
            batch_size (int): Batch size of training the classifier.
            cls_model (string): Use GCN or GIN as the backbone of the classifier. 
            cls_nlayers (int): Number of GNN layers of the classifier.
            cls_hidden (int): Number of hidden units of the classifier.
            cls_dropout (float): Dropout ratio of the classifier.
            cls_lr (float): Initial learning rate of training the classifier. 
            cls_epochs (int): Training epochs of the classifier.
            alpha (float): Mixup ratio.
            ckpt_path (string): Location for saving checkpoints. 
            sim_method (string): Similarity function used to compute the assignment matrix. (default: :obj:`cos`)
        """
        print("vanilla")
        # print(f"GMNET_conf: {self.GMNET_conf}")
        # print(f"SupCon_conf: {self.SupCon_conf}")
        print("batch_size: {}, cls_model: {}, cls_nlayers: {}, cls_hidden: {}, cls_dropout: {}, cls_lr: {}, cls_epochs: {}, alpha1: {}, alpha2: {}, ckpt_path: {}, sim_method: {}".format(batch_size, cls_model, cls_nlayers, cls_hidden, cls_dropout, cls_lr, cls_epochs, alpha1, alpha2, ckpt_path, sim_method))

        # criterion = CustomCrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()
        evaluator = Evaluator(name='ogbg-molhiv')
        self.atom_encoder = AtomEncoder(emb_dim=cls_hidden).to(device)
        test_accs = []
        
        split_idx = self.dataset.get_idx_split()
        # 打印数据集大小
        print(f"Total graphs: {len(self.dataset)}")
        print(f"Train graphs: {len(split_idx['train'])}")
        print(f"Validation graphs: {len(split_idx['valid'])}")
        print(f"Test graphs: {len(split_idx['test'])}")

        # 获取训练集
        train_idx = split_idx['train']
        train_set = self.dataset[train_idx.tolist()]

        # 获取验证集
        valid_idx = split_idx['valid']
        val_set = self.dataset[valid_idx.tolist()]

        # 获取测试集
        test_idx = split_idx['test']
        test_set = self.dataset[test_idx.tolist()]


        for i in range(10):

            # self._get_GMNET(self.GMNET_conf['nlayers'], self.GMNET_conf['nhidden'], self.GMNET_conf['bs'], self.GMNET_conf['lr'], self.GMNET_conf['epochs'])
            # self.train_GMNET(train_set)
            
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = 8)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers = 8)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers = 8)

            if (cls_model == 'GCN'):
                model = GCN(self.dataset[0].x.shape[1], self.num_cls, cls_nlayers, cls_hidden, cls_dropout)
            elif (cls_model == 'GIN'):
                model = GIN(cls_hidden, self.num_cls - 1, cls_nlayers, cls_hidden, cls_dropout)

            # self.train_SupCon(model = model, train_set = train_set)

            model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=cls_lr, weight_decay=1e-5)
            # scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

            best_acc = 0.0
            best_model = None         
            for epoch in range(1, cls_epochs + 1):

                train_loss = self.train(model, device, train_loader, optimizer, criterion)
                print("Epoch [{}] Train_loss {}".format(epoch, train_loss))
                
                acc_val = self.eval(model, device, val_loader, evaluator)[self.dataset.eval_metric]

                print("Epoch [{}] Test results:".format(epoch),
                        "auc_val: {:.4f}".format(acc_val),)

                if acc_val >= best_acc:
                    best_acc = acc_val
                    best_model = copy.deepcopy(model)
            
            acc_test = self.eval(best_model, device, test_loader, evaluator)[self.dataset.eval_metric]
            print("Split {}: auc_test: {:.4f}".format(i, acc_test),)

            test_accs.append(acc_test)
            
        print("Final result: auc_test: {:.4f}+-{:.4f}".format(np.mean(test_accs), np.std(test_accs)))


    def train(self, model, device, data_loader, optimizer, loss_fn, alpha=0.2):
        """
        optimizer是给定优化器（torch.optim）
        loss_fn是给定损失函数
        """
        loss = 0

        for step, batch in enumerate(data_loader):
            batch = batch.to(device)

            if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                pass
            else:               
                batch.x = self.atom_encoder(batch.x)
                batch = self.ifplus(batch=batch, alpha=alpha, model=model)

                model.train()
                optimizer.zero_grad()
                _, op=model(batch)
                
                #loss=loss_fn(train_op,train_labels)
                #RuntimeError: result type Float can't be cast to the desired output type Long
                #train_op的dtype是torch.float32
                #train_labels的dtype是torch.int64
                a=loss_fn(op, batch.y1.float())

                loss=batch.lam * loss_fn(op, batch.y1.float()) + (1-batch.lam) * loss_fn(op, batch.y2.float())

                loss.backward()
                optimizer.step()

        return loss.item()
    

    def eval(self, model, device, loader, evaluator):
        model.eval()
        y_true = []
        y_pred = []

        for step, batch in enumerate(loader):
            batch = batch.to(device)

            if batch.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    _, pred = model(batch)

                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim = 0).numpy()
        y_pred = torch.cat(y_pred, dim = 0).numpy()

        input_dict = {"y_true": y_true, "y_pred": y_pred}

        return evaluator.eval(input_dict)

        
    def train_SupCon(self, model, train_set):

        train_loader = DataLoader(train_set, batch_size=self.SupCon_conf["bs"], shuffle=True, num_workers = 8)

        criterion = SupConLoss()
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr = self.SupCon_conf['lr'], weight_decay=1e-5)
        
        for epoch in range(1, self.SupCon_conf['epochs'] + 1):
            print("====epoch {} ====".format(epoch))
            loss_accum = 0.0

            for step, batch in enumerate(train_loader):
                batch = batch.cuda()
                model.train()
                optimizer.zero_grad()
                output, _ = model(batch)
                output = F.normalize(output)
                loss = criterion(output.unsqueeze(1), batch.y.long())

                loss.backward()
                optimizer.step()

                loss_accum += loss.item()

            train_loss = loss_accum / (step + 1)
            print("Epoch [{}] Train_loss {}".format(epoch, train_loss))

        print("SupCon training done.")


    def train_GMNET(self, train_set):
        self.GMNET.to(device)
        
        train_set = TripleSet(train_set)
        train_loader = DataLoader(train_set, batch_size = self.GMNET_conf['bs'], shuffle = True, num_workers = 8)
        optimizer = optim.Adam(self.GMNET.parameters(), lr = self.GMNET_conf['lr'], weight_decay=1e-4)
        
        for epoch in range(1, self.GMNET_conf['epochs'] + 1):
            print("====epoch {} ====".format(epoch))
            self.GMNET.train()
            train_loss = 0.0
            for data_batch in train_loader:
                anchor_data, pos_data, neg_data = data_batch
                anchor_data, pos_data, neg_data = anchor_data.to(device), pos_data.to(device), neg_data.to(device)

                optimizer.zero_grad()
                
                x_1, y = self.GMNET(anchor_data, pos_data, pred_head = False)
                x_2, z = self.GMNET(anchor_data, neg_data, pred_head = False)
                
                loss = triplet_loss(x_1, y, x_2, z)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                train_loss += loss
                
            print("Epoch [{}] Train_loss {}".format(epoch, train_loss / len(train_loader)))
        
        print("GMNET training done.")
        self.GMNET.eval()
        
        
    def Mixup(self, batch, alpha1, alpha2, sim_method = 'cos', normalize_method = 'softmax', temperature = 1.0,):    
        
        if alpha1 + alpha2 > 0:
            lam = np.random.beta(alpha1, alpha2)
        else:
            lam = 0.5
            
        lam = max(lam, 1 - lam)
        
        batch = batch.to(device)
        
        batch1 = batch.clone()

        data_list = list(batch1.to_data_list())
        
        import random
        data_list2 = data_list.copy()
        random.shuffle(data_list2)
        
        batch2 =  Batch.from_data_list(data_list2).to(device)

        h1, h2 = self.GMNET.dis_encoder(batch1, batch2, node_emd = True)
        h1, h2 = h1.detach(), h2.detach()
        
        for i in range(len(data_list)):
            data_list[i].emb = h1[batch1._slice_dict['x'][i] : batch1._slice_dict['x'][i + 1],:]

            data_list2[i].emb = h2[batch2._slice_dict['x'][i] : batch2._slice_dict['x'][i + 1],:]

        batch_size = len(data_list)
        
        mixed_data_list = []

        for i in range(len(data_list)):
            # match = data_list[i].emb @ data_list2[i].emb.T
            if sim_method == 'cos':
                emb1 = data_list[i].emb / data_list[i].emb.norm(dim = 1)[:,None]
                emb2 = data_list2[i].emb / data_list2[i].emb.norm(dim = 1)[:,None]
                match = emb1 @ emb2.T / temperature 
            elif sim_method == 'abs_diff':
                match = -(data_list[i].emb.unsqueeze(1) - data_list2[i].emb.unsqueeze(0)).norm(dim = -1)

            if (normalize_method == 'softmax'):
                normalized_match = softmax(match.detach().clone(), dim = 0)
            elif(normalize_method == 'sinkhorn'):
                normalized_match = Sinkhorn(match.detach().clone())
            
            mixed_adj = lam * to_dense_adj(data_list[i].edge_index)[0].double()+ (1-lam) * normalized_match.double() @ to_dense_adj(data_list2[i].edge_index)[0].double() @ normalized_match.double().T

            mixed_adj[mixed_adj < 0.1] = 0
            
            mixed_x = lam * data_list[i].x + (1-lam) * normalized_match.float() @ data_list2[i].x

            edge_index, edge_weights = dense_to_sparse(mixed_adj)
            
            data = Data(x = mixed_x.float(), edge_index = edge_index, edge_weights = edge_weights, y1 = data_list[i].y, y2 = data_list2[i].y)

            mixed_data_list.append(data)
            
        b = Batch.from_data_list(mixed_data_list)
        b.lam = lam
        return b
    

    def MixupPlus(self, batch, model, alpha, sim_method = 'cos', normalize_method = 'softmax', temperature = 1.0,):    
        
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 0.5
            
        lam = max(lam, 1 - lam)
        
        batch = batch.cuda()
        
        batch1 = batch.clone()

        data_list = list(batch1.to_data_list())
        
        import random
        data_list2 = data_list.copy()
        random.shuffle(data_list2)
        
        batch2 =  Batch.from_data_list(data_list2).cuda()

        h1, h2 = self.GMNET.dis_encoder(batch1, batch2, node_emd = True)
        h1, h2 = h1.detach(), h2.detach()
        
        for i in range(len(data_list)):
            data_list[i].emb = h1[batch1._slice_dict['x'][i] : batch1._slice_dict['x'][i + 1],:]

            data_list2[i].emb = h2[batch2._slice_dict['x'][i] : batch2._slice_dict['x'][i + 1],:]

        batch_size = len(data_list)
        
        mixed_data_list = []

        for i in range(len(data_list)):
            # match = data_list[i].emb @ data_list2[i].emb.T
            if sim_method == 'cos':
                emb1 = data_list[i].emb / data_list[i].emb.norm(dim = 1)[:,None]
                emb2 = data_list2[i].emb / data_list2[i].emb.norm(dim = 1)[:,None]
                match = emb1 @ emb2.T / temperature 
            elif sim_method == 'abs_diff':
                match = -(data_list[i].emb.unsqueeze(1) - data_list2[i].emb.unsqueeze(0)).norm(dim = -1)

            if (normalize_method == 'softmax'):
                normalized_match = softmax(match.detach().clone(), dim = 0)
            elif(normalize_method == 'sinkhorn'):
                normalized_match = Sinkhorn(match.detach().clone())
            
            mixed_adj = lam * to_dense_adj(data_list[i].edge_index)[0].double()+ (1-lam) * normalized_match.double() @ to_dense_adj(data_list2[i].edge_index)[0].double() @ normalized_match.double().T

            mixed_adj[mixed_adj < 0.1] = 0
            
            mixed_x = lam * data_list[i].x + (1-lam) * normalized_match.float() @ data_list2[i].x

            edge_index, edge_weights = dense_to_sparse(mixed_adj)
            
            data = Data(x = mixed_x.float(), edge_index = edge_index, edge_weights = edge_weights, y1 = data_list[i].y, y2 = data_list2[i].y)

            mixed_data_list.append(data)
            
        b = Batch.from_data_list(mixed_data_list)
        b.lam = self.compute_dynamic_lambda(model, b, batch1, batch2, lam)

        return b
    
    
    def ifplus(self, batch, model, alpha):

        lam = np.random.beta(alpha, alpha)

        batch = batch.to(device)
        
        batch1 = batch.clone()

        data_list = list(batch1.to_data_list())
        
        import random
        data_list2 = data_list.copy()
        random.shuffle(data_list2)
        
        batch2 =  Batch.from_data_list(data_list2).to(device)

        mixed_data_list = []

        for i in range(len(data_list)):
            data1 = data_list[i]
            data2 = data_list2[i]

            max_num_nodes = max(data1.num_nodes, data2.num_nodes)
            
            if data1.num_nodes < max_num_nodes:
                data1 = self.pad_graph(data1, max_num_nodes - data1.num_nodes)
            if data2.num_nodes < max_num_nodes:
                data2 = self.pad_graph(data2, max_num_nodes - data2.num_nodes)
            
            adj1 = to_dense_adj(data1.edge_index, max_num_nodes=max_num_nodes)[0].double()
            adj2 = to_dense_adj(data2.edge_index, max_num_nodes=max_num_nodes)[0].double()
            
            mixed_adj = lam * adj1 + (1-lam) * adj2
            mixed_adj[mixed_adj < 0.1] = 0
            
            #mixed_x = lam * data_list[i].x + (1-lam) * normalized_match.float() @ data_list2[i].x

            mixed_x = lam * data1.x + (1-lam) * data2.x

            edge_index, edge_weights = dense_to_sparse(mixed_adj)

            mixed_data = Data(x=mixed_x.float(), edge_index=edge_index, edge_attr=edge_weights, y1=data1.y, y2=data2.y)

            mixed_data_list.append(mixed_data)


        b = Batch.from_data_list(mixed_data_list)
        b.lam = self.compute_dynamic_lambda(model, b, batch1, batch2, lam)
        
        return b
    
    def ifMixup(self, batch, alpha):

        lam = np.random.beta(alpha, alpha)

        batch = batch.to(device)
        
        batch1 = batch.clone()

        data_list = list(batch1.to_data_list())
        
        import random
        data_list2 = data_list.copy()
        random.shuffle(data_list2)
        
        batch2 =  Batch.from_data_list(data_list2).to(device)

        mixed_data_list = []

        for i in range(len(data_list)):
            data1 = data_list[i]
            data2 = data_list2[i]

            max_num_nodes = max(data1.num_nodes, data2.num_nodes)
            
            if data1.num_nodes < max_num_nodes:
                data1 = self.pad_graph(data1, max_num_nodes - data1.num_nodes)
            if data2.num_nodes < max_num_nodes:
                data2 = self.pad_graph(data2, max_num_nodes - data2.num_nodes)
            
            adj1 = to_dense_adj(data1.edge_index, max_num_nodes=max_num_nodes)[0].double()
            adj2 = to_dense_adj(data2.edge_index, max_num_nodes=max_num_nodes)[0].double()
            
            mixed_adj = lam * adj1 + (1-lam) * adj2
            mixed_adj[mixed_adj < 0.1] = 0
            
            #mixed_x = lam * data_list[i].x + (1-lam) * normalized_match.float() @ data_list2[i].x

            mixed_x = lam * data1.x + (1-lam) * data2.x

            edge_index, edge_weights = dense_to_sparse(mixed_adj)

            mixed_data = Data(x=mixed_x.float(), edge_index=edge_index, edge_attr=edge_weights, y1=data1.y, y2=data2.y)

            mixed_data_list.append(mixed_data)


        b = Batch.from_data_list(mixed_data_list)
        b.lam = lam
        
        return b
    
    def pad_graph(self, data, padding_size):
        num_features = data.x.shape[1]
        
        pad_x = torch.zeros((padding_size, num_features), device=data.x.device)
        padded_x = torch.cat([data.x, pad_x], dim=0)
        
        return Data(x=padded_x, edge_index=data.edge_index, y=data.y, num_nodes=data.num_nodes + padding_size)
    

    def compute_dynamic_lambda(self, model, mixed_data, data1, data2, lam):
        # 将模型设置为推理模式
        model.eval()

        with torch.no_grad():
            # 提取混合数据和原始数据的特征向量
            z_tilde, _ = model(mixed_data)
            z_a, _ = model(data1)
            z_b, _ = model(data2)

        # 计算特征向量之间的L2距离
        dist_a = torch.norm(softmax(z_tilde - z_a, dim=1), p=2, dim=1)
        dist_b = torch.norm(softmax(z_tilde - z_b, dim=1), p=2, dim=1)

        # 计算新的混合比例
        lambda_a = lam * torch.exp(-dist_a)
        lambda_b = (1 - lam) * torch.exp(-dist_b)
        
        new_lambda = lambda_a / (lambda_a + lambda_b)
        
        return new_lambda
    


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        # 如果 input 是 (N, C) 形状的，target 是 (N,) 形状的
        # input 是未经过 softmax 的 logits
        # target 是对应的类别索引
        
        # 计算 log softmax
        log_softmax = F.log_softmax(input, dim=1)
        
        # 根据 target 获取每个样本对应类别的 log softmax 值
        loss_per_sample = -log_softmax[range(input.shape[0]), target]
        
        return loss_per_sample
    
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss