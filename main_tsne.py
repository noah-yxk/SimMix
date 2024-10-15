from dig.auggraph.method.SMixup import mixup_tsne
import numpy as np
import torch
from torch.nn import functional as F

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")


# GMNET_conf = {"nlayers": 6, "nhidden": 256, "bs": 256, "lr": 0.001, "epochs": 500}
# SupCon_conf = {"bs": 1024, "lr": 0.01, "epochs": 300}
# model = mixup_tsne.smixup(data_root_path="dataset/", dataset="IMDBB", SupCon_conf=SupCon_conf, GMNET_conf=GMNET_conf)
# model.train_test(batch_size=256, cls_model="GIN", cls_nlayers=4, cls_hidden=32, cls_dropout=0.5, cls_lr=0.001, cls_epochs=300, alpha1=0.2, alpha2=0.2, ckpt_path="ckpt_path/IMDBB/GIN")


GMNET_conf = {"nlayers": 5, "nhidden": 256, "bs": 256, "lr": 0.001, "epochs": 500}
SupCon_conf = {"bs": 1024, "lr": 0.01, "epochs": 500}
model = mixup_tsne.smixup(data_root_path="dataset/", dataset="PROTEINS",  SupCon_conf=SupCon_conf, GMNET_conf=GMNET_conf)
model.train_test(batch_size=256, cls_model="GIN", cls_nlayers=4, cls_hidden=32, cls_dropout=0.1, cls_lr=0.001, cls_epochs=300, alpha1=0.2, alpha2=0.2, ckpt_path="ckpt_path/PROTEINS")


# GMNET_conf = {"nlayers": 5, "nhidden": 256, "bs": 256, "lr": 0.001, "epochs": 500}
# SupCon_conf = {"bs": 1024, "lr": 0.01, "epochs": 500}
# model = mixup_tsne.smixup(data_root_path="dataset/", dataset="NCI1", SupCon_conf=SupCon_conf)
# model.train_test(batch_size=256, cls_model="GIN", cls_nlayers=4, cls_hidden=32, cls_dropout=0.5, cls_lr=0.01, cls_epochs=500, alpha1=20, alpha2=1, ckpt_path="ckpt_path/NCI1/GIN")

# GMNET_conf = {"nlayers": 4, "nhidden": 256, "bs": 8, "lr": 0.001, "epochs": 500}
# SupCon_conf = {"bs": 64, "lr": 0.01, "epochs": 500}
# model = mixup_tsne.smixup(data_root_path="dataset/", dataset="REDDITB", SupCon_conf=SupCon_conf)
# model.train_test(batch_size=16, cls_model="GIN", cls_nlayers=4, cls_hidden=32, cls_dropout=0.5, cls_lr=0.01, cls_epochs=500, alpha1=0.2, alpha2=0.2, ckpt_path="ckpt_path/REDDITB/GCN")


# GMNET_conf = {"nlayers": 5, "nhidden": 256, "bs": 256, "lr": 0.001, "epochs": 500}
# SupCon_conf = {"bs": 1024, "lr": 0.01, "epochs": 500}
# model = mixup_tsne.smixup(data_root_path="dataset/", dataset="IMDBM", SupCon_conf=SupCon_conf)
# model.train_test(batch_size=256, cls_model="GIN", cls_nlayers=4, cls_hidden=32, cls_dropout=0.1, cls_lr=0.001, cls_epochs=500, alpha1=0.2, alpha2=0.2, ckpt_path="ckpt_path/IMDBM")


# GMNET_conf = {"nlayers": 4, "nhidden": 256, "bs": 8, "lr": 0.001, "epochs": 200}
# SupCon_conf = {"bs": 32, "lr": 0.01, "epochs": 500}
# model = mixup_tsne.smixup(data_root_path="dataset/", dataset="REDDITM5", SupCon_conf=SupCon_conf)
# model.train_test(batch_size=16, cls_model="GIN", cls_nlayers=4, cls_hidden=32, cls_dropout=0.5, cls_lr=0.01, cls_epochs=300, alpha1=0.2, alpha2=0.2, ckpt_path="ckpt_path/REDDITM5/GIN")