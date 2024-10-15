from dig.auggraph.method import SMixup
import numpy as np
import torch
from torch.nn import functional as F

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")


GMNET_conf = {"nlayers": 6, "nhidden": 256, "bs": 256, "lr": 0.001, "epochs": 1}
model = SMixup.smixup(data_root_path="dataset/", dataset="IMDBB", GMNET_conf=GMNET_conf)
model.train_test(batch_size=256, cls_model="GIN", cls_nlayers=4, cls_hidden=32, cls_dropout=0.5, cls_lr=0.001, cls_epochs=300, alpha=0.2, ckpt_path="ckpt_path/IMDBB")


# GMNET_conf = {"nlayers": 5, "nhidden": 256, "bs": 256, "lr": 0.001, "epochs": 500}
# model = SMixup.smixup(data_root_path="dataset/", dataset="PROTEINS", GMNET_conf=GMNET_conf)
# model.train_test(batch_size=256, cls_model="GIN", cls_nlayers=4, cls_hidden=32, cls_dropout=0.5, cls_lr=0.001, cls_epochs=300, alpha=0.2, ckpt_path="ckpt_path/PROTEINS")


# GMNET_conf = {"nlayers": 5, "nhidden": 256, "bs": 256, "lr": 0.001, "epochs": 500}
# model = SMixup.smixup(data_root_path="dataset/", dataset="NCI1", GMNET_conf=GMNET_conf)
# model.train_test(batch_size=256, cls_model="GIN", cls_nlayers=4, cls_hidden=32, cls_dropout=0.5, cls_lr=0.01, cls_epochs=500, alpha=0.2, ckpt_path="ckpt_path/NCI1")


# GMNET_conf = {"nlayers": 4, "nhidden": 256, "bs": 8, "lr": 0.001, "epochs": 500}
# model = SMixup.smixup(data_root_path="dataset/", dataset="REDDITB", GMNET_conf=GMNET_conf)
# model.train_test(batch_size=16, cls_model="GIN", cls_nlayers=4, cls_hidden=32, cls_dropout=0, cls_lr=0.01, cls_epochs=500, alpha=0.2, ckpt_path="ckpt_path/REDDITB")


# GMNET_conf = {"nlayers": 5, "nhidden": 256, "bs": 256, "lr": 0.001, "epochs": 500}
# model = SMixup.smixup(data_root_path="dataset/", dataset="IMDBM", GMNET_conf=GMNET_conf)
# model.train_test(batch_size=256, cls_model="GCN", cls_nlayers=4, cls_hidden=32, cls_dropout=0.5, cls_lr=0.001, cls_epochs=300, alpha=0.2, ckpt_path="ckpt_path/IMDBM")


# GMNET_conf = {"nlayers": 4, "nhidden": 256, "bs": 8, "lr": 0.001, "epochs": 500}
# model = SMixup.smixup(data_root_path="dataset/", dataset="REDDITM5", GMNET_conf=GMNET_conf)
# model.train_test(batch_size=256, cls_model="GIN", cls_nlayers=4, cls_hidden=32, cls_dropout=0.5, cls_lr=0.01, cls_epochs=500, alpha=0.2, ckpt_path="ckpt_path/REDDITM5/GIN")

# GMNET_conf = {"nlayers": 5, "nhidden": 256, "bs": 512, "lr": 0.001, "epochs": 500}
# model = SMixup.smixup(data_root_path="dataset/", dataset="molhiv", GMNET_conf=GMNET_conf)
# model.train_test(batch_size=512, cls_model="GIN", cls_nlayers=5, cls_hidden=300, cls_dropout=0.5, cls_lr=0.001, cls_epochs=200, alpha=0.2, ckpt_path="ckpt_path/molhiv/GIN")