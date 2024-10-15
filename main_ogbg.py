from dig.auggraph.method.SMixup import smixup_ogbg
import numpy as np
import torch
from torch.nn import functional as F

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")


GMNET_conf = {"nlayers": 5, "nhidden": 256, "bs": 512, "lr": 0.001, "epochs": 500}
SupCon_conf = {"bs": 512, "lr": 0.001, "epochs": 100}
model = smixup_ogbg.smixup(data_root_path="dataset/", dataset="molhiv", GMNET_conf=GMNET_conf, SupCon_conf=SupCon_conf)
model.train_test(batch_size=32, cls_model="GIN", cls_nlayers=5, cls_hidden=256, cls_dropout=0.5, cls_lr=0.001, cls_epochs=100, alpha1=0.2, alpha2=0.2, ckpt_path="ckpt_path/molhiv")