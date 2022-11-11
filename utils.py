#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-06 18:50:26
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$

import os, torch, argparse, random
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
from torch import nn, optim
from tqdm import tqdm

def load_data(met_path, pro_path, task):
    met_df = pd.read_csv(met_path)
    pro_df = pd.read_csv(pro_path)

    ids, labels, boards, met, pro = extract_met_pro(met_df, pro_df)

    if task == "met":
        return ids, labels, boards, met
    elif task == "pro":
        return ids, labels, boards, pro
    elif task == "both":
        data = np.concatenate([met, pro], axis = -1)
        return ids, labels, boards, data

def extract_met_pro(met_df, pro_df):
    met_ids, met_boards, met_labels, met_data = get_data_and_info(met_df)
    pro_ids, pro_boards, pro_labels, pro_data = get_data_and_info(pro_df)

    intersections = sorted(list(set(met_ids).intersection(set(pro_ids))))
    
    new_labels, new_met, new_pro, new_boards =[], [], [], []
    for id_ in intersections:
        met_idx = met_ids.index(id_)
        pro_idx = pro_ids.index(id_)

        met_b = met_boards[met_idx]
        pro_b = pro_boards[pro_idx]
        met_l = met_labels[met_idx]
        pro_l = pro_labels[pro_idx]
        met_d = met_data[met_idx]
        pro_d = pro_data[pro_idx]

        assert met_b == pro_b
        assert met_l == pro_l

        new_labels.append(met_l)
        new_boards.append(met_b)
        new_met.append(met_d)
        new_pro.append(pro_d)
    return np.array(intersections), np.array(new_labels), np.array(new_boards), np.array(new_met), np.array(new_pro) 

def get_data_and_info(df):
    ids = df.pop("id").values
    boards = df.pop("board").values
    labels = df.pop("label").values
    data = df.values
    return list(ids), boards, labels, data

def train_test_split(ids, labels, boards, data, marker):
    mask =  boards == marker
    return ids[mask], labels[mask], boards[mask], data[mask]

def get_data_train_test_indices(X, Y, train_fold = 4, test_fold = 1):
    total_folds = train_fold + test_fold
    sfolder = StratifiedKFold(n_splits = total_folds, shuffle = False)
    for train, test in sfolder.split(X, Y):
        break
    return train, test 

def get_data_train_test(X, Y, ids, train_fold = 4, test_fold = 1):
    train_indices, test_indices = get_data_train_test_indices(X, Y, train_fold, test_fold)
    return X[train_indices], Y[train_indices], ids[train_indices], X[test_indices], Y[test_indices], ids[test_indices]

class Block(nn.Module):
    def __init__(self, in_dim, out_dim, drop):
        super(Block, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(),
            nn.Dropout(drop)
            )

    def forward(self, x):
        y = self.layer(x)
        return y

class MLP(nn.Module):
    """docstring for MLP"""
    def __init__(self, in_dim, out_dim, hid_dim, nlayer = 3, drop = 0.1):
        super(MLP, self).__init__()
        
        self.in_layer = Block(in_dim, hid_dim, drop)
        tmp_layers = [Block(hid_dim, hid_dim, drop) for _ in range(nlayer)]
        self.tmp_layers = nn.ModuleList(tmp_layers)
        self.out_layer = Block(hid_dim, out_dim, drop)

    def forward(self, x):
        x = self.in_layer(x)
        for l in self.tmp_layers:
            x = l(x) + x
        x = self.out_layer(x)
        return x

def tensor(arr, type = "float"):
    if type == "float":
        return torch.FloatTensor(arr).cuda()
    else:
        return torch.LongTensor(arr).cuda()

def cal_auc_tensor(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    auc = roc_auc_score(labels, preds)
    return auc

def write_preds(preds, labels, ids, model_dir, out_name):
    preds = torch.sigmoid(preds)
    preds = preds.detach().cpu().numpy().squeeze()
    labels = labels.cpu().numpy().squeeze()
    df = pd.DataFrame()
    df["pred"] = preds
    df["label"] = labels
    df["id"] = ids
    df.to_csv(os.path.join(model_dir, out_name), index = False)

    fig = plt.figure(figsize = (3, 3))
    fpr, tpr, ths = roc_curve(labels, preds)
    plt.plot(fpr, tpr)
    plt.savefig(os.path.join(model_dir, os.path.splitext(out_name)[0] + ".png"))

def save_model(model, model_dir, out_name):
    os.makedirs(model_dir, exist_ok = True)
    torch.save({"model": model.state_dict()}, os.path.join(model_dir, out_name))

def load_model(model, in_dir, in_name):
    model_sd = torch.load(os.path.join(in_dir, in_name))
    model.load_state_dict(model_sd["model"])
    return model

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

def model_params(model):
    num = np.sum([l.numel() for l in model.parameters()])
    return num/1e6

