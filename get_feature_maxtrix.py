import os
import numpy as np
import pandas as pd
import torch as th
from warnings import simplefilter
from model import Model
from sklearn.model_selection import KFold
from load_data import load, remove_graph
from utils import get_metrics_auc, set_seed, plot_result_auc,\
    plot_result_aupr, EarlyStopping, get_metrics
from args import args

def get_matrix():
    g = load(args.dataset)
    feature = {'drug': g.nodes['drug'].data['h'],
               'disease': g.nodes['disease'].data['h'],
               'protein': g.nodes['protein'].data['h'],
               'gene': g.nodes['gene'].data['h'],
               'pathway': g.nodes['pathway'].data['h']}
    model = Model(etypes=g.etypes, ntypes=g.ntypes,
                  in_feats=feature['drug'].shape[1],
                  hidden_feats=args.hidden_feats,
                  num_heads=args.num_heads,
                  dropout=args.dropout)
    score, R, D = model(g, feature)
    pred = th.sigmoid(score)

    return pred


