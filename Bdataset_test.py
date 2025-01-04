import os
import numpy as np
import pandas as pd
import torch as th
from warnings import simplefilter
from model import Model
from sklearn.model_selection import KFold
from utils import get_metrics_auc, set_seed, plot_result_auc,\
    plot_result_aupr, EarlyStopping, get_metrics
from args import args
import torch
import pickle
from load_data import *
from Leonurine import *
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import mean_squared_error

def test_Bdataset(model_path,model_file, threshold=5, max_stack=None):
    #get label
    ind = np.array(pd.read_csv('dataset/Bdataset/Omics/new_drug2.csv', index_col=False))[:, 1]
    col = np.array(pd.read_csv('dataset/Bdataset/Omics/new_disease.csv', index_col=False))[:, 1]
    label = pd.read_csv('dataset/Bdataset/new_Bdataset_baseline2.csv', header=None, index_col=False)
    label.columns = col
    label.index = ind

    #获得Kdataset和Bdastaset的disease交集
    #kd = np.array(pd.read_csv('dataset/Kdataset/Omics/disease.csv', index_col=False))[:, 1]
    #bd = np.array(pd.read_csv('dataset/Bdataset/Omics/disease.csv', index_col=False))[:, 1]
    #intersect = np.intersect1d(kd,bd)
    intersect = label.columns

    relation = get_pred_matrix(model_path)

    b_r = np.array(pd.read_csv('dataset/Bdataset/omics/new_drug2.csv', index_col=False))
    pred = []
    true = []
    for i in b_r:
        smile = i[2]
        distance = drugs_Similarity(model_file, model_path, smile)

        index = 1
        distance = distance[np.argsort(distance[:, index])]
        if max_stack == None:
            max_stack = len(distance)
        top = distance[:max_stack, 0]

        top_maxtrix = relation.loc[top]
        d = np.reciprocal(distance[:max_stack, index].astype(float)).reshape(max_stack, 1)
        top_maxtrix = np.multiply(top_maxtrix, d)

        result = np.sum(top_maxtrix, axis=0)
        # 对齐disease
        result = result.loc[intersect]

        pknow = label.loc[i[1]]
        for ind in result.index:
            result.loc[ind] = result.loc[ind] + pknow.loc[ind]

        result = min_max_normalize(result)
        result = result.T.sort_values(ascending=False)

        #for thre in result.index:
        #    pred.append(result[thre])
        #    true.append(label.loc[i[1], thre])
        for thre in range(threshold):
            pred.append(result.iloc[thre])
            true.append(label.loc[i[1], result.index[thre]])

    return pred, true

result = []
#ther = [0, 0.1,0.2,0.25,0.3,0.35,0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
ther = [5,10,20,30,40,50,60,70,80,90,100]
#ther = [10]
for i in ther:
    #save_model4000/fold5_0.99_0.61.pth
    pred, true = test_Bdataset('save_model4000/','save_model4000/fold5_0.99_0.61.pth', i, max_stack=100)
    print(len(pred))
    print(len(true))
    #pred, true = test_Bdataset('save_model_decode/', 'save_model_decode/fold7_0.99_0.54.pth', i, max_stack=100)
    AUC, aupr, acc, f1, pre, rec, spec, the = get_metrics(np.array(true),np.array(pred))
    print('AUC',AUC, 'AUPR',aupr,'acc', acc, 'f1',f1)
    print('pre',pre,'recall', rec,'spec', spec, 'the',the)
    result.append([i, AUC, aupr, acc, f1, pre, rec, spec, the])
pd.DataFrame(result, columns=['thre', 'AUC', 'aupr', 'acc', 'f1', 'pre', 'rec', 'spec', 'the']).to_csv('Bdataset_result_topn.csv')






