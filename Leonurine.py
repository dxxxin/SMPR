import os
import numpy as np
import pandas as pd
import torch as th
from warnings import simplefilter
from model import Model
from sklearn.model_selection import KFold
from utils import get_metrics_auc, set_seed, plot_result_auc, \
    plot_result_aupr, EarlyStopping, get_metrics
from args import args
import torch
import pickle
from load_data import *

from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity, paired_distances


# 计算皮尔逊相关系数
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([5, 6, 7, 8, 7])
# correlation, p_value = pearsonr(x, y)
# print(f"皮尔逊相关系数: {correlation}, p-value: {p_value}")

def relign_disease(pred):
    fd = np.array(
        pd.read_csv('.\\dataset\\Kdataset\\omics\\disease.csv', index_col=False))
    data = pred[0].reshape(len(pred[0]), 1)
    res = np.concatenate((fd, data), axis=1)
    return res

def drugs_Similarity(model_file, model_path, smile):
    ECPFs = get_drug_ecpf(smile)
    #print(ECPFs)
    ECPFs = th.tensor([ECPFs]).float()
    model = torch.load(model_file)
    model.eval()
    R = model(ECPFs=ECPFs)
    R = R.detach().numpy()[0]

    fd = open(model_path+'R.pkl', 'rb')
    drugs = np.array(pickle.load(fd))

    fd = np.array(pd.read_csv('dataset/Kdataset/omics/drug.csv', index_col=False))[:, 1]
    result = []
    for i in range(len(drugs)):
        correlation, p_value = pearsonr(R, drugs[i])
        dis = np.linalg.norm(R - drugs[i])
        cosine = cosine_similarity(R.reshape(1, -1), drugs[i].reshape(1, -1))[0][0]
        result.append([fd[i], dis, cosine, correlation, p_value])
    #pd.DataFrame(result, columns=['id', 'distance', 'cos', 'correlation', 'p_value']).to_csv('Leonurine_Euclidean.csv',
    #                                                                                  index=False)
    return np.array(result),R

def get_pred_matrix(pred_path):
    fd = open(pred_path+'/pred.pkl', 'rb')
    matrix = np.array(pickle.load(fd))
    #print(matrix.shape)
    relation = pd.DataFrame(matrix)

    drugs = np.array(pd.read_csv('dataset/Kdataset/omics/drug.csv', index_col=False))[:, 1]
    diseases = np.array(pd.read_csv('dataset/Kdataset/omics/disease.csv', index_col=False))[:, 1]

    relation.columns = diseases
    relation.index = drugs
    #print(relation)

    return relation

def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def get_target_relation(smile, max_stack=100, dis='eud', pknow=''):
    relation = get_pred_matrix('save_model4000/')
    distance = drugs_Similarity('save_model4000/fold5_0.99_0.61.pth','save_model4000/', smile)
    #pknow = pd.read_csv('dataset/Leonurine/Prior_knowledge.csv',header=None, index_col=False).T
    col = np.array(pd.read_csv('dataset/Leonurine/disease.csv', index_col=False))[:,1]
    #pknow.index = col

    index= 1
    if dis =='eud':
        index = 1
    elif dis == 'cos':
        index = 2
    elif dis == 'pearson':
        index = 3
    distance = distance[np.argsort(distance[:, index])]
    top = distance[:max_stack, 0]

    top_maxtrix = relation.loc[top]
    d = np.reciprocal(distance[:max_stack, index].astype(float)).reshape(max_stack, 1)
    top_maxtrix = np.multiply(top_maxtrix, d)

    result = np.sum(top_maxtrix, axis=0)

    if pknow == '':
        for ind in result.index:
            result.loc[ind] = result.loc[ind]
    else:
        pknow.index = col
        for ind in result.index:
            result.loc[ind] = result.loc[ind] + float(pknow.loc[ind])

    result = min_max_normalize(result)
    result = result.T.sort_values(ascending=False)

    return result
    #pd.DataFrame(result, columns=['score']).to_csv('./save_data/related_diseases.csv')
    #pd.DataFrame(distance, columns=['id', 'distance', 'eud', 'correlation', 'p_value']).to_csv('./save_data/drugs_distance.csv', index=False)


#益母草碱
#smile = 'COC1=CC(=CC(=C1O)OC)C(=O)OCCCCN=C(N)N'
#result = get_target_relation(smile, dis='edu')
#pd.DataFrame(result, columns=['score']).to_csv('益母草碱_all_diseases.csv')
#print(result)
#pd.DataFrame(result, columns=['score']).to_csv('Leonurine_relation_disease2.csv')

#丹参
#smile = 'C1=CC(=C(C=C1C[C@H](C(=O)O)OC(=O)/C=C/C2=C3[C@@H]([C@H](OC3=C(C=C2)O)C4=CC(=C(C=C4)O)O)C(=O)O[C@H](CC5=CC(=C(C=C5)O)O)C(=O)O)O)O'
#print(smile)
#result = get_target_relation(smile, dis='edu')
#print(result)
#pd.DataFrame(result, columns=['score']).to_csv('丹參酚酸B_all_diseases.csv')









