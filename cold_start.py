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

def drugs_Similarity(model_file, model_path, name,  smile):
    ECPFs = get_drug_ecpf(smile)
    #print(ECPFs)
    ECPFs = th.tensor([ECPFs]).float()
    model = torch.load(model_file)
    model.eval()
    R = model(ECPFs=ECPFs)
    R = R.detach().numpy()[0]

    fd = open(model_path+name+'_R.pkl', 'rb')
    drugs = np.array(pickle.load(fd).detach().numpy())

    fd = np.array(pd.read_csv('dataset/cold_start/omics/drug.csv', index_col=False))[:, 1]
    result = []
    for i in range(len(drugs)):
        correlation, p_value = pearsonr(R, drugs[i])
        dis = np.linalg.norm(R - drugs[i])
        cosine = cosine_similarity(R.reshape(1, -1), drugs[i].reshape(1, -1))[0][0]
        result.append([fd[i], dis, cosine, correlation, p_value])
    #pd.DataFrame(result, columns=['id', 'distance', 'cos', 'correlation', 'p_value']).to_csv('Leonurine_Euclidean.csv',
    #                                                                                  index=False)
    return np.array(result),R

def get_pred_matrix(pred_path, name):
    fd = open(pred_path+'/'+name+'_pred.pkl', 'rb')
    matrix = np.array(pickle.load(fd))
    #print(matrix.shape)
    relation = pd.DataFrame(matrix)

    drugs = np.array(pd.read_csv('dataset/cold_start/omics/drug.csv', index_col=False))[:, 1]
    diseases = np.array(pd.read_csv('dataset/cold_start/omics/disease.csv', index_col=False))[:, 1]

    relation.columns = diseases
    relation.index = drugs
    #print(relation)

    return relation

def test_Bdataset(model_path,model_file, name, threshold=5, max_stack=None):
    #get label
    ind = np.array(pd.read_csv('dataset/cold_start/omics/cold_start_drug.csv', index_col=False))[:, 1]
    col = np.array(pd.read_csv('dataset/cold_start/omics/disease.csv', index_col=False))[:, 1]
    label = pd.read_csv('dataset/cold_start/cold_start_random_baseline.csv', header=None, index_col=False)
    label.columns = col
    label.index = ind

    #获得Kdataset和Bdastaset的disease交集
    kd = np.array(pd.read_csv('dataset/cold_start/omics/disease.csv', index_col=False))[:, 1]
    bd = np.array(pd.read_csv('dataset/cold_start/omics/disease.csv', index_col=False))[:, 1]
    b_r = np.array(pd.read_csv('dataset/cold_start/omics/cold_start_drug.csv', index_col=False))
    k_r = np.array(pd.read_csv('dataset/cold_start/omics/drug.csv', index_col=False))

    fd = open(model_path + '/'+name+'_pred.pkl', 'rb')
    matrix = np.array(pickle.load(fd))
    relation = pd.DataFrame(matrix)
    drugs = np.array(pd.read_csv('dataset/cold_start/omics/drug.csv', index_col=False))[:, 1]
    diseases = np.array(pd.read_csv('dataset/Kdataset/omics/disease.csv', index_col=False))[:, 1]
    relation.columns = diseases
    relation.index = drugs

    pred = []
    true = []

    diss = []
    b_embed = []
    for i in b_r:
        smile = i[2]
        distance, R = drugs_Similarity(model_file, model_path, name, smile)
        index = 1
        b_embed.append(list(R))
        diss.append(list(distance[:,index]))

        distance = distance[np.argsort(distance[:, index])]
        if max_stack == None:
            max_stack = len(distance)
        top = distance[:max_stack, 0]

        top_maxtrix = relation.loc[top]
        d = np.reciprocal(distance[:max_stack, index].astype(float)).reshape(max_stack, 1)
        #d = np.array(distance[:max_stack, index].astype(float)).reshape(max_stack, 1)
        #print(d)
        top_maxtrix = np.multiply(top_maxtrix, d)

        result = np.sum(top_maxtrix, axis=0)
        #result = min_max_normalize(result)
        #result = result.T.sort_values(ascending=False)

        #对齐disease
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
            #pred.append(1)
            true.append(label.loc[i[1], result.index[thre]])

    k_r = np.array(pd.read_csv('dataset/cold_start/omics/drug.csv', index_col=False))
    diss = pd.DataFrame(diss)
    diss.index = b_r[:,1]
    diss.columns = k_r[:,1]
    diss.to_csv('cold_start_similarity.csv')
    #print(pred)
    #with open('B_embed.pkl', 'wb') as file:
    #    pickle.dump(b_embed, file)
    return pred, true

result = []
#ther = [0, 0.1,0.2,0.25,0.3,0.35,0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
ther = [454]
files = ['fold1_0.97_0.5','fold2_0.97_0.45','fold3_0.97_0.44','fold4_0.97_0.46','fold5_0.97_0.45']
for f in files:
    #save_model4000/fold5_0.99_0.61.pth
    pred, true = test_Bdataset('save_cold_start/','save_cold_start/'+f+'.pth',f, 454, max_stack=100)
    #print(pred)
    #print(true)
    #accuracy = accuracy_score(true, pred)
    #print(f"Model Accuracy: {accuracy}")
    #pred, true = test_Bdataset('save_model_decode/', 'save_model_decode/fold7_0.99_0.54.pth', i, max_stack=100)
    AUC, aupr, acc, f1, pre, rec, spec, the = get_metrics(np.array(true),np.array(pred))
    print('AUC',AUC, 'AUPR',aupr,'acc', acc, 'f1',f1)
    print('pre',pre,'recall', rec,'spec', spec, 'the',the)
    result.append([f, AUC, aupr, acc, f1, pre, rec, spec, the])

    count = 0
    for i in pred:
        if i > the:
            count+=1
    print(count, len(pred))
pd.DataFrame(result, columns=['thre', 'AUC', 'aupr', 'acc', 'f1', 'pre', 'rec', 'spec', 'the']).to_csv('cold_start_result_topn.csv')







