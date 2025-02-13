import os
import numpy as np
import pandas as pd
import torch
import torch as th
from warnings import simplefilter
from model import Model
from model import *
from sklearn.model_selection import KFold
from load_data import load, remove_graph
from utils import get_metrics_auc, set_seed, plot_result_auc, \
    plot_result_aupr, EarlyStopping, get_metrics
from args import args
import pickle
import random


def train():
    simplefilter(action='ignore', category=FutureWarning)
    print(args)
    set_seed(args.seed)
    try:
        os.mkdir(args.saved_path)
    except:
        pass

    if args.device_id:
        print('Training on GPU')
        device = th.device('cuda:{}'.format(args.device_id))
    else:
        print('Training on CPU')
        device = th.device('cpu')

    # load DDA data for Kfold splitting
    df = pd.read_csv('./dataset/{}/{}_baseline.csv'.format(args.dataset, args.dataset),
                     header=None).values
    data = np.array([[i, j, df[i, j]] for i in range(df.shape[0]) for j in range(df.shape[1])])
    data = data.astype('int64')
    data_pos = data[np.where(data[:, -1] == 1)[0]]
    data_neg = data[np.where(data[:, -1] == 0)[0]]
    assert len(data) == len(data_pos) + len(data_neg)

    set_seed(args.seed)
    kf = KFold(n_splits=args.nfold, shuffle=True,
               random_state=args.seed)
    fold = 1
    pred_result = np.zeros(df.shape)
    acc_result = []
    min_auc_aupr = 0

    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kf.split(data_pos),
                                                                            kf.split(data_neg)):

        print('{}-Cross Validation: Fold {}'.format(args.nfold, fold))
        # get the index list for train and test set
        train_pos_id, test_pos_id = data_pos[train_pos_idx], data_pos[test_pos_idx]
        train_neg_id, test_neg_id = data_neg[train_neg_idx], data_neg[test_neg_idx]
        train_pos_idx = [tuple(train_pos_id[:, 0]), tuple(train_pos_id[:, 1])]
        test_pos_idx = [tuple(test_pos_id[:, 0]), tuple(test_pos_id[:, 1])]
        train_neg_idx = [tuple(train_neg_id[:, 0]), tuple(train_neg_id[:, 1])]
        test_neg_idx = [tuple(test_neg_id[:, 0]), tuple(test_neg_id[:, 1])]
        assert len(test_pos_idx[0]) + len(test_neg_idx[0]) + len(train_pos_idx[0]) + len(train_neg_idx[0]) == len(data)
        # print('train_pos_idx',train_pos_idx)

        g, drug_structre, ECPFs = load(args.dataset)

        print('ECPFs.shape', ECPFs.shape)
        ECPFs = th.tensor(ECPFs).float()

        # remove test set DDA from train graph
        g = remove_graph(g, test_pos_id[:, :-1]).to(device)

        #删除20%药物-疾病连接关系 测试
        #删除20%
        #ransample = random.sample(range(len(train_pos_id)), int(len(train_pos_id)*0.2))
        #g = remove_graph(g, train_pos_id[ransample, :-1]).to(device)
        print(args.dataset)
        if args.dataset == 'Kdataset':
            feature = {'drug': g.nodes['drug'].data['h'],
                       'disease': g.nodes['disease'].data['h'],
                       'protein': g.nodes['protein'].data['h'],
                       'gene': g.nodes['gene'].data['h'],
                       'pathway': g.nodes['pathway'].data['h']}
        elif args.dataset == 'cold_start':
            feature = {'drug': g.nodes['drug'].data['h'],
                       'disease': g.nodes['disease'].data['h'],
                       'protein': g.nodes['protein'].data['h'],
                       'gene': g.nodes['gene'].data['h'],
                       'pathway': g.nodes['pathway'].data['h']}
        elif args.dataset == 'Fdataset':
            feature = {'drug': g.nodes['drug'].data['h'],
                       'disease': g.nodes['disease'].data['h'],
                       'protein': g.nodes['protein'].data['h']}

        # get the mask list for train and test set that used for performance calculation
        print('df', df.shape)
        mask_label = np.ones(df.shape)
        mask_label[test_pos_idx[0], test_pos_idx[1]] = 0
        mask_label[test_neg_idx[0], test_neg_idx[1]] = 0
        mask_test = np.where(mask_label == 0)
        mask_test = [tuple(mask_test[0]), tuple(mask_test[1])]
        mask_train = np.where(mask_label == 1)
        mask_train = [tuple(mask_train[0]), tuple(mask_train[1])]
        print('Number of total training samples: {}, pos samples: {}, neg samples: {}'.format(len(mask_train[0]),
                                                                                              len(train_pos_idx[0]),
                                                                                              len(train_neg_idx[0])))
        print('Number of total testing samples: {}, pos samples: {}, neg samples: {}'.format(len(mask_test[0]),
                                                                                             len(test_pos_idx[0]),
                                                                                             len(test_neg_idx[0])))
        assert len(mask_test[0]) == len(test_neg_idx[0]) + len(test_pos_idx[0])
        label = th.tensor(df).float().to(device)
        # print('g.etypes,',g.etypes)
        # print('g.ntypes,', g.ntypes)

        # load model and optimizer
        model = Model(etypes=g.etypes, ntypes=g.ntypes,
                      in_feats=feature['drug'].shape[1],
                      hidden_feats=args.hidden_feats,
                      num_heads=args.num_heads,
                      dropout=args.dropout)
        model.to(device)

        optimizer = th.optim.Adam(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
        optim_scheduler = th.optim.lr_scheduler.CyclicLR(optimizer,
                                                         base_lr=0.1 * args.learning_rate,
                                                         max_lr=args.learning_rate,
                                                         gamma=0.995,
                                                         step_size_up=20,
                                                         mode="exp_range",
                                                         cycle_momentum=False)
        criterion = th.nn.BCEWithLogitsLoss(pos_weight=th.tensor(len(train_neg_idx[0]) / len(train_pos_idx[0])))
        #loss_func = th.nn.MSELoss()
        #criterion = th.nn.BCEWithLogitsLoss()
        print('Loss pos weight: {:.3f}'.format(len(train_neg_idx[0]) / len(train_pos_idx[0])))
        stopper = EarlyStopping(patience=args.patience, saved_path=args.saved_path)

        # model training
        spl_r = 0.005
        for epoch in range(1, args.epoch + 1):
            model.train()
            score, D, R = model(g, feature, ECPFs)
            pred = th.sigmoid(score)
            # pred = score

            #pred_spl = score[mask_train].flatten()
            #true_spl = label[mask_train].flatten()

            '''temp_true = true_spl.detach().numpy()
            spl_index_pos = np.argwhere(temp_true==1)
            spl_index_neg = np.argwhere(temp_true==0)
            spl_index_neg = random.sample(list(spl_index_neg), int(len(spl_index_neg)*spl_r))
            true_spl = torch.cat((true_spl[spl_index_neg], true_spl[spl_index_pos]))
            pred_spl = torch.cat((pred_spl[spl_index_neg], pred_spl[spl_index_pos]))'''

            #loss = criterion(pred_spl, true_spl)
            loss = criterion(score[mask_train].flatten(), label[mask_train].flatten())
            #spl_r = spl_r + 0.1 * spl_r / loss
            #if spl_r > 1:
            #    spl_r = 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optim_scheduler.step()
            model.eval()

            AUC_, _ = get_metrics_auc(label[mask_train].cpu().detach().numpy(),
                                      pred[mask_train].cpu().detach().numpy())

            early_stop = stopper.step(loss.item(), AUC_, model)

            if epoch % 50 == 0:
                AUC, AUPR = get_metrics_auc(label[mask_test].cpu().detach().numpy(),
                                            pred[mask_test].cpu().detach().numpy())
                print('Epoch {} Loss: {:.3f}; Train AUC {:.3f}; AUC {:.3f}; AUPR: {:.3f}; spl_r: {:.3f}'.format(epoch, loss.item(),
                                                                                                 AUC_, AUC, AUPR, spl_r))

                print('-' * 50)
                if early_stop:
                    break

        stopper.load_checkpoint(model)
        model.eval()
        score, D, R = model(g, feature, ECPFs)
        pred = th.sigmoid(score).cpu().detach().numpy()
        # pred = score.cpu().detach().numpy()

        pred_result[test_pos_idx] = pred[test_pos_idx]
        pred_result[test_neg_idx] = pred[test_neg_idx]

        # save the result
        AUC, aupr, acc, f1, pre, rec, spec, _ = get_metrics(label.cpu().detach().numpy().flatten(),
                                                            pred_result.flatten())
        print(
            'Overall: AUC {:.3f}; AUPR: {:.3f}; Acc: {:.3f}; F1: {:.3f}; Precision {:.3f}; Recall {:.3f}; Spec {:.3f}'.
                format(AUC, aupr, acc, f1, pre, rec, spec))
        acc_result.append([AUC, aupr, acc, f1, pre, rec, spec])
        if AUC + aupr > min_auc_aupr:
            th.save(model,'./save_model_spl/best_model.pth')
            min_auc_aupr = AUC + aupr
        th.save(model,
                './save_model_spl/fold' + str(fold) + '_' + str(round(AUC, 2)) + '_' + str(round(aupr, 2)) + '.pth')
        with open('./save_cold_start/fold'+str(fold)+'_'+str(round(AUC, 2))+'_'+str(round(aupr, 2))+'.pkl', 'wb') as file:
            pickle.dump(D, file)
        with open('./save_cold_start/fold'+str(fold)+'_'+str(round(AUC, 2))+'_'+str(round(aupr, 2))+'_R.pkl', 'wb') as file:
            pickle.dump(R, file)
        with open('./save_cold_start/fold'+str(fold)+'_'+str(round(AUC, 2))+'_'+str(round(aupr, 2))+'_pred.pkl', 'wb') as file:
            pickle.dump(pred, file)
        fold += 1

    #pd.DataFrame(acc_result, columns=['AUC', 'aupr', 'acc', 'f1', 'pre', 'rec', 'spec']).to_csv(
    #    'Fdataset_acc_result_decode.csv', index=False)

    '''g, drug_structre, ECPFs = load(args.dataset)
    print('ECPFs.shape', ECPFs.shape)
    ECPFs = th.tensor(ECPFs).float()
    feature = {'drug': g.nodes['drug'].data['h'],
               'disease': g.nodes['disease'].data['h'],
               'protein': g.nodes['protein'].data['h'],
               'gene': g.nodes['gene'].data['h'],
               'pathway': g.nodes['pathway'].data['h']}'''

    #model = torch.load('save_model4000/fold5_0.99_0.61.pth')
    #model.eval()

    #score, D, R = model(g, feature, ECPFs)
    #pred = th.sigmoid(score).cpu().detach().numpy()
    #print('best_model:',min_auc_aupr)
    #with open('./save_model_decode2/D.pkl', 'wb') as file:
    #    pickle.dump(D.detach().numpy(), file)
    #with open('./save_model_decode2/R.pkl', 'wb') as file:
    #    pickle.dump(R.detach().numpy(), file)
    #with open('./save_model_decode2/pred.pkl', 'wb') as file:
    #     pickle.dump(pred, file)


    #pd.DataFrame(pred).to_csv('result_4000.csv', index=False, header=False)



if __name__ == '__main__':
    train()
