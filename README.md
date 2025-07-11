

# SMPR:  
SMPR:a structure-based multimodal prediction model for drug-disease repositioning<br>

## Model:  
![](img/SMPR.png)

## Introduction: 
    SMPR is a drug disease repositioning model.
    The model has made improvements to address the following issues:
    1.Enhance the importance of drug structure.
    2.Drug cold start. Effective prediction can also be made for drugs that are not in the dataset.
    3.Facing pharmacological workers. Provide executable software for convenient use.

SMPR provides two datasets, Dataset A and Dataset C. The cold start training dataset is modified based on Dataset A.<br>
Dataset A contains 894 drugs and 454 diseases.<br>
Dataset C contains 579 drugs and 274 diseases.<br>
## The Dataset A:<br>
![](img/Dataset_A.png)
## The Dataset C:<br>
![](img/Dataset_C.png)

## Requirements :  
    ptython=3.10
    torch=2.4.1+cuda121
    DGL >= 0.5.2

## Use the model:  
    The model reposition results are saved in result_4000.csv  

    For model training, please run Python main.py directly.  

    model_300dim.pkl is the mol2vec saved model.  

    The trained model and results are saved in save_model4000/:  
    D.pkl is the disease embedding feature.  
    R.dkl is the drug embedding feature.  
    fold5_0.99_0.61.pth is the saved model.  

    The training main parameters that can be called include:   
    1. --dataset, defult Kdataset, Fdataset (Dataset F), Kdataset (Dataset A),cold_start  
    2. --nfold, defult 5  
    3. --learning_rate, default 0.005  
    4. --hidden_feats, default 64  

    For the cold start:  
    For a cold start, recommand Leonurine.py. You can add a prior knowledge for new drug, the relationship between drug and diseases of Kdataset, can recommand dataset/Leonurine/Prior_knowledge.csv  

## Files :  
    An executable program can be downloaded and used directly on a Windows system.  
    The exe file is available at https://drive.google.com/file/d/1Z-9kS8z5skg0C1SyKYjGWP_37AtIQgnc/view?usp=drive_link

## Cite:
    Dong, Xin, et al. "SMPR: A structure-enhanced multimodal drug-disease prediction model for drug repositioning and cold start." arXiv preprint arXiv:2503.13322 (2025).
