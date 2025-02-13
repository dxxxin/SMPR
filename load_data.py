import dgl
import torch
import torch as th
import numpy as np
import pandas as pd
from gensim.models import word2vec
from rdkit.Chem import rdDistGeom as molDG
from rdkit.Chem import AllChem
import rdkit
from rdkit import Chem

smile2ecfp = word2vec.Word2Vec.load('./model_300dim.pkl')

def get_ECFP(mol, radio):
    ECFPs = mol2alt_sentence(mol, radio)
    if len(ECFPs) % (radio + 1) != 0:
        ECFPs = ECFPs[:-(len(ECFPs) % (radio + 1))]
    ECFP_by_radio = list((np.array(ECFPs).reshape((int(len(ECFPs) / (radio + 1)), (radio + 1))))[:, radio])
    return ECFP_by_radio


def mol2alt_sentence(mol, radius):
    #https://github.com/samoturk/mol2vec

    radii = list(range(int(radius) + 1))
    info = {}

    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)  # info: dictionary identifier, atom_idx, radius
    #_ = AllChem.MorganGenerator(mol, radius, bitInfo=info)  # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}

    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        for r in radii:  # iterate over radii
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)

def load(dataset):
    """Load the heterogeneous network of Bdataset or Kdataset.
       Note: It is also applicable to load other heterogeneous networks of your own datasets.
    """

    if dataset == 'Fdataset':
        return load_Fdataset()
    if dataset == 'Kdataset':
        return load_Kdataset()
    if dataset == 'cold_start':
        return load_cold_start()

def read_drug_smile(dataset):
    if dataset == 'Kdataset':
        fd = pd.read_csv('dataset/Kdataset/omics/drug.csv', index_col=False)
    if dataset == 'cold_start':
        fd = pd.read_csv('dataset/cold_start/omics/drug.csv', index_col=False)
    fd = np.array(fd)

    drug_smile = {}
    ECPFs = np.zeros((len(fd),300),dtype =float)
    num = 0
    for i in fd:
        temp_ecpf = np.zeros(300, dtype=float)
        temp_num = 0
        ecpfs = get_ECFP(Chem.MolFromSmiles(i[2]), 1)

        for ecpf in ecpfs:
            try:
                out = smile2ecfp.wv.word_vec(ecpf)
            except Exception as err:
                #print(err, temp_ecpf, temp_num)
                continue

            temp_ecpf += out
            temp_num+=1
        if temp_num == 0:
            ECPFs[num] = temp_ecpf
        else:
            ECPFs[num] = temp_ecpf/temp_num
        drug_smile[i[0]]={'name':i[1],'smile':i[2], 'ECFP':ecpf}
        num+=1
    return drug_smile,ECPFs

def read_Fdataset_drug_smile():
    fd = pd.read_csv('dataset/FDataset/drug.csv',header=None, index_col=False)
    fd = np.array(fd)

    drug_smile = {}
    ECPFs = np.zeros((len(fd),300),dtype =float)
    num = 0
    for i in fd:
        temp_ecpf = np.zeros(300, dtype=float)
        temp_num = 0
        ecpfs = get_ECFP(Chem.MolFromSmiles(i[2]), 1)

        for ecpf in ecpfs:
            try:
                out = smile2ecfp.wv.word_vec(ecpf)
            except Exception as err:
                #print(err, temp_ecpf, temp_num)
                continue

            temp_ecpf += out
            temp_num+=1
        if temp_num == 0:
            ECPFs[num] = temp_ecpf
        else:
            ECPFs[num] = temp_ecpf/temp_num
        drug_smile[i[0]]={'name':i[1],'smile':i[2], 'ECFP':ecpf}
        num+=1
    return drug_smile,ECPFs

def get_drug_ecpf(smile):
    ECPFs = np.zeros(300, dtype=float)
    ecpfs = get_ECFP(Chem.MolFromSmiles(smile), 1)
    num = 0
    for ecpf in ecpfs:
        try:
            out = smile2ecfp.wv.word_vec(ecpf)
            num+=1
        except Exception as err:
            # print(err, temp_ecpf, temp_num)
            continue
        ECPFs += out
    #print(ECPFs)
    if num ==0:
        return ECPFs
    else:
        ECPFs = ECPFs/num
        return ECPFs

def load_Kdataset():
    """Load the heterogeneous network of Kdataset.
    """

    drug_drug = pd.read_csv('./dataset/Kdataset/drug_drug_baseline.csv', header=None).values
    drug_sim = drug_drug
    for i in range(len(drug_drug)):
        sorted_idx = np.argpartition(drug_drug[i], 15)
        drug_drug[i, sorted_idx[-15:]] = 1
    drug_drug = pd.DataFrame(np.array(np.where(drug_drug == 1)).T, columns=['Drug1', 'Drug2'])
    protein_protein = pd.read_csv('./dataset/Kdataset/interactions/protein_protein.csv')
    gene_gene = pd.read_csv('./dataset/Kdataset/interactions/gene_gene.csv')
    pathway_pathway = pd.read_csv('./dataset/Kdataset/interactions/pathway_pathway.csv')
    disease_disease = pd.read_csv('./dataset/Kdataset/disease_disease_baseline.csv', header=None).values
    disease_sim = disease_disease
    for i in range(len(disease_disease)):
        sorted_idx = np.argpartition(disease_disease[i], 15)
        disease_disease[i, sorted_idx[-15:]] = 1
    disease_disease = pd.DataFrame(np.array(np.where(disease_disease == 1)).T, columns=['Disease1', 'Disease2'])
    drug_protein = pd.read_csv('./dataset/Kdataset/associations/drug_protein.csv')
    protein_gene = pd.read_csv('./dataset/Kdataset/associations/protein_gene.csv')
    gene_pathway = pd.read_csv('./dataset/Kdataset/associations/gene_pathway.csv')
    pathway_disease = pd.read_csv('./dataset/Kdataset/associations/pathway_disease.csv')
    drug_disease = pd.read_csv('./dataset/Kdataset/associations/Kdataset.csv')
    graph_data = {
        ('drug', 'drug_drug', 'drug'): (th.tensor(drug_drug['Drug1'].values),
                                        th.tensor(drug_drug['Drug2'].values)),
        ('drug', 'drug_protein', 'protein'): (th.tensor(drug_protein['Drug'].values),
                                              th.tensor(drug_protein['Protein'].values)),
        ('protein', 'protein_drug', 'drug'): (th.tensor(drug_protein['Protein'].values),
                                              th.tensor(drug_protein['Drug'].values)),
        ('protein', 'protein_protein', 'protein'): (th.tensor(protein_protein['Protein1'].values),
                                                    th.tensor(protein_protein['Protein2'].values)),
        ('protein', 'protein_gene', 'gene'): (th.tensor(protein_gene['Protein'].values),
                                              th.tensor(protein_gene['Gene'].values)),
        ('gene', 'gene_protein', 'protein'): (th.tensor(protein_gene['Gene'].values),
                                              th.tensor(protein_gene['Protein'].values)),
        ('gene', 'gene_gene', 'gene'): (th.tensor(gene_gene['Gene1'].values),
                                        th.tensor(gene_gene['Gene2'].values)),
        ('gene', 'gene_pathway', 'pathway'): (th.tensor(gene_pathway['Gene'].values),
                                              th.tensor(gene_pathway['Pathway'].values)),
        ('pathway', 'pathway_gene', 'gene'): (th.tensor(gene_pathway['Pathway'].values),
                                              th.tensor(gene_pathway['Gene'].values)),
        ('pathway', 'pathway_pathway', 'pathway'): (th.tensor(pathway_pathway['Pathway1'].values),
                                                    th.tensor(pathway_pathway['Pathway2'].values)),
        ('pathway', 'pathway_disease', 'disease'): (th.tensor(pathway_disease['Pathway'].values),
                                                    th.tensor(pathway_disease['Disease'].values)),
        ('disease', 'disease_pathway', 'pathway'): (th.tensor(pathway_disease['Disease'].values),
                                                    th.tensor(pathway_disease['Pathway'].values)),
        ('disease', 'disease_disease', 'disease'): (th.tensor(disease_disease['Disease1'].values),
                                                    th.tensor(disease_disease['Disease2'].values)),
        ('drug', 'drug_disease', 'disease'): (th.tensor(drug_disease['Drug'].values),
                                              th.tensor(drug_disease['Disease'].values)),
        ('disease', 'disease_drug', 'drug'): (th.tensor(drug_disease['Disease'].values),
                                              th.tensor(drug_disease['Drug'].values)),
    }
    g = dgl.heterograph(graph_data)
    drug_feature = np.hstack((drug_sim, np.zeros((g.num_nodes('drug'), g.num_nodes('disease')))))
    dis_feature = np.hstack((np.zeros((g.num_nodes('disease'), g.num_nodes('drug'))), disease_sim))
    #print('drug_sim', drug_sim.shape)
    #print('disease_sim', disease_sim.shape)
    #print(th.from_numpy(drug_feature).shape)
    #print(th.from_numpy(dis_feature).shape)
    g.nodes['drug'].data['h'] = th.from_numpy(drug_feature).to(th.float32)
    g.nodes['disease'].data['h'] = th.from_numpy(dis_feature).to(th.float32)
    g.nodes['protein'].data['h'] = th.zeros((g.num_nodes('protein'), drug_feature.shape[1])).to(th.float32)
    g.nodes['gene'].data['h'] = th.zeros((g.num_nodes('gene'), drug_feature.shape[1])).to(th.float32)
    g.nodes['pathway'].data['h'] = th.zeros((g.num_nodes('pathway'), drug_feature.shape[1])).to(th.float32)

    smiles, ECPFs = read_drug_smile('Kdataset')
    return g, smiles, ECPFs

def load_cold_start():
    """Load the heterogeneous network of cold_start.
    """

    drug_drug = pd.read_csv('./dataset/cold_start/drug_drug_baseline.csv', header=None).values
    drug_sim = drug_drug
    for i in range(len(drug_drug)):
        sorted_idx = np.argpartition(drug_drug[i], 15)
        drug_drug[i, sorted_idx[-15:]] = 1
    drug_drug = pd.DataFrame(np.array(np.where(drug_drug == 1)).T, columns=['Drug1', 'Drug2'])
    protein_protein = pd.read_csv('./dataset/cold_start/interactions/protein_protein.csv')
    gene_gene = pd.read_csv('./dataset/cold_start/interactions/gene_gene.csv')
    pathway_pathway = pd.read_csv('./dataset/cold_start/interactions/pathway_pathway.csv')
    disease_disease = pd.read_csv('./dataset/cold_start/disease_disease_baseline.csv', header=None).values
    disease_sim = disease_disease
    for i in range(len(disease_disease)):
        sorted_idx = np.argpartition(disease_disease[i], 15)
        disease_disease[i, sorted_idx[-15:]] = 1
    disease_disease = pd.DataFrame(np.array(np.where(disease_disease == 1)).T, columns=['Disease1', 'Disease2'])
    drug_protein = pd.read_csv('./dataset/cold_start/associations/drug_protein.csv')
    protein_gene = pd.read_csv('./dataset/cold_start/associations/protein_gene.csv')
    gene_pathway = pd.read_csv('./dataset/cold_start/associations/gene_pathway.csv')
    pathway_disease = pd.read_csv('./dataset/cold_start/associations/pathway_disease.csv')
    drug_disease = pd.read_csv('./dataset/cold_start/associations/Kdataset.csv')
    graph_data = {
        ('drug', 'drug_drug', 'drug'): (th.tensor(drug_drug['Drug1'].values),
                                        th.tensor(drug_drug['Drug2'].values)),
        ('drug', 'drug_protein', 'protein'): (th.tensor(drug_protein['Drug'].values),
                                              th.tensor(drug_protein['Protein'].values)),
        ('protein', 'protein_drug', 'drug'): (th.tensor(drug_protein['Protein'].values),
                                              th.tensor(drug_protein['Drug'].values)),
        ('protein', 'protein_protein', 'protein'): (th.tensor(protein_protein['Protein1'].values),
                                                    th.tensor(protein_protein['Protein2'].values)),
        ('protein', 'protein_gene', 'gene'): (th.tensor(protein_gene['Protein'].values),
                                              th.tensor(protein_gene['Gene'].values)),
        ('gene', 'gene_protein', 'protein'): (th.tensor(protein_gene['Gene'].values),
                                              th.tensor(protein_gene['Protein'].values)),
        ('gene', 'gene_gene', 'gene'): (th.tensor(gene_gene['Gene1'].values),
                                        th.tensor(gene_gene['Gene2'].values)),
        ('gene', 'gene_pathway', 'pathway'): (th.tensor(gene_pathway['Gene'].values),
                                              th.tensor(gene_pathway['Pathway'].values)),
        ('pathway', 'pathway_gene', 'gene'): (th.tensor(gene_pathway['Pathway'].values),
                                              th.tensor(gene_pathway['Gene'].values)),
        ('pathway', 'pathway_pathway', 'pathway'): (th.tensor(pathway_pathway['Pathway1'].values),
                                                    th.tensor(pathway_pathway['Pathway2'].values)),
        ('pathway', 'pathway_disease', 'disease'): (th.tensor(pathway_disease['Pathway'].values),
                                                    th.tensor(pathway_disease['Disease'].values)),
        ('disease', 'disease_pathway', 'pathway'): (th.tensor(pathway_disease['Disease'].values),
                                                    th.tensor(pathway_disease['Pathway'].values)),
        ('disease', 'disease_disease', 'disease'): (th.tensor(disease_disease['Disease1'].values),
                                                    th.tensor(disease_disease['Disease2'].values)),
        ('drug', 'drug_disease', 'disease'): (th.tensor(drug_disease['Drug'].values),
                                              th.tensor(drug_disease['Disease'].values)),
        ('disease', 'disease_drug', 'drug'): (th.tensor(drug_disease['Disease'].values),
                                              th.tensor(drug_disease['Drug'].values)),
    }
    g = dgl.heterograph(graph_data)
    print(drug_sim.shape)
    print('g.num_nodes(drug)',g.num_nodes('drug'))
    drug_feature = np.hstack((drug_sim, np.zeros((g.num_nodes('drug'), g.num_nodes('disease')))))
    dis_feature = np.hstack((np.zeros((g.num_nodes('disease'), g.num_nodes('drug'))), disease_sim))
    #print('drug_sim', drug_sim.shape)
    #print('disease_sim', disease_sim.shape)
    #print(th.from_numpy(drug_feature).shape)
    #print(th.from_numpy(dis_feature).shape)
    g.nodes['drug'].data['h'] = th.from_numpy(drug_feature).to(th.float32)
    g.nodes['disease'].data['h'] = th.from_numpy(dis_feature).to(th.float32)
    g.nodes['protein'].data['h'] = th.zeros((g.num_nodes('protein'), drug_feature.shape[1])).to(th.float32)
    g.nodes['gene'].data['h'] = th.zeros((g.num_nodes('gene'), drug_feature.shape[1])).to(th.float32)
    g.nodes['pathway'].data['h'] = th.zeros((g.num_nodes('pathway'), drug_feature.shape[1])).to(th.float32)
    smiles, ECPFs = read_drug_smile('cold_start')
    return g, smiles, ECPFs

def load_Fdataset():
    """Load the heterogeneous network of Bdataset.
    """
    drug_drug = pd.read_csv('./dataset/Fdataset/drug_drug_baseline.csv', header=None).values
    drug_sim = drug_drug
    for i in range(len(drug_drug)):
        sorted_idx = np.argpartition(drug_drug[i], 15)
        drug_drug[i, sorted_idx[-15:]] = 1
    drug_drug = pd.DataFrame(np.array(np.where(drug_drug == 1)).T, columns=['Drug1', 'Drug2'])
    #print('drug_drug',drug_drug)

    protein_protein = pd.read_csv('./dataset/Fdataset/protein_protein.csv')
    disease_disease = pd.read_csv('./dataset/Fdataset/disease_disease_baseline.csv', header=None).values
    disease_sim = disease_disease
    for i in range(len(disease_disease)):
        sorted_idx = np.argpartition(disease_disease[i], 15)
        disease_disease[i, sorted_idx[-15:]] = 1
    disease_disease = pd.DataFrame(np.array(np.where(disease_disease == 1)).T, columns=['Disease1', 'Disease2'])
    drug_protein = pd.read_csv('./dataset/Fdataset/drug_protein.csv')

    disease_protein = pd.read_csv('./dataset/Fdataset/disease_protein.csv')

    drug_disease = pd.read_csv('./dataset/Fdataset/baseline.csv')
    graph_data = {
        ('drug', 'drug_drug', 'drug'): (th.tensor(drug_drug['Drug1'].values),
                                        th.tensor(drug_drug['Drug2'].values)),
        ('drug', 'drug_protein', 'protein'): (th.tensor(drug_protein['Drug'].values),
                                              th.tensor(drug_protein['Protein'].values)),

        ('disease', 'disease_protein', 'protein'): (th.tensor(disease_protein['Disease'].values),
                                              th.tensor(disease_protein['Protein'].values)),

        ('protein', 'protein_drug', 'drug'): (th.tensor(drug_protein['Protein'].values),
                                              th.tensor(drug_protein['Drug'].values)),
        ('protein', 'protein_protein', 'protein'): (th.tensor(protein_protein['Protein1'].values),
                                                    th.tensor(protein_protein['Protein2'].values)),
        ('disease', 'disease_disease', 'disease'): (th.tensor(disease_disease['Disease1'].values),
                                                    th.tensor(disease_disease['Disease2'].values)),
        ('drug', 'drug_disease', 'disease'): (th.tensor(drug_disease['Drug'].values),
                                              th.tensor(drug_disease['Disease'].values)),
        ('disease', 'disease_drug', 'drug'): (th.tensor(drug_disease['Disease'].values),
                                              th.tensor(drug_disease['Drug'].values)),
    }
    g = dgl.heterograph(graph_data)

    drug_feature = np.hstack((drug_sim, np.zeros((g.num_nodes('drug'), g.num_nodes('disease')))))
    dis_feature = np.hstack((np.zeros((g.num_nodes('disease'), g.num_nodes('drug'))), disease_sim))
    g.nodes['drug'].data['h'] = th.from_numpy(drug_feature).to(th.float32)
    g.nodes['disease'].data['h'] = th.from_numpy(dis_feature).to(th.float32)
    g.nodes['protein'].data['h'] = th.zeros((g.num_nodes('protein'), drug_feature.shape[1])).to(th.float32)

    smiles, ECPFs = read_Fdataset_drug_smile()
    return g, smiles, ECPFs

def remove_graph(g, test_id):
    """Delete the drug-disease associations which belong to test set
    from heterogeneous network.
    """

    test_drug_id = test_id[:, 0]
    test_dis_id = test_id[:, 1]
    edges_id = g.edge_ids(th.tensor(test_drug_id),
                          th.tensor(test_dis_id),
                          etype=('drug', 'drug_disease', 'disease'))
    g = dgl.remove_edges(g, edges_id, etype=('drug', 'drug_disease', 'disease'))
    edges_id = g.edge_ids(th.tensor(test_dis_id),
                          th.tensor(test_drug_id),
                          etype=('disease', 'disease_drug', 'drug'))
    g = dgl.remove_edges(g, edges_id, etype=('disease', 'disease_drug', 'drug'))
    return g

