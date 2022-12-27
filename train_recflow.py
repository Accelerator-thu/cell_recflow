import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wot
import anndata as ad
import scanpy as sc
import sklearn.neighbors as nb
from scipy import stats

from utils import *
from rectifiedflow import *

def main():
    # Data loading

    FLE_COORDS_PATH ='data/fle_coords.txt'
    FULL_DS_PATH = 'data/ExprMatrix.h5ad'
    VAR_DS_PATH = 'data/ExprMatrix.var.genes.h5ad'
    CELL_DAYS_PATH = 'data/cell_days.txt'
    GENE_SETS_PATH = 'data/gene_sets.gmx'
    GENE_SET_SCORES_PATH = 'data/gene_set_scores.csv'
    CELL_SETS_PATH = 'data/cell_sets.gmt'

    coord_df = pd.read_csv(FLE_COORDS_PATH, index_col='id', sep='\t')
    days_df = pd.read_csv(CELL_DAYS_PATH, index_col='id', sep='\t')
    adata = wot.io.read_dataset(FULL_DS_PATH, obs=[CELL_DAYS_PATH])

    days = adata.obs['day'].unique()[:-1]
    indices = [adata.obs['day'].isin([day]) for day in days]
    adatalist = [adata[index] for index in indices]

    pcadata = np.load('adata_dim_400.npy')
    pcadatalist = [pcadata[index] for index in indices]


    index_pair_list = [(i, i + 1) for i in range(15)]
    # days[index_pair_list]
    n_obs_list = [pcadatalist[i].shape[0] for i in range(len(pcadatalist))]

    iterations = 2000
    batchsize = 2048
    input_dim = 400
    hidden_dim = 256
    recFlow_dict = {}
    loss_curve_dict = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pairs_train = torch.load('pairs_train.pt')
    recFlow = RectifiedFlow(n_obs_list=n_obs_list, model=MLP(input_dim=input_dim, hidden_num=hidden_dim), num_steps=1000, reverse=False)
    optimizer = torch.optim.Adam(recFlow.model.parameters(), lr=5e-3)
    recflow, loss_curve = train_rectified_flow(recFlow, optimizer, pairs_train, batchsize, iterations)
    # recFlow_dict[f'hidden_dim_{hidden_dim}'], loss_curve_dict[f'hidden_dim_{hidden_dim}'] = recflow, loss_curve

    target = recflow.simulate_target(z0=torch.arange(5061), zid=15, N=1000)
    
    test_crt_pcadatapair = np.load('pcadata/pcadata_contiguous_15_16.npy', allow_pickle=True)[0]
    ref_crt_pcadata = test_crt_pcadatapair[:n_obs_list[15]]
    new_crt_pcadata = test_crt_pcadatapair[n_obs_list[15]:]

    # n_neighbors = 50
    # KNN = nb.NearestNeighbors(n_neighbors=n_neighbors)
    # KNN.fit(ref_crt_pcadata)

    PATH_TMAP = 'tmaps/tmaps_7.5_8.0.h5ad'
    tmaps = wot.io.read_dataset(PATH_TMAP)
    tensor_tmapX = torch.tensor(tmaps.X)
    target = target.cpu()

    dot_sim = torch.matmul(torch.tensor(ref_crt_pcadata), target.cpu().T)
    cos_sim = dot_sim / torch.unsqueeze(torch.norm(target.cpu(), p=2, dim=1), 0) / torch.unsqueeze(torch.norm(torch.tensor(ref_crt_pcadata.T), p=2, dim=0), 1)
    corrs = torch.zeros(new_crt_pcadata.shape[0])
    # cvrg = torch.zeros(new_crt_pcadata.shape[0])
    # print(tensor_tmapX.shape)
    # print(dot_sim.shape)
    for i in range(new_crt_pcadata.shape[0]):
        # ind_seq = tensor_tmapX[:, i].argsort()[-200:]
        # corrs[i] = stats.pearsonr(tensor_tmapX[:, i][ind_seq], dot_sim[i][ind_seq]).statistic
        # corrs[i] = stats.pearsonr(tensor_tmapX[:, i], torch.softmax(dot_sim[:, i], 0)).statistic
        # corrs[i] = stats.pearsonr(torch.nan_to_num(torch.log(tensor_tmapX[:, i])), cos_sim[:, i]).statistic
        # corrs[i] = stats.spearmanr(tensor_tmapX[:, i][ind_seq], cos_sim[:, i][ind_seq]).correlation
        corrs[i] = stats.spearmanr(tensor_tmapX[:, i], cos_sim[:, i]).correlation
        # _, indices = KNN.kneighbors(target[i].reshape(1, -1))
        # summation = tensor_tmapX[:, i][indices].sum()
        # cvrg[i] = summation / tensor_tmapX[:, i].sort()[0][-n_neighbors:].sum()
        
        
    plt.figure(figsize=(10, 7.5))
    plt.title('Spearman Correlation')
    # plt.title(f'Coverage when n = {n_neighbors}')
    # plt.title('Pearson Correlation between Cosine Similarity and Transition MAP')
    plt.hist(corrs)

    # plt.scatter(torch.log(tensor_tmapX[:, 0]), dot_sim[:, 0])

    # ind_seq = torch.argsort(tensor_tmapX[0])[-50:]
    # tensor_tmapX[0][ind_seq]

    # plt.xlabel('TMAP')
    # plt.ylabel('Cosine Similarity')
    # plt.scatter(torch.log(tensor_tmapX[0]), cos_sim[0])




if __name__ == '__main__':
    main()