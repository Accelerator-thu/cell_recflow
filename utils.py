import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wot
import anndata as ad
# import mnnpy as mp
import scanpy as sc
import sklearn.neighbors as nb
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

def NNVerification(index_pair, n_neighbors=50, do_concatenate=True, FULL_DS_PATH='data/ExprMatrix.h5ad', CELL_DAYS_PATH='data/cell_days.txt', split_mode='day'):
    time_ref, time_new = index_pair
    assert time_new == time_ref + 1, 'Noncontiguous time indices.'
    mnn_path = f"pcadata/{time_ref}_{time_new}.csv"
    crt_path = f"pcadata/pcadata_contiguous_{time_ref}_{time_new}.npy"
    print("Loading from "+mnn_path)
    mnn_pair = pd.read_csv(mnn_path)
    print("Loading from "+crt_path)
    crt_pcadata = np.load(crt_path, allow_pickle=True)
    # print(crt_pcadata)
    
    adata = wot.io.read_dataset(FULL_DS_PATH, obs=[CELL_DAYS_PATH])
    obs_unique = filter(None, adata.obs[split_mode].unique())
    adata_indices = [adata.obs[split_mode].isin([obs_uni_ele]) for obs_uni_ele in obs_unique]
    adatalist = [adata[index] for index in adata_indices]
    checkpoints = adata.obs[split_mode].unique()
    print(f"Time pair: day {checkpoints[time_ref]}, day {checkpoints[time_new]}")
    
    KNN = nb.NearestNeighbors(n_neighbors=n_neighbors)

    n_obs_ref = adata[adata.obs[split_mode].isin([checkpoints[time_ref]])].n_obs
    n_obs_new = adata[adata.obs[split_mode].isin([checkpoints[time_new]])].n_obs
    
    print(n_obs_ref)
    print(n_obs_new)
    pcadata_ref = pcadata_new = None
    if do_concatenate:
        pcadata_ref = crt_pcadata[0][:n_obs_ref]
        pcadata_new = crt_pcadata[0][n_obs_ref:]
    # print(crt_pcadata)
    # print(pcadata_ref)
    KNN.fit(pcadata_new)
    
    paired_ref_cell_indices = mnn_pair['ref cell'].unique()
    paired_new_cell_indices = mnn_pair['new cell'].unique()
    
    dists, knn_indices = KNN.kneighbors(pcadata_ref)
    intersection = []
    
    for paired_ref_cell_index in paired_ref_cell_indices:
        knn_set = set(knn_indices[paired_ref_cell_index])
        new_cell_indices = mnn_pair['ref cell'] == paired_ref_cell_index
        mnn_set = set(mnn_pair[new_cell_indices]['new cell'])
        intersection.append(len(knn_set & mnn_set) / len(mnn_set))
        
    intersection = np.array(intersection)
        
    print(f"Average intersection ratio: {np.average(intersection)}")
    
    unpaired_ref_cell_indices = [index for index in range(n_obs_ref) if index not in paired_ref_cell_indices]
    unpaired_new_cell_indices = [index for index in range(n_obs_new) if index not in paired_new_cell_indices]
    
    return intersection, unpaired_ref_cell_indices, unpaired_new_cell_indices
    


def gen_pairs_train(index_pair, n_obs_list, unpaired_ref_cells, unpaired_new_cells):
    mnn_pair = pd.read_csv(f"pcadata/{index_pair[0]}_{index_pair[1]}.csv")
    mnn_pair_ref_new_pair_indices = torch.from_numpy(np.stack((mnn_pair['ref cell'], mnn_pair['new cell']), axis=0)).T

    crt_pcadata = np.load(f'pcadata/pcadata_contiguous_{index_pair[0]}_{index_pair[1]}.npy', allow_pickle=True)
    ref_crt_pcadata = crt_pcadata[0][:n_obs_list[index_pair[0]]]
    new_crt_pcadata = crt_pcadata[0][n_obs_list[index_pair[0]]:]

    # print(unpaired_new_cells)
    unpaired_ref_cell_pca_adata = ref_crt_pcadata[unpaired_ref_cells]
    unpaired_new_cell_pca_adata = new_crt_pcadata[unpaired_new_cells]

    KNN_ref = nb.NearestNeighbors(n_neighbors=50)
    KNN_new = nb.NearestNeighbors(n_neighbors=50)
    KNN_ref.fit(ref_crt_pcadata)
    KNN_new.fit(new_crt_pcadata)

    _, unpaired_ref_cell_knn_indices = KNN_new.kneighbors(unpaired_ref_cell_pca_adata)
    _, unpaired_new_cell_knn_indices = KNN_ref.kneighbors(unpaired_new_cell_pca_adata)

    rd_ref_cols = np.random.randint(50, size=(unpaired_ref_cell_knn_indices.shape[0] * 3))
    rd_new_cols = np.random.randint(50, size=(unpaired_new_cell_knn_indices.shape[0] * 3))
    rd_ref_rows = np.repeat(np.arange(unpaired_ref_cell_knn_indices.shape[0]), 3, axis=0)
    rd_new_rows = np.repeat(np.arange(unpaired_new_cell_knn_indices.shape[0]), 3, axis=0)
    unpaired_ref_rows = np.repeat(np.array(unpaired_ref_cells), 3, axis=0)
    unpaired_new_rows = np.repeat(np.array(unpaired_new_cells), 3, axis=0)

    sampled_unpaired_ref_cell_knn_indices = np.array([unpaired_ref_cell_knn_indices[rd_ref_rows[i]][rd_ref_cols[i]] for i in range(len(rd_ref_cols))])
    sampled_unpaired_new_cell_knn_indices = np.array([unpaired_new_cell_knn_indices[rd_new_rows[i]][rd_new_cols[i]] for i in range(len(rd_new_cols))])

    knn_init_ref = torch.from_numpy(np.stack((unpaired_ref_rows, sampled_unpaired_ref_cell_knn_indices)).T)
    knn_init_new = torch.from_numpy(np.stack((sampled_unpaired_new_cell_knn_indices, unpaired_new_rows)).T)

    pairs_train = torch.concat((mnn_pair_ref_new_pair_indices, knn_init_ref, knn_init_new), dim=0)
    # rd_ref_indices = np.stack(rd_ref_rows, rd_ref_cols)
    return pairs_train, ref_crt_pcadata, new_crt_pcadata


