import numpy as np
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix, vstack

def read_space(dm_file):
    with open(dm_file) as f:
        dmlines=f.readlines()
    f.close()

    m = []
    vocab = []
    for l in dmlines:
        items=l.rstrip().split()
        vocab.append(items[0])
        vec=[float(i) for i in items[1:]]
        m.append(vec)
    m = np.array(m)
    return m,vocab

def read_cols(cols_file):
    cols = []
    f = open(cols_file,'r')
    for l in f:
        cols.append(l.rstrip('\n'))
    return cols

def hash_input_vectorized_(pn_mat, weight_mat, percent_hash):
    kc_mat = pn_mat.dot(weight_mat.T)
    #print(pn_mat.shape,weight_mat.shape,kc_mat.shape)
    kc_use = np.squeeze(kc_mat.toarray().sum(axis=0,keepdims=1))
    kc_use = kc_use / sum(kc_use)
    kc_sorted_ids = np.argsort(kc_use)[:-kc_use.shape[0]-1:-1] #Give sorted list from most to least used KCs
    m, n = kc_mat.shape
    wta_csr = csr_matrix(np.zeros(n))
    for i in range(0, m, 2000):
        part = wta_vectorized(kc_mat[i: i+2000].toarray(), k=percent_hash)
        wta_csr = vstack([wta_csr, csr_matrix(part, shape=part.shape)])
    hashed_kenyon = wta_csr[1:]
    return hashed_kenyon, kc_use, kc_sorted_ids


def hash_dataset_(dataset_mat, weight_mat, percent_hash):
    m, n = dataset_mat.shape
    dataset_mat = csr_matrix(dataset_mat)
    hs, kc_use, kc_sorted_ids = hash_input_vectorized_(dataset_mat, weight_mat, percent_hash)
    hs = (hs > 0).astype(np.int_)
    return hs

def wta_vectorized(feature_mat, k):
    # thanks https://stackoverflow.com/a/59405060
    m, n = feature_mat.shape
    k = int(k * n / 100)
    # get (unsorted) indices of top-k values
    topk_indices = np.argpartition(feature_mat, -k, axis=1)[:, -k:]
    # get k-th value
    rows, _ = np.indices((m, k))
    kth_vals = feature_mat[rows, topk_indices].min(axis=1)
    # get boolean mask of values smaller than k-th
    is_smaller_than_kth = feature_mat < kth_vals[:, None]
    # replace mask by 0
    feature_mat[is_smaller_than_kth] = 0
    return feature_mat

def run_PCA(m, k):
    pca = PCA(n_components=k)
    pca.fit(m)
    return pca.transform(m)

