"""Use the fruit fly to do locality sensitive binary hashing
Usage:
  run_fly.py --dataset=<str> 
  run_fly.py (-h | --help)
  run_fly.py --version

Options:
  -h --help                    Show this screen.
  --version                    Show version.
  --dataset=<str>              Name of dataset, either bnc or wiki.
"""

from utils import run_PCA
import random
from docopt import docopt
from utils import read_space, hash_dataset_, read_cols
import MEN
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import hstack, vstack, lil_matrix, coo_matrix
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing

class Fly():

    def __init__(self, pn_size=None, kc_size=None, wta=None, proj_size=None):
        self.pn_size = pn_size
        self.kc_size = kc_size
        self.wta = wta
        self.proj_size = proj_size
        weight_mat, self.shuffled_idx = self.create_projections(self.proj_size)
        self.projections = lil_matrix(weight_mat)
        print("     INIT FLY:",self.pn_size,self.kc_size,self.proj_size,self.wta)


    def create_projections(self,proj_size):
        weight_mat = np.zeros((self.kc_size, self.pn_size))
        idx = list(range(self.pn_size))
        random.shuffle(idx)
        used_idx = idx.copy()
        c = 0

        while c < self.kc_size:
            for i in range(0,len(idx),proj_size):
                p = idx[i:i+proj_size]
                for j in p:
                    weight_mat[c][j] = 1
                c+=1
                if c >= self.kc_size:
                    break
            random.shuffle(idx) #reshuffle if needed -- if all KCs are not filled
            used_idx.extend(idx)
        return weight_mat, used_idx[:self.kc_size * proj_size]

    def hash_space(self,m):
        return hash_dataset_(dataset_mat=m, weight_mat=self.projections, percent_hash=self.wta)

    def similarities(self,m=None,metric='cosine'):
        similarities = None
        if metric == 'cosine':
            similarities = 1-pairwise_distances(m, metric="cosine")
        if metric == 'hamming':
            similarities = 1-pairwise_distances(m.todense(), metric="hamming")
        return similarities

    def evaluate(self,m,vocab,metric):
        sims = self.similarities(m,metric)
        return MEN.compute_men_spearman(sims,vocab,MEN_annot)

    def print_projections(self,vocab,k):
        for row in self.projections[:k]:
            words = ''
            cs = np.where(row.toarray()[0] == 1)[0]
            for i in cs:
                words+=vocab[i]+' '
            words+='|'
            print(words)




if __name__ == '__main__':
    args = docopt(__doc__, version='Test fly, ver 0.1')
    dataset = args["--dataset"]


    if dataset == "bnc":
        data = "data/BNC-MEN.dm"
        column_labels = "data/BNC-MEN.cols"
        MEN_annot = "data/MEN_dataset_lemma_form_full"
    else:
        data = "data/wiki_all.dm"
        column_labels = "data/wiki_all.cols"
        MEN_annot = "data/MEN_dataset_natural_form_full"

    m_original, vocab = read_space(data)
    scaler = preprocessing.MinMaxScaler().fit(m_original)
    m_original = scaler.transform(m_original)
    cols = read_cols(column_labels)
    print("VOCAB SIZE:",len(vocab))
    print("DIMENSIONALITY OF RAW SPACE:",len(cols))

    best_hyperparameters_before_flying = {'kpca':0}
    best_hyperparameters_after_flying = {'proj_size':0, 'kc_size':0, 'wta':0, 'kpca':0}

    best_score_before = 0
    best_score_after = 0

    for proj_size in [6,10,15]:
        for kc_size in [1000,3000,5000]:
            for wta in [2,5,10]:

                #Running PCA with different numbers of components
                for k in [None,32,64,128,256]:
                    if k != None:
                        m = run_PCA(m_original,k)
                    else:
                        m = m_original.copy()
                    pn_size = m.shape[1]
                    print("***")
                    print("SIZES PN LAYER:",pn_size,"KC LAYER:",kc_size)
                    print("SIZE OF PROJECTIONS:",proj_size)
                    print("SIZE OF FINAL HASH:",wta,"%")
                    print("NUM PCA COMPONENTS:", k)

                    #Init fly

                    fly = Fly(pn_size, kc_size, wta, proj_size)
                    sp, count = fly.evaluate(m,vocab,'cosine')
                    print ("      SPEARMAN BEFORE FLYING:",sp, "(calculated over",count,"items.)")
                    if sp > best_score_before:
                        best_score_before = sp
                        best_hyperparameters_before_flying = {'kpca':k}

                    hashed_m = fly.hash_space(m)
                    sp, count = fly.evaluate(hashed_m,vocab,'hamming')
                    print ("      SPEARMAN AFTER FLYING:",sp, "(calculated over",count,"items.)")
                    if sp > best_score_after:
                        best_score_after = sp
                        best_hyperparameters_after_flying = {'proj_size':proj_size, 'kc_size':kc_size, 'wta':wta, 'kpca':k}

    print("\n\n*** FINAL RESULTS ***")
    print("BEST SCORE BEFORE FLYING:", best_score_before, best_hyperparameters_before_flying)
    print("BEST SCORE AFTER FLYING:", best_score_after, best_hyperparameters_after_flying)
