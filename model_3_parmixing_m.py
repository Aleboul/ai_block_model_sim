import rpy2.robjects as robjects
r = robjects.r
r['source']('model_3_parmixing_m.R')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import multiprocessing as mp
from sklearn.cluster import AgglomerativeClustering

def ecdf(X):
    """ Compute uniform ECDF.
    
    Inputs
    ------
        X (np.array[float]) : array of observations
    Output
    ------
        Empirical uniform margin
    """

    index = np.argsort(X)
    ecdf = np.zeros(len(index))

    for i in index :
        ecdf[i] = (1.0 / X.shape[0]) * np.sum(X <= X[i])

    return ecdf

def theta(R, mode = "extremal_coefficient") :
    """
        This function computes the w-madogram
        Inputs
        ------
        R (array([float]) of n_sample \times d) : rank's matrix
                                              w : element of the simplex
                            miss (array([int])) : list of observed data
                           corr (True or False) : If true, return corrected version of w-madogram
        
        Outputs
        -------
        w-madogram
    """

    Nnb = R.shape[1]
    Tnb = R.shape[0]
    V = np.zeros([Tnb, Nnb])
    for j in range(0, Nnb):
        V[:,j] = R[:,j]
    value_1 = np.amax(V,1)
    value_2 = (1/Nnb) * np.sum(V, 1)
    mado = (1/Tnb) * np.sum(value_1 - value_2)

    if mode == "extremal_coefficient":
        value = (mado + 1/2) / (1/2-mado)
    if mode == "madogram":
        value = mado
    return value

def find_max(M, S):
    mask = np.zeros(M.shape, dtype = bool)
    values = np.ones((len(S),len(S)),dtype = bool)
    mask[np.ix_(S,S)] = values
    np.fill_diagonal(mask,0)
    max_value = M[mask].max()
    i , j = np.where(np.multiply(M, mask * 1) == max_value) # Sometimes doublon happens for excluded clusters, if n is low
    return i[0], j[0]

def clust(Theta, n, alpha = None):
    """ Performs clustering in AI-block model
    
    Inputs
    ------
        Theta : extremal correlation matrix
        alpha : threshold, of order sqrt(ln(d)/n)
    
    Outputs
    -------
        Partition of the set \{1,\dots, d\}
    """
    d = Theta.shape[1]

    # Initialisation

    S = np.arange(d)
    l = 0

    if alpha is None:
        alpha = 2 * np.sqrt(np.log(d)/n)
    
    cluster = {}
    while len(S) > 0:
        l = l + 1
        if len(S) == 1:
            cluster[l] = np.array(S)
        else :
            a_l, b_l = find_max(Theta, S)
            if Theta[a_l,b_l] < alpha :
                cluster[l] = np.array([a_l])
            else :
                index_a = np.where(Theta[a_l,:] >= alpha)
                index_b = np.where(Theta[b_l,:] >= alpha)
                cluster[l] = np.intersect1d(S,index_a,index_b)
        S = np.setdiff1d(S, cluster[l])
    
    return cluster

def init_pool_processes():
    sp.random.seed()

def make_sample(n_sample, d, k, m, p, seed, K):
    generate_copula = robjects.globalenv['generate_copula']
    generate_randomness = robjects.globalenv['generate_randomness']
    robservation = robjects.globalenv['robservation']
    copoc = generate_copula(d = d, K = K, seed = seed)
    sizes = copoc[1]
    copoc = copoc[0]
    sample = generate_randomness(n_sample, p = p, d = d, seed = seed, copoc = copoc)
    data = robservation(sample, m = m, d = d, k = k)

    O_bar = {}
    l = 0
    _d = 0
    for d_ in sizes:
        l += 1
        O_bar[l] = np.arange(_d, _d + d_)
        _d += d_

    return np.array(data), O_bar

def perc_exact_recovery(O_hat, O_bar):
    value = 0
    for key1, true_clust in O_bar.items():
        for key2, est_clust in O_hat.items():
            if len(true_clust) == len(est_clust):
                test = np.intersect1d(true_clust,est_clust)
                if len(test) > 0 and len(test) == len(true_clust) and np.sum(np.sort(test) - np.sort(true_clust)) == 0 :
                    value +=1
    return value / len(O_bar)

def operation_model_1_ECO(dict, seed):
    """ Operation to perform Monte carlo simulation
    Input
    -----
        dict : dictionnary containing
                - d1 : dimension of the first sample
                - d2 : dimension of the second sample
                - n_sample : sample's length
    """

    sp.random.seed(1*seed)
    # Generate first sample
    sample, O_bar = make_sample(n_sample = dict['n_sample'], d = dict['d'], k = dict['k'], m = dict['m'], p = dict['p'], seed = seed, K = dict['K'])
    # initialization
    d = sample.shape[1]

    R = np.zeros([dict['k'], d])
    for j in range(0,d):
        X_vec = sample[:,j]
        R[:,j] = ecdf(X_vec)
    
    Theta = np.ones([d,d])
    for j in range(0,d):
        for i in range(0,j):
            Theta[i,j] = Theta[j,i] = 2 - theta(R[:,[i,j]], mode = "extremal_coefficient")


    O_hat = clust(Theta, n = dict['k'])

    perc = perc_exact_recovery(O_hat, O_bar)

    return perc

def operation_model_1_HC(dict, seed):
    """ Operation to perform Monte carlo simulation

    Input
    -----
        dict : dictionnary containing
                - d1 : dimension of the first sample
                - d2 : dimension of the second sample
                - n_sample : sample's length
    """

    sp.random.seed(1*seed)
    sample, O_bar = make_sample(n_sample = dict['n_sample'], d = dict['d'], k = dict['k'], m = dict['m'], p = dict['p'], seed = seed, K = dict['K'])
    # initialization
    d = sample.shape[1]

    R = np.zeros([dict['k'], d])
    for j in range(0,d):
        X_vec = sample[:,j]
        R[:,j] = ecdf(X_vec)
    
    Theta = np.ones([d,d])
    for j in range(0,d):
        for i in range(0,j):
            Theta[i,j] = Theta[j,i] =theta(R[:,[i,j]], mode = "madogram")

    HC = AgglomerativeClustering(n_clusters=dict['K'], affinity = 'precomputed', linkage = 'average').fit(Theta)
    labels = HC.labels_
    label = np.unique(labels)
    O_hat = {}

    for lab, l in enumerate(label):
        l += 1
        index = np.where(labels == lab)
        O_hat[l] = index[0]

    perc = perc_exact_recovery(O_hat, O_bar)

    return perc

def operation_model_1_SKmeans(dict, seed):
    """ Operation to perform Monte carlo simulation
    Input
    -----
        dict : dictionnary containing
                - d1 : dimension of the first sample
                - d2 : dimension of the second sample
                - n_sample : sample's length
    """
    SKmeans = robjects.globalenv['SKmeans']
    sp.random.seed(1*seed)
    # initialization
    # function(K, seed, n_sample, p, d, copoc, m, k)
    labels = SKmeans(K = dict['K'], seed = seed, n_sample = dict['n_sample'], p = dict['p'], d = dict['d'], m = dict['m'], k = dict['k'])
    sizes = labels[1]
    labels = np.array(labels[0])
    label = np.unique(labels)
    O_hat = {}

    for lab, l in enumerate(label):
        lab += 1
        index = np.where(labels == lab)
        O_hat[l] = np.int_(index[0])
    O_bar = {}
    l = 0
    _d = 0
    for d_ in sizes:
        l += 1
        O_bar[l] = np.arange(_d, _d + d_)
        _d += d_

    perc = perc_exact_recovery(O_hat, O_bar)

    return perc

d = 200
#m_sample = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
m_sample = [3,6,9,12,15,18,21,24,27,30]
n = 10000
n_iter = 100
K = 10
p = 0.5
pool = mp.Pool(processes= 10, initializer=init_pool_processes)
mode = "ECO"

stockage = []

for m in m_sample:

    k = int(np.floor(n / m))

    input = {'n_sample' : k * m, 'd' : d, 'k' : k, 'm' : m, 'p' : p, 'K' : K}

    if mode == "ECO":

        result_objects = [pool.apply_async(operation_model_1_ECO, args = (input,i)) for i in range(n_iter)]
    
    if mode == "HC":
        
        result_objects = [pool.apply_async(operation_model_1_HC, args = (input,i)) for i in range(n_iter)]
    
    if mode == "SKmeans":

        result_objects = [pool.apply_async(operation_model_1_SKmeans, args = (input,i)) for i in range(n_iter)]


    results = [r.get() for r in result_objects]

    stockage.append(results)

    df = pd.DataFrame(stockage)

    print(df)

pool.close()
pool.join()


df.to_csv('results_model_3_ECO_200_mixing_p0.5_1_m.csv')

## Attention, results_model_3_ECO_1600_mixing_p1.0_1_m.csv a été enregistré avec l'intitulé results_model_3_ECO_200_mixing_p0.9_1_m.csv
