import rpy2.robjects as robjects
r = robjects.r
r['source']('model_1_parmixing_m.R')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import multiprocessing as mp
from sklearn.model_selection import train_test_split

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

def SECO(R_1, R_2, clst):
    """ evaluation of the criteria

    Input
    -----
        R (np.array(float)) : n x d rank matrix
                  w (float) : element of the simplex
           cols (list(int)) : partition of the columns

    Output
    ------
        Evaluate (theta - theta_\Sigma) 
    
    """

    d = R_1.shape[0]

    ### Evaluate the cluster as a whole

    value = theta(R_1)

    _value_ = []
    for key, c in clst.items():
        _R_2 = R_2[:,c]
        _value_.append(theta(_R_2))

    return np.sum(_value_) - value

def find_max(M, S):
    mask = np.zeros(M.shape, dtype = bool)
    values = np.ones((len(S),len(S)),dtype = bool)
    mask[np.ix_(S,S)] = values
    np.fill_diagonal(mask,0)
    max_value = M[mask].max()
    i , j = np.where(np.multiply(M, mask * 1) == max_value) # Sometimes doublon happens for excluded clusters, if n is low
    return i[0], j[0]

def clust(Theta, n, m,alpha = None):
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
        alpha = 2 * (1/m + np.sqrt(np.log(d)/n))
    
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

def make_sample(n_sample, d, k, m, p, seed):
    generate_randomness = robjects.globalenv['generate_randomness']
    robservation = robjects.globalenv['robservation']
    generate_copula = robjects.globalenv['generate_copula']
    copoc = generate_copula()
    sample = generate_randomness(n_sample, p = p, d = d, copoc = copoc, seed = seed)
    data = robservation(sample, m = m, d = d, k = k)

    return np.array(data)

def perc_exact_recovery(O_hat, O_bar):
    value = 0
    for key1, true_clust in O_bar.items():
        for key2, est_clust in O_hat.items():
            if len(true_clust) == len(est_clust):
                test = np.intersect1d(true_clust,est_clust)
                if len(test) > 0 and len(test) == len(true_clust) and np.sum(np.sort(test) - np.sort(true_clust)) == 0 :
                    value +=1
    return value / len(O_bar)

def operation_model_1CV_ECO(dict, seed, _alpha_):
    """ Operation to perform Monte carlo simulation

    Input
    -----
        dict : dictionnary containing
                - d1 : dimension of the first sample
                - d2 : dimension of the second sample
                - n_sample : sample's length
    """
    sp.random.seed(1*seed)

    sample = make_sample(n_sample = dict['n_sample'], d = dict['d1'] + dict['d2'], k = dict['k'], m = dict['m'], p = dict['p'], seed = seed)

    train_sample, test_sample = train_test_split(sample, train_size = 1/3)

    test_sample_1, test_sample_2 = train_test_split(test_sample, train_size = 1/2)
    # initialization
    d = train_sample.shape[1]

    R = np.zeros([train_sample.shape[0], d])
    for j in range(0,d):
        X_vec = train_sample[:,j]
        R[:,j] = ecdf(X_vec)
    
    Theta = np.ones([d,d])
    for j in range(0,d):
        for i in range(0,j):
            Theta[i,j] = Theta[j,i] = 2 - theta(R[:,[i,j]])
    output = []
    for alpha in _alpha_:
        O_hat = clust(Theta, n = dict['n_sample'], m = dict['m'], alpha = alpha)
        O_bar = {1 : np.arange(0,dict['d1']), 2 : np.arange(dict['d1'],d)}

        perc = perc_exact_recovery(O_hat, O_bar)

        R_1 = np.zeros([test_sample_1.shape[0], d])
        for j in range(0,d):
            X_vec = test_sample_1[:,j]
            R_1[:,j] = ecdf(X_vec)
    
        Theta = np.ones([d,d])
        for j in range(0,d):
            for i in range(0,j):
                Theta[i,j] = Theta[j,i] = 2 - theta(R_1[:,[i,j]])

        R_2 = np.zeros([test_sample_2.shape[0], d])
        for j in range(0,d):
            X_vec = test_sample_2[:,j]
            R_2[:,j] = ecdf(X_vec)

        Theta = np.ones([d,d])
        for j in range(0,d):
            for i in range(0,j):
                Theta[i,j] = Theta[j,i] = 2 - theta(R_2[:,[i,j]])

        seco = SECO(R_1, R_2, O_hat)
        output.append([perc, seco])


    return output

d1 = 800
d2 = 800
d = d1 + d2
k = 2400
m = 20
p = 0.9
n_iter = 100
_alpha_ = np.array([0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0]) * (1/m + np.sqrt(np.log(d) / (k // 3) ))
pool = mp.Pool(processes= 10, initializer=init_pool_processes)
mode = "ECO"
stockage = []

n_sample = k * m

input = {'d1' : d1, 'd2' : d2, 'n_sample' : n_sample, 'k' : k, 'm' : m, 'p' : p}

if mode == "ECO":

    result_objects = [pool.apply_async(operation_model_1CV_ECO, args = (input,i, _alpha_)) for i in range(n_iter)]

results = np.array([r.get() for r in result_objects])

print(results)

seco = []
perc = []
for r in results:
    seco.append(r[:,1])
    perc.append(r[:,0])

seco = pd.DataFrame(seco)
perc = pd.DataFrame(perc)

print(seco)
print(perc)

pool.close()
pool.join()


perc.to_csv('perc_model_1CV_ECO_1600_p0.5.csv')
seco.to_csv('seco_model_1CV_ECO_1600_p0.5.csv')
