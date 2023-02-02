library(copula)
library(stats)
source('kPCclustering.R')

generate_copula <- function(d, K, seed){
    set.seed(seed)
    thetabase = copClayton@iTau(0.5)
    opow.Clayton = opower(copClayton, thetabase)
    theta = c(1,10/7,10/7,10/7,10/7,10/7)
    probs = rep(0,K)
    for (k in 1:K-1){
        p = (1/2)**k
        probs[k] = p
    }
    probs[K] = 1 - sum(probs)
    sizes = as.vector(rmultinom(n = 1, size = d, prob = probs))
    sizes_sum = cumsum(sizes)
    copoc = onacopulaL(opow.Clayton, list(theta[1], NULL, list(
                                                                list(theta[2], 1:sizes_sum[1]),
                                                                list(theta[3], (sizes_sum[1]+1):sizes_sum[2]),
                                                                list(theta[4], (sizes_sum[2]+1):sizes_sum[3]),
                                                                list(theta[5], (sizes_sum[3]+1):sizes_sum[4]),
                                                                list(theta[6], (sizes_sum[4]+1):sizes_sum[5])
    )))

    return(list(copoc, sizes))
}

generate_randomness <- function(nobservation, p, d, seed, copoc){
    set.seed(seed)
    sample_ = rnacopula(nobservation, copoc)
    I_ = rbinom(nobservation, size = 1,prob = p)
    series = matrix(0, nobservation+1, d)
    series[1,] = rnacopula(1,copoc)
    for (i in 1:nobservation){
        if (I_[i] ==1){
            series[i+1,] = sample_[i,]
        }   else    {
            series[i+1,] = series[i,]
        }
    }
    return(series[-1,])
}

robservation <- function(randomness, m, d, k){
    mat = matrix(randomness, m)
    dat = apply(mat, 2, max)
    return(matrix(dat, k, d)^m)
}

SKmeans <- function(K, seed, n_sample, p, d, m, k){
    copoc = generate_copula(d,K,seed)
    sizes = copoc[[2]]
    copoc = copoc[[1]]
    sample = generate_randomness(n_sample, p, d, seed, copoc)
    data = t(robservation(sample, m, d, k))

    nrms=sqrt(rowSums(data^2))
    data=data/nrms
    centroids = clusterMeans(data, k = K)
    output = getClusterIndex(data, centroids)
    return(list(output, sizes))
}

#copoc = generate_copula(d = 200, K = 5, 1)[[1]]
#print(copoc)
#
#sample = generate_randomness(1000, 0.9,200, 1, copoc)
#
#data = t(robservation(sample, 10, 200, 1000 / 10))
#
#nrms = sqrt(rowSums(data^2))
#data = data / nrms
#
#centroids = clusterMeans(data, k = 5)
#output = getClusterIndex(data, centroids)
#
#print(output)