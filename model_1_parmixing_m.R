library(copula)
library(stats)
source('kPCclustering.R')


generate_copula <- function(){
    thetabase = copClayton@iTau(0.5)
    opow.Clayton = opower(copClayton, thetabase)
    theta = c(1,10/7,10/7)
    copoc = onacopulaL(opow.Clayton, list(theta[1], NULL, list(
                                                                list(theta[2], 1:100), #1:800
                                                                list(theta[3], 101:200) #801:1600
    )))
    return(copoc)
}


generate_randomness <- function(nobservation, p, d,seed, copoc){
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

SKmeans <- function(seed, n_sample, p, d,m, k){
    copoc = generate_copula()
    sample = generate_randomness(n_sample, p, d,seed, copoc)
    data = t(robservation(sample, m, d, k))

    nrms=sqrt(rowSums(data^2))
    data=data/nrms
    centroids = clusterMeans(data, k = 2)
    output = getClusterIndex(data, centroids)
    return(output)
}

#copoc = generate_copula()
#print(copoc)
#
#sample = generate_randomness(1000, 0.9,200, 1, copoc)
#
#data = t(robservation(sample, 10, 200, 1000 / 10))
#
#nrms = sqrt(rowSums(data^2))
#data = data / nrms
#
#centroids = clusterMeans(data, k = 2)
#output = getClusterIndex(data, centroids)
#
#print(output)
