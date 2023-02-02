import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('qb-light.mplstyle')

char_model = ['model_1', 'model_2', 'model_3']
char_mixing = ['p0.5', 'p0.7', 'p0.9', 'p1.0']
k = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

for mixing in char_mixing:
    figure, axis = plt.subplots(1, 3, sharex=True, sharey=True)
    for index, model in enumerate(char_model):

        path = model + "/" + mixing + "/"

        DF_ECO_200 = pd.read_csv(path + 'results_ECO_200.csv', index_col=0)
        DF_ECO_1600 = pd.read_csv(path + 'results_ECO_1600.csv', index_col=0)
        DF_HC_1600 = pd.read_csv(path + 'results_HC_1600.csv', index_col=0)
        DF_SKmeans_1600 = pd.read_csv(
            path + 'results_SKmeans_1600.csv', index_col=0)

        exact_recov_rate_ECO_200 = DF_ECO_200.mean(axis=1)
        exact_recov_rate_ECO_1600 = DF_ECO_1600.mean(axis=1)
        exact_recov_rate_HC_1600 = DF_HC_1600.mean(axis=1)
        exact_recov_rate_SKmeans_1600 = DF_SKmeans_1600.mean(axis=1)

        axis[index].plot(k, exact_recov_rate_HC_1600, marker='P', linestyle='solid',
                         markerfacecolor='#B6BFE0', label='Oracle-HC-1600', c='#6E7FC2')
        axis[index].plot(k, exact_recov_rate_SKmeans_1600, marker='*', linestyle='solid',
                         markerfacecolor='#C0CCC5', label='Oracle-SKmeans-1600', c='#819A8C')
        axis[index].plot(k, exact_recov_rate_ECO_200, marker='D', linestyle='dotted',
                         markerfacecolor='#EFAD9A', label='ECO-200', c='#DF5C35')
        axis[index].plot(k, exact_recov_rate_ECO_1600, marker='D', linestyle='solid',
                         markerfacecolor='#EFAD9A', label='ECO-1600', c='#DF5C35')
        num_exp = str(index + 1)
        axis[index].set_title("E"+num_exp)

    axis[1].set_xlabel("k")
    axis[1].legend(prop={'size': 12})
    axis[0].set_ylabel("Exact Recovery Rate")
    figure.suptitle('F2', ha='left', va='center', x=0.05,
                    y=0.5, rotation=90, fontsize=14)
    figure.set_size_inches(15, 5)
    figure.savefig(mixing + "_k.pdf", dpi=1)
