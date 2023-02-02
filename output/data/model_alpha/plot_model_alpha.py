import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('qb-light.mplstyle')

char_model = ['model_1', 'model_2', 'model_3']
char_mixing = ['p0.5', 'p0.7', 'p0.9', 'p1.0']
tau = np.array([0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0])

for mixing in char_mixing:
    figure, axis1 = plt.subplots(1, 3, sharex=True, sharey=True)
    for index, model in enumerate(char_model):

        path = model + "/" + mixing + "/"

        SECO = pd.read_csv(path + 'seco_ECO_1600.csv', index_col=0)
        PERC = pd.read_csv(path + 'perc_ECO_1600.csv', index_col=0)

        mean_seco = SECO.mean(axis = 0)
        mean_perc = PERC.mean(axis = 0)
        mean_seco = np.log(1+mean_seco - np.min(mean_seco))

        axis1[index].plot(tau, mean_perc, marker='D', linestyle='dotted',
                         markerfacecolor='#B6BFE0', c='#6E7FC2')
        ax2 = axis1[index].twinx()
        ax2.plot(tau, mean_seco, marker='D', linestyle='solid',
                         markerfacecolor='#EFAD9A', label='ECO-1600', c='#DF5C35')
        ax2.grid(None)
        num_exp = str(index + 1)
        axis1[index].set_title("E"+num_exp)

    axis1[1].set_xlabel(r"$\frac{\alpha}{1/m + \sqrt{\ln(d)/k}}$")
    axis1[0].set_ylabel("Exact Recovery Rate")
    ax2.set_ylabel('SECO')
    figure.suptitle('F3', ha='left', va='center', x=0.05,
                    y=0.5, rotation=90, fontsize=14)
    figure.set_size_inches(15, 5)
    figure.savefig(mixing + "_alpha.pdf", dpi=1)
