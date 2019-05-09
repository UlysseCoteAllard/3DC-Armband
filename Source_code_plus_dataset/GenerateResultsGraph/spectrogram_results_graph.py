import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.stats import wilcoxon


def label_diff(current_cycle, p_value, sign_to_use="="):
    x1, x2 = -.20+current_cycle, .20+current_cycle  # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    y, h, col = accuracies_3DC[current_cycle].mean() + accuracies_3DC[current_cycle].std()/2., 3, 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)

    plt.rcParams.update({"font.size": 45})
    plt.text((x1 + x2) * .5, y + h+0.01, "p" + sign_to_use + "%.4f" % p_value, ha='center', va='bottom', color=col, )


if __name__ == '__main__':
    accuracies_myo = []
    accuracies_3DC = []
    results_accuracies = []
    armbands = []
    cycles = []
    for cycle in range(1, 5):
        file_object_Myo = open("../results/results_MYO_Spectrograms_" + str(cycle) + "_cycles.txt", mode='r')

        accuracies_cycle = np.array([])
        for line in file_object_Myo:
            if "[" in line:
                print(line)
                line = line.strip("[")
                line = line.strip("]")
                accuracies_cycle = np.fromstring(line, dtype=float, sep=',')
                print(np.shape(accuracies_cycle))

        file_object_Myo.close()

        results_accuracies.extend(accuracies_cycle)
        cycles.extend(np.ones(len(accuracies_cycle)) * cycle)
        armbands.extend(['Myo'] * len(accuracies_cycle))

        accuracies_myo.append(accuracies_cycle)

        file_object_3dc = open("../results/results_3DC_Spectrograms_" + str(cycle) + "_cycles.txt", mode='r')

        accuracies_cycle = np.array([])
        for line in file_object_3dc:
            if "[" in line:
                print(line)
                line = line.strip("[")
                line = line.strip("]")
                accuracies_cycle = np.fromstring(line, dtype=float, sep=',')
                print(np.shape(accuracies_cycle))

        file_object_3dc.close()
        results_accuracies.extend(accuracies_cycle)
        cycles.extend(np.ones(len(accuracies_cycle))*cycle)
        armbands.extend(['3DC Armband'] * len(accuracies_cycle))

        accuracies_3DC.append(accuracies_cycle)

    results_accuracies = np.array(results_accuracies)
    df = pd.DataFrame(results_accuracies, columns=["Accuracy (%)"])
    df["Number of Training Cycles"] = pd.Series(cycles, dtype=int)
    df["Armband"] = pd.Series(armbands)
    sns.set(style="white", font_scale=4)
    ax = sns.barplot(data=df, x="Number of Training Cycles", y="Accuracy (%)", hue="Armband")
    legend = ax.legend()
    legend.texts[0].set_text("Myo Armband")
    sns.set(style="dark", font_scale=2)
    sns.despine()
    ax.set_ylim([55, 95])

    for cycle in range(4):
        stat, p = wilcoxon(accuracies_3DC[cycle], accuracies_myo[cycle])
        print(p)
        if p < 0.05:
            p_rounded = np.round(p, decimals=5)
            if p_rounded > 0:
                label_diff(current_cycle=cycle, p_value=p_rounded, sign_to_use="=")
            else:
                label_diff(current_cycle=cycle, p_value=0.0001, sign_to_use="<")
    plt.show()
