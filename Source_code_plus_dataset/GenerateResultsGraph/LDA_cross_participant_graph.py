import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.stats import wilcoxon, shapiro, ttest_rel
from sklearn.metrics import confusion_matrix as confusion_matrix_function
from sklearn.utils.multiclass import unique_labels


def label_diff(current_cycle, p_value, sign_to_use="="):
    x1, x2 = -.20+current_cycle, .20+current_cycle
    y, h, col = accuracies_3DC[current_cycle].mean() + accuracies_3DC[current_cycle].std()/3., 2, 'k'

    plt.rcParams.update({"font.size": 45})
    plt.text((x1 + x2) * .5, y + h, "p" + sign_to_use + "%.4f" % p_value, ha='center', va='bottom', color=col)
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=3., c=col)


def print_confusion_matrix(path_ground_truth, path_predictions, class_names, fontsize=24,
                           normalize=True, fig=None, axs=None, title=None):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    ground_truth = np.load(path_ground_truth)
    predictions = np.load(path_predictions)

    print(np.shape(predictions[0]))
    predictions = [x for y in predictions for x in y]  # Calculate the confusion matrix across all participants
    ground_truth = [x for y in ground_truth for x in y]
    #predictions = predictions[20]
    #ground_truth = ground_truth[20]
    confusion_matrix_calculated = confusion_matrix_function(ground_truth, predictions)
    if normalize:
        confusion_matrix_calculated = confusion_matrix_calculated.astype('float') /\
                                      confusion_matrix_calculated.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        print("Normalized confusion matrix")
    else:
        fmt = 'd'
        print('Confusion matrix, without normalization')
    df_cm = pd.DataFrame(
        confusion_matrix_calculated, index=class_names, columns=class_names,
    )
    if fig is None:
        fig, axs = plt.subplots(1, 2, figsize=(20, 20))
        index_axs = 0
    else:
        index_axs = 1

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt, ax=axs[index_axs], cbar=False)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    if index_axs == 0:
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        axs[index_axs].set(
            # ... and label them with the respective list entries
            title=title,
            ylabel='True label',
            xlabel='Predicted label')
    else:
        heatmap.yaxis.set_ticklabels("")
        axs[index_axs].set(
            # ... and label them with the respective list entries
            title=title,
            xlabel='Predicted label')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha='right', fontsize=fontsize)
    axs[index_axs].xaxis.label.set_size(fontsize + 4)
    axs[index_axs].yaxis.label.set_size(fontsize + 4)
    axs[index_axs].title.set_size(fontsize + 6)
    return fig, axs

if __name__ == '__main__':

    classes = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Supination",
               "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]
    font_size = 14
    sns.set(style='dark')


    # Generate the confusion matrix
    for cycle in range(4, 5):
        fig, axs = print_confusion_matrix("../results/ground_truth_MYO_cross_subjects_LDA_" + str(cycle) + "_cycles.npy", "../results/predictions_MYO_cross_subjects_LDA_" + str(cycle) + "_cycles.npy", classes, title="Myo Armband", fontsize=font_size)
        print_confusion_matrix("../results/ground_truth_LDA_cross_subject_3DC_" + str(cycle) + "_cycles.npy", "../results/pedictions_LDA_cross_subject_3DC_" + str(cycle) + "_cycles.npy", classes, fig=fig, axs=axs, title="3DC Armband", fontsize=font_size)
        if cycle == 1:
            fig.suptitle("LDA classifier on never-seen-before participants with " + str(cycle) + " cycle of training",
                         fontsize=28)
        else:
            fig.suptitle("LDA classifier on never-seen-before participants with " + str(cycle) + " cycles of training",
                         fontsize=28)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')  # works fine on Windows!
        plt.tight_layout()
        plt.gcf().subplots_adjust(bottom=0.13)
        plt.gcf().subplots_adjust(top=0.90)

        plt.show()

    accuracies_myo = []
    accuracies_3DC = []
    results_accuracies = []
    armbands = []
    cycles = []
    for cycle in range(1, 5):
        file_object_Myo = open("../results/results_MYO_cross_subjects_LDA_" + str(cycle) + "_cycles.txt", mode='r')

        accuracies_cycle = np.array([])
        for line in file_object_Myo:
            if "[" in line:
                print(line)
                line = line.strip("[")
                line = line.strip("]")
                accuracies_cycle = np.fromstring(line, dtype=float, sep=',') * 100.
                print(np.shape(accuracies_cycle))

        file_object_Myo.close()

        results_accuracies.extend(accuracies_cycle)
        cycles.extend(np.ones(len(accuracies_cycle)) * cycle)
        armbands.extend(['Myo Armband'] * len(accuracies_cycle))

        accuracies_myo.append(accuracies_cycle)

        file_object_3dc = open("../results/results_LDA_cross_subject_3DC_" + str(cycle) + "_cycles.txt", mode='r')

        accuracies_cycle = np.array([])
        for line in file_object_3dc:
            if "[" in line:
                print(line)
                line = line.strip("[")
                line = line.strip("]")
                accuracies_cycle = np.fromstring(line, dtype=float, sep=',') * 100.
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
    ax.set_ylim([20, 40])
    plt.subplots_adjust(left=0.08, right=1., top=0.93, bottom=0.14)

    legend = ax.legend()
    ax.legend().draggable()
    legend.texts[0].set_text("Myo Armband")
    sns.set(style="dark", font_scale=4)
    sns.despine()
    plt.title("Accuracy for never-seen-before participants \n employing LDA for classification")

    for cycle in range(4):
        _, normality_p_value = shapiro(accuracies_3DC[cycle] - accuracies_myo[cycle])
        if(normality_p_value < 0.1):
            stat, p = wilcoxon(accuracies_3DC[cycle], accuracies_myo[cycle])
            print("p-value : ", p)
            if p < 0.05:
                p_rounded = np.round(p, decimals=5)
                if p_rounded > 0:
                    label_diff(current_cycle=cycle, p_value=p_rounded, sign_to_use="=")
                else:
                    label_diff(current_cycle=cycle, p_value=0.0001, sign_to_use="<")
        else:
            stat, p = ttest_rel(accuracies_3DC[cycle], accuracies_myo[cycle])
            print("p-value : ", p)
            if p < 0.05:
                p_rounded = np.round(p, decimals=5)
                if p_rounded > 0:
                    label_diff(current_cycle=cycle, p_value=p_rounded, sign_to_use="=")
                else:
                    label_diff(current_cycle=cycle, p_value=0.0001, sign_to_use="<")
        print("Normality : ", shapiro(accuracies_3DC[cycle] - accuracies_myo[cycle]))
    mng = plt.get_current_fig_manager()
    plt.legend(loc='upper left')
    mng.window.state('zoomed')  # works fine on Windows!

    #plt.tight_layout()
    #plt.gcf().subplots_adjust(bottom=0.13)
    #plt.gcf().subplots_adjust(top=0.90)
    plt.show()
