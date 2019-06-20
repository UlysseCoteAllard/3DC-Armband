import numpy as np
from scipy import signal

def butter_highpass(frequency_to_cut, fs, order=4):
    nyq = 0.5 * fs
    low_frequency_normalized = frequency_to_cut / nyq
    sos = signal.butter(N=order, Wn=low_frequency_normalized, btype='highpass', output='sos')
    return sos

def butter_highpass_filter(data, lowcut, fs, order=4):
    sos = butter_highpass(lowcut, fs, order=order)
    y = signal.sosfilt(sos, data)
    return y

def applies_high_pass_for_dataset(dataset, frequency):
    dataset_to_return = []
    for example in dataset:
        example_formatted = []
        for vector_electrode in example:
            example_formatted.append(butter_highpass_filter(vector_electrode, 20, frequency))
        dataset_to_return.append(example_formatted)
    return dataset_to_return


def get_signals_and_draw(emg_data, load_myo_data):
    emg_examples_axes_swapped = np.swapaxes(emg_data, 1, 0)
    print(np.shape(emg_data))
    if load_myo_data:
        number_of_vector_per_example = 1825  # equivalent to 250ms with a sampling rate of 200Hz.
        emg_examples_swapped_and_filtered = []
        for data in emg_examples_axes_swapped:
            emg_examples_swapped_and_filtered.append(butter_highpass_filter(data, lowcut=20, fs=200))
    else:
        number_of_vector_per_example = 10818  # equivalent to 250ms with a sampling rate of 200Hz.
        emg_examples_swapped_and_filtered = []
        for data in emg_examples_axes_swapped:
            emg_examples_swapped_and_filtered.append(butter_highpass_filter(data, lowcut=20, fs=1000))

    emg_examples_filtered = np.swapaxes(emg_examples_swapped_and_filtered, 1, 0)
    example = []
    for i in range(len(emg_examples_filtered)):
        emg_vector = emg_examples_filtered[i]
        if len(example) == 0:
            example = emg_vector
        else:
            example = np.row_stack((example, emg_vector))
        if len(example) >= number_of_vector_per_example:
            example = example.transpose()
            draw_example(example=example, myo_data=load_myo_data)
            break


def draw_example(example, myo_data):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    print(np.shape(example))

    if myo_data:
        indexes = pd.Series(np.linspace(0, 10, len(example[0])), name="Secondes")
    else:
        indexes = pd.Series(np.linspace(0, 10, len(example[0])), name="Secondes")
    dictionary_labels = {}
    for i in range(len(example)):
        dictionary_labels.update({str(i+1): example[i, :]})

    data = pd.DataFrame(data=np.swapaxes(example, 1, 0),
                        columns=pd.Series(list(dictionary_labels.keys()), name="Channel"),
                        index=indexes)
    data = data.cumsum(axis=0).stack().reset_index(name="val")
    print(data.keys())
    sns.set(style="whitegrid", font_scale=4)
    g = sns.FacetGrid(data, col="Channel", col_wrap=1, height=3.5, despine=True, size=16)
    import matplotlib.ticker as ticker
    def signal_plot(x, y, **kwargs):
        ax = plt.gca()
        data = kwargs.pop("data")
        data.plot(x=x, y=y, sharex=True, sharey=True, ax=ax, linewidth=2.5, grid=False, **kwargs)
    g.map_dataframe(signal_plot, "Secondes", "val")
    g.set_ylabels("")
    g.set_yticklabels("")
    g.set_titles("")
    plt.show()


def load_data(get_train_data, load_myo_data, path, participant_to_get, cycle_to_get, gesture_to_get):
    examples_dataset = []
    labels_datasets = []
    if get_train_data:
        train_or_test_str = "train"
    else:
        train_or_test_str = "test"
    if load_myo_data:
        path_file_emg = path + '%d\\%s\\MyoArmband\\EMG\\Myo_EMG_gesture_%d_' % (participant_to_get,
                                                                                 train_or_test_str,
                                                                                 cycle_to_get)
    else:
        path_file_emg = path + '%d\\%s\\3dc\\EMG\\3dc_EMG_gesture_%d_' % (participant_to_get, train_or_test_str,
                                                                          cycle_to_get)

    examples_to_format = []
    for line in open(path_file_emg + '%d.txt' % gesture_to_get):
        # strip() remove the "\n" character, split separate the data in a list. np.float_ transform
        # each element of the list from a str to a float
        emg_signal = np.float32(line.strip().split(","))
        examples_to_format.append(emg_signal)
    get_signals_and_draw(examples_to_format, load_myo_data=load_myo_data)
    return examples_dataset, labels_datasets

if __name__ == '__main__':
    load_data(get_train_data=True, load_myo_data=False, path="../Dataset/Participant", participant_to_get=50,
                  cycle_to_get=0, gesture_to_get=0)
