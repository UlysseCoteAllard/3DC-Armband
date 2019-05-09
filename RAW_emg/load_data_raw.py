import numpy as np
from scipy import signal

number_of_gestures = 11
number_of_cycles = 4


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

def format_examples(emg_examples, load_myo_data, label=None):
    emg_examples_axes_swapped = np.swapaxes(emg_examples, 1, 0)
    
    if load_myo_data:
        number_of_vector_per_example = 50  # equivalent to 250ms with a sampling rate of 200Hz.
        size_non_overlap = 5  # Original amount of overlap used with the Myo.
        emg_examples_swapped_and_filtered = []
        for five_seconds_data in emg_examples_axes_swapped:
            emg_examples_swapped_and_filtered.append(butter_highpass_filter(five_seconds_data, lowcut=20, fs=200))
    else:
        number_of_vector_per_example = 250  # equivalent to 250ms with a sampling rate of 200Hz.
        size_non_overlap = 25  # Overlap needed to obtain the same amount of examples as with the Myo that use an
        # overlap of 5
        emg_examples_swapped_and_filtered = []
        for five_seconds_data in emg_examples_axes_swapped:
            emg_examples_swapped_and_filtered.append(butter_highpass_filter(five_seconds_data, lowcut=20, fs=1000))
    
    dataset_examples_to_format = []
    example = []

    emg_examples_filtered = np.swapaxes(emg_examples_swapped_and_filtered, 1, 0)
    for emg_vector in emg_examples_filtered:
        if len(example) == 0:
            example = emg_vector
        else:
            example = np.row_stack((example, emg_vector))

        if len(example) >= number_of_vector_per_example:
            example = example.transpose()
            dataset_examples_to_format.append(example)
            example = example.transpose()
            example = example[size_non_overlap:]

    if load_myo_data:
        return dataset_examples_to_format
    else:
        return dataset_examples_to_format


def get_example_in_middle(emg_data, load_myo_data):
    emg_examples_axes_swapped = np.swapaxes(emg_data, 1, 0)

    if load_myo_data:
        number_of_vector_per_example = 50  # equivalent to 250ms with a sampling rate of 200Hz.
        emg_examples_swapped_and_filtered = []
        for five_seconds_data in emg_examples_axes_swapped:
            emg_examples_swapped_and_filtered.append(butter_highpass_filter(five_seconds_data, lowcut=20, fs=200))
    else:
        number_of_vector_per_example = 250  # equivalent to 250ms with a sampling rate of 200Hz.
        emg_examples_swapped_and_filtered = []
        for five_seconds_data in emg_examples_axes_swapped:
            emg_examples_swapped_and_filtered.append(butter_highpass_filter(five_seconds_data, lowcut=20, fs=1000))

    emg_examples_filtered = np.swapaxes(emg_examples_swapped_and_filtered, 1, 0)
    example = []
    print("ALLO : ", np.shape(emg_examples_filtered))

    for i in range(int(len(emg_examples_filtered)/2), len(emg_examples_filtered)):
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
        indexes = pd.Series(range(1, 5 * len(example[0]) + 1, 5), name="Time")
    else:
        indexes = pd.Series(range(1, len(example[0]) + 1), name="Time")
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

    def signal_plot(x, y, **kwargs):
        ax = plt.gca()
        data = kwargs.pop("data")
        data.plot(x=x, y=y, sharex=True, sharey=True, ax=ax, linewidth=10, grid=False, **kwargs)
    g.map_dataframe(signal_plot, "Time", "val")
    g.set_ylabels("")
    g.set_xlabels("")
    g.set_xticklabels("")
    g.set_yticklabels("")
    g.set_titles("")
    plt.show()

    if myo_data:
        frequency = 200
    else:
        frequency = 1000
    from scipy import fftpack
    X = fftpack.fft(example[0])
    freqs = fftpack.fftfreq(len(example[0])) * frequency

    fig, ax = plt.subplots()

    ax.stem(freqs, np.abs(X))
    ax.set_xlabel('Frequency in Hertz [Hz]')
    ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
    ax.set_xlim(-frequency / 2, frequency / 2)
    plt.show()


def get_data_and_process_from_file(get_train_data, load_myo_data, path):
    examples_dataset = []
    labels_datasets = []
    if get_train_data:
        train_or_test_str = "train"
    else:
        train_or_test_str = "test"
    for participant in range(1, 23):  # There is 22 participants total in the dataset
        print("Participant : ", participant)

        examples_participant = []
        labels_participant = []
        for cycle in range(4):  # There is 4 cycles both for the train and test set
            if load_myo_data:
                path_file_emg = path + '%d\\%s\\MyoArmband\\EMG\\Myo_EMG_gesture_%d_' % (participant, train_or_test_str,
                                                                                         cycle)
            else:
                path_file_emg = path + '%d\\%s\\3dc\\EMG\\3dc_EMG_gesture_%d_' % (participant, train_or_test_str, cycle)
            examples = []
            labels = []
            for gesture_index in range(number_of_gestures):
                examples_to_format = []
                for line in open(path_file_emg + '%d.txt' % gesture_index):
                    # strip() remove the "\n" character, split separate the data in a list. np.float_ transform
                    # each element of the list from a str to a float
                    emg_signal = np.float32(line.strip().split(","))
                    examples_to_format.append(emg_signal)
                examples_formatted = format_examples(examples_to_format, load_myo_data=load_myo_data,
                                                     label=gesture_index)
                examples.extend(examples_formatted)
                labels.extend(np.ones(len(examples_formatted)) * gesture_index)
            examples_participant.append(examples)
            labels_participant.append(labels)
        examples_dataset.append(examples_participant)
        labels_datasets.append(labels_participant)
    return examples_dataset, labels_datasets


def draw_examples(get_train_data, load_myo_data, path, participant_to_get, cycle_to_get, gesture_to_get):
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
    get_example_in_middle(examples_to_format, load_myo_data=load_myo_data)
    return examples_dataset, labels_datasets


def read_data(path, load_myo_data):
    list_dataset_train_emg, list_labels_train_emg = get_data_and_process_from_file(get_train_data=True,
                                                                                   load_myo_data=load_myo_data,
                                                                                   path=path)
    if load_myo_data:
        np.save("../processed_datasets/RAW_Myo_train", (list_dataset_train_emg, list_labels_train_emg))
    else:
        np.save("../processed_datasets/RAW_3DC_train", (list_dataset_train_emg, list_labels_train_emg))

    list_dataset_test_emg, list_labels_test_emg = get_data_and_process_from_file(get_train_data=False,
                                                                                 load_myo_data=load_myo_data,
                                                                                 path=path)
    if load_myo_data:
        np.save("../processed_datasets/RAW_Myo_test", (list_dataset_test_emg, list_labels_test_emg))
    else:
        np.save("../processed_datasets/RAW_3DC_test", (list_dataset_test_emg, list_labels_test_emg))


if __name__ == '__main__':
    draw_examples(get_train_data=True, load_myo_data=False, path="../Dataset/Participant", participant_to_get=21,
                  cycle_to_get=2, gesture_to_get=9)
