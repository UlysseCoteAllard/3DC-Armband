import numpy as np
import scipy.signal as signal
import LDA.extract_features as extract_features

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


def applies_high_pass_for_dataset(dataset, frequency, cutoff=20):
    dataset_to_return = []
    for example in dataset:
        example_formatted = []
        for vector_electrode in example:
            example_formatted.append(butter_highpass_filter(vector_electrode, cutoff, frequency))
        dataset_to_return.append(example_formatted)
    return dataset_to_return


def format_examples(emg_examples, load_myo_data):
    # Start by applying the high pass filter
    # print(np.shape(emg_examples))
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
        return extract_features.format_dataset(dataset_examples_to_format, method='TD_stats')
    else:
        return extract_features.format_dataset(dataset_examples_to_format, method='TD_stats')


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
                examples_formatted = format_examples(examples_to_format, load_myo_data=load_myo_data)
                examples.extend(examples_formatted)
                labels.extend(np.ones(len(examples_formatted)) * gesture_index)
            examples_participant.append(examples)
            labels_participant.append(labels)
        examples_dataset.append(examples_participant)
        labels_datasets.append(labels_participant)
    return examples_dataset, labels_datasets


def read_data(path, load_myo_data):
    list_dataset_train_emg, list_labels_train_emg = get_data_and_process_from_file(get_train_data=True,
                                                                                   load_myo_data=load_myo_data,
                                                                                   path=path)
    if load_myo_data:
        np.save("../processed_datasets/LDA_Myo_train", (list_dataset_train_emg, list_labels_train_emg))
    else:
        np.save("../processed_datasets/LDA_3DC_train", (list_dataset_train_emg, list_labels_train_emg))

    list_dataset_test_emg, list_labels_test_emg = get_data_and_process_from_file(get_train_data=False,
                                                                                 load_myo_data=load_myo_data,
                                                                                 path=path)
    if load_myo_data:
        np.save("../processed_datasets/LDA_Myo_test", (list_dataset_test_emg, list_labels_test_emg))
    else:
        np.save("../processed_datasets/LDA_3DC_test", (list_dataset_test_emg, list_labels_test_emg))


if __name__ == '__main__':
    read_data("../Dataset/Participant", load_myo_data=True)
