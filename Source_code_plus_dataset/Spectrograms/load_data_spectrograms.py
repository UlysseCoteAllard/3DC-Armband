import numpy as np
import scipy.signal as signal
import Source_code_plus_dataset.Spectrograms.calculate_spectrograms as calculate_spectrograms

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


def format_examples(emg_examples, load_myo_data):
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
        return calculate_spectrograms.calculate_spectrogram_dataset(dataset_examples_to_format, frequency=200)
    else:
        return calculate_spectrograms.calculate_spectrogram_dataset(dataset_examples_to_format, frequency=1000)


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
            print("Cycle : ", cycle)
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
            print(np.shape(examples))
            examples_participant.append(examples)
            labels_participant.append(labels)
        examples_dataset.append(examples_participant)
        labels_datasets.append(labels_participant)
        if load_myo_data:
            np.save("../processed_datasets/spectrograms_processed/Spectrogram_MYO_"+train_or_test_str+"_participant_" +
                    str(participant), (examples_participant, labels_participant))
        else:
            np.save("../processed_datasets/spectrograms_processed/Spectrogram_3DC_"+train_or_test_str+"_participant_" +
                    str(participant), (examples_participant, labels_participant))
    return examples_dataset, labels_datasets


def read_data(path, load_myo_data):
    
    get_data_and_process_from_file(get_train_data=True, load_myo_data=load_myo_data, path=path)

    get_data_and_process_from_file(get_train_data=False, load_myo_data=load_myo_data, path=path)


def draw_example(example, myo_data):
    import matplotlib.pyplot as plt

    print(np.shape(example))

    def show_spectrogram(frequencies_samples, channels, spectrogram_of_vector):
        plt.rcParams.update({'font.size': 36})
        np.set_printoptions(suppress=True)
        print(spectrogram_of_vector)
        print(np.shape(channels))

        np.append(frequencies_samples, 510.)
        print(np.shape(frequencies_samples))

        # time_segment_sample = [0., 65.,  130., 195., 250.]
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        # print(time_segment_sample)
        print(frequencies_samples)
        plt.pcolormesh(channels, frequencies_samples, np.swapaxes(spectrogram_of_vector, 0, 1))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [ms]')
        plt.title("STFT")
        plt.subplots_adjust(left=0., bottom=0., right=1., top=1.)

    canals = []
    for electrode in example:
        if myo_data:
            spectrogram_of_vector, time_segment_sample, frequencies_samples = \
                calculate_spectrograms.calculate_spectrogram_vector(electrode, fs=200, npserseg=20, noverlap=10)
        else:
            spectrogram_of_vector, time_segment_sample, frequencies_samples = \
                calculate_spectrograms.calculate_spectrogram_vector(electrode, fs=1000, npserseg=100, noverlap=50)
        canals.append(np.swapaxes(spectrogram_of_vector, 0, 1))
    example_to_classify = np.swapaxes(canals, 0, 1)
    print(np.shape(example_to_classify))

    if myo_data:
        channels = range(9)
    else:
        channels = range(11)

    for time_example in example_to_classify:
        show_spectrogram(frequencies_samples, channels, time_example)
        plt.figure()
    plt.show()


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


def draw_examples_spectrogram(get_train_data, load_myo_data, path, participant_to_get, cycle_to_get, gesture_to_get):
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

if __name__ == '__main__':
    draw_examples_spectrogram(get_train_data=True, load_myo_data=True, path="../Dataset/Participant",
                              participant_to_get=12, cycle_to_get=3, gesture_to_get=6)
