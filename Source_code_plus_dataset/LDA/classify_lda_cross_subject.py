import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

from Source_code_plus_dataset.LDA import load_data_lda


def scramble(examples, labels):
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)
    new_labels = []
    new_examples = []

    for i in random_vec:
        new_labels.append(labels[i])
        new_examples.append(examples[i])

    return new_examples, new_labels


def calculate_fitness_LDA_for_a_specific_number_of_cycles(examples_training, labels_training, examples_tests,
                                                          labels_tests,
                                                          nmbr_of_cycles_for_training):
    accuracy_for_all_participants = []
    predictions_for_all_participants = []
    ground_truth_for_all_participants = []
    for participant_to_test_on in range(len(labels_training)):
        X_train = []
        Y_train = []
        for participant in range(len(labels_training)):
            if participant_to_test_on != participant:
                for cycle in range(nmbr_of_cycles_for_training + 1):  # Index start at 0
                    X_train.extend(examples_training[participant][cycle])
                    Y_train.extend(labels_training[participant][cycle])
        X_train_scrambled, Y_train_scrambled = scramble(X_train, Y_train)
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_scrambled, Y_train_scrambled)

        X_test = []
        Y_test = []
        for cycle in range(0, 4):  # We necessarily have four cycles of test from the data
            X_test.extend(examples_tests[participant_to_test_on][cycle])
            Y_test.extend(labels_tests[participant_to_test_on][cycle])

        y_pred = lda.predict(X_test)
        accuracy = accuracy_score(y_pred, Y_test)
        print("ACCURACY FOR PARTICIPANT : ", participant_to_test_on, " IS : ", accuracy)
        predictions_for_all_participants.append(np.array(y_pred))
        ground_truth_for_all_participants.append(np.array(Y_test))
        accuracy_for_all_participants.append(accuracy)
    return accuracy_for_all_participants, predictions_for_all_participants, ground_truth_for_all_participants


def calculate_fitness_3DC():
    # Comment between here
    '''
    load_data_lda.read_data("../Dataset/Participant", load_myo_data=False)
    '''
    # and here if the evaluation dataset was already processed and saved with "load_evaluation_dataset"
    import os

    print(os.listdir("../"))

    print("LOADING DATASET TRAINING...")
    datasets_training = np.load("../processed_datasets/LDA_3DC_train.npy")
    print("Finished loading dataset training...")

    print("LOADING DATASET TEST...")
    datasets_test = np.load("../processed_datasets/LDA_3DC_test.npy")
    print("Finished loading dataset test...")

    train_examples, train_labels = datasets_training
    test_examples, test_labels = datasets_test

    for cycle in range(0, 4):  # To get cycles: 0, 1, 2 and 3
        accuracies, predictions_for_all_participants, ground_truth_for_all_participants = \
            calculate_fitness_LDA_for_a_specific_number_of_cycles(train_examples, train_labels, test_examples,
                                                                  test_labels, cycle)
        print("ACCURACIES for " + str(cycle + 1) + " cycle(s) : " + str(accuracies))
        print("Average accuracies for " + str(cycle + 1) + " cycle(s) : " + str(np.mean(accuracies)))
        with open("../results/results_LDA_cross_subject_3DC_" + str(cycle + 1) + "_cycles.txt", "a") as myfile:
            myfile.write("LDA Best: \n\n")
            myfile.write(str(accuracies) + '\n')
            myfile.write(str(np.mean(accuracies)) + '\n')
            myfile.write('\n\n\n\n')

        np.save("../results/pedictions_LDA_cross_subject_3DC_" + str(cycle + 1) + "_cycles.npy",
                np.array(predictions_for_all_participants))
        np.save("../results/ground_truth_LDA_cross_subject_3DC_" + str(cycle + 1) + "_cycles.npy",
                np.array(ground_truth_for_all_participants))


def calculate_fitness_Myo():
    # Comment between here
    '''
    load_data_lda.read_data("../Dataset/Participant", load_myo_data=True)
    '''
    # and here if the evaluation dataset was already processed and saved wit    h "load_evaluation_dataset"
    import os

    print(os.listdir("../"))
    print("LOADING DATASET TRAINING...")
    datasets_training = np.load("../processed_datasets/LDA_Myo_train.npy")
    print("Finished loading dataset training...")

    print("LOADING DATASET TEST...")
    datasets_test = np.load("../processed_datasets/LDA_Myo_test.npy")
    print("Finished loading dataset test...")

    train_examples, train_labels = datasets_training
    test_examples, test_labels = datasets_test
    for cycle in range(0, 4):  # To get cycles: 0, 1, 2 and 3
        accuracies, predictions_for_all_participants, ground_truth_for_all_participants = \
            calculate_fitness_LDA_for_a_specific_number_of_cycles(train_examples, train_labels, test_examples,
                                                                  test_labels, cycle)
        print("ACCURACIES for " + str(cycle + 1) + " cycle(s) : " + str(accuracies))
        print("Average accuracies for " + str(cycle + 1) + " cycle(s) : " + str(np.mean(accuracies)))
        with open("../results/results_MYO_cross_subjects_LDA_" + str(cycle + 1) + "_cycles.txt", "a") as myfile:
            myfile.write("LDA Best: \n\n")
            myfile.write(str(accuracies) + '\n')
            myfile.write(str(np.mean(accuracies)) + '\n')
            myfile.write('\n\n\n\n')
        np.save("../results/predictions_MYO_cross_subjects_LDA_" + str(cycle + 1) + "_cycles.npy",
                predictions_for_all_participants)
        np.save("../results/ground_truth_MYO_cross_subjects_LDA_" + str(cycle + 1) + "_cycles.npy",
                ground_truth_for_all_participants)


if __name__ == '__main__':
    calculate_fitness_3DC()
    calculate_fitness_Myo()
