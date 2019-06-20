import numpy as np
from Source_code_plus_dataset.Spectrograms import Spectrogramm_CNN_3DC, load_data_spectrograms, Spectrogram_CNN_Myo
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import time
import copy


def confusion_matrix(pred, Y, number_class=7):
    confusion_matrice = []
    for x in range(0, number_class):
        vector = []
        for y in range(0, number_class):
            vector.append(0)
        confusion_matrice.append(vector)
    for prediction, real_value in zip(pred, Y):
        prediction = int(prediction)
        real_value = int(real_value)
        confusion_matrice[prediction][real_value] += 1
    return np.array(confusion_matrice)


def scramble(examples, labels):
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)
    new_labels = []
    new_examples = []

    for i in random_vec:
        new_labels.append(labels[i])
        new_examples.append(examples[i])
    return new_examples, new_labels


def calculate_fitness(examples_training, labels_training, examples_tests, labels_tests, nmbr_of_cycles_for_training,
                      training_with_myo):
    try:
        if training_with_myo:
            accuracy_for_all_participants = np.load("intermediate_results/Myo_results" +
                                                    str(nmbr_of_cycles_for_training) + ".npy")
            accuracy_for_all_participants = accuracy_for_all_participants.tolist()
        else:
            accuracy_for_all_participants = np.load("intermediate_results/3DC_results" +
                                                    str(nmbr_of_cycles_for_training) + ".npy")
            accuracy_for_all_participants = accuracy_for_all_participants.tolist()
        print(accuracy_for_all_participants)
    except FileNotFoundError:
        accuracy_for_all_participants = []

    number_of_re_run = 20 - len(accuracy_for_all_participants)
    predictions_for_all_participants = []
    ground_truth_for_all_participants = []
    for _ in range(number_of_re_run):
        accuracy_test = []
        predictions_participant = []
        ground_truth_participant = []
        for participant in range(len(labels_training)):
            X_train = []
            Y_train = []

            for cycle in range(nmbr_of_cycles_for_training+1):
                X_train.extend(examples_training[participant][cycle])
                Y_train.extend(labels_training[participant][cycle])
            print(np.shape(X_train))
            X_train_scrambled, Y_train_scrambled = scramble(X_train, Y_train)
            valid_examples = X_train_scrambled[0:int(len(X_train_scrambled) * 0.1)]
            labels_valid = Y_train_scrambled[0:int(len(Y_train_scrambled) * 0.1)]

            X_fine_tune = X_train_scrambled[int(len(X_train_scrambled) * 0.1):]
            Y_fine_tune = Y_train_scrambled[int(len(Y_train_scrambled) * 0.1):]

            print(torch.from_numpy(np.array(Y_fine_tune, dtype=np.int64)).size(0))
            print(np.shape(np.array(X_fine_tune, dtype=np.float32)))
            train = TensorDataset(torch.from_numpy(np.array(X_fine_tune, dtype=np.float32)),
                                  torch.from_numpy(np.array(Y_fine_tune, dtype=np.int64)))
            validation = TensorDataset(torch.from_numpy(np.array(valid_examples, dtype=np.float32)),
                                       torch.from_numpy(np.array(labels_valid, dtype=np.int64)))

            trainloader = torch.utils.data.DataLoader(train, batch_size=512, shuffle=True)
            validationloader = torch.utils.data.DataLoader(validation, batch_size=512, shuffle=True)

            if training_with_myo:
                cnn = Spectrogram_CNN_Myo.Net(number_of_class=11).cuda()
            else:
                cnn = Spectrogramm_CNN_3DC.Net(number_of_class=11).cuda()

            criterion = nn.CrossEntropyLoss(size_average=False)
            optimizer = optim.Adam(cnn.parameters(), lr=0.00681292)

            precision = 1e-8
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                             verbose=True, eps=precision)

            cnn = train_model(cnn, criterion, optimizer, scheduler,
                              dataloaders={"train": trainloader, "val": validationloader}, precision=precision)

            cnn.eval()

            X_test = []
            Y_test = []
            for cycle in range(0, 4):  # We necessarily have four cycles of test from the data
                X_test.extend(examples_tests[participant][cycle])
                Y_test.extend(labels_tests[participant][cycle])
            test = TensorDataset(torch.from_numpy(np.array(X_test, dtype=np.float32)),
                                 torch.from_numpy(np.array(Y_test, dtype=np.int64)))

            test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)

            total = 0
            correct_prediction_test = 0
            predictions = []
            ground_truths = []
            for k, data_test_0 in enumerate(test_loader, 0):
                # get the inputs
                inputs_test, ground_truth_test = data_test_0
                inputs_test, ground_truth_test = Variable(inputs_test.cuda()), Variable(ground_truth_test.cuda())

                outputs_test_0 = cnn(inputs_test)
                _, predicted = torch.max(outputs_test_0.data, 1)
                correct_prediction_test += torch.sum(predicted == ground_truth_test.data)
                predictions.extend(predicted.cpu().numpy())
                ground_truths.extend(ground_truth_test.data.cpu().numpy())
                total += ground_truth_test.size(0)
            predictions_participant.append(predictions)
            ground_truth_participant.append(ground_truths)
            accuracy = float(correct_prediction_test.item()/total)
            print("ACCURACY TEST FINAL : %.3f %%" % (100 * accuracy))
            accuracy_test.append(100 * accuracy)
            print("ACCURACY TEST RIGHT NOW : ", str(accuracy_test))
        accuracy_for_all_participants.append(accuracy_test)
        print("ACCURACY FOR ALL PARTICIPANTS RIGHT NOW : ", str(accuracy_for_all_participants))
        predictions_for_all_participants.append(predictions_participant)
        ground_truth_for_all_participants.append(ground_truth_participant)
        if training_with_myo:
            np.save("intermediate_results/Myo_results"+str(nmbr_of_cycles_for_training), accuracy_for_all_participants)
        else:
            np.save("intermediate_results/3DC_results"+str(nmbr_of_cycles_for_training), accuracy_for_all_participants)

    print("AVERAGE ACCURACY TEST 0 %.3f" % np.array(accuracy_for_all_participants).mean())
    return accuracy_for_all_participants, predictions_for_all_participants, ground_truth_for_all_participants


def train_model(cnn, criterion, optimizer, scheduler, dataloaders, num_epochs=500, precision=1e-8):
    since = time.time()

    best_loss = float('inf')

    patience = 30
    patience_increase = 10

    best_weights = copy.deepcopy(cnn.state_dict())
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                cnn.train(True)  # Set model to training mode
            else:
                cnn.train(False)  # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0
            total = 0

            for i, data in enumerate(dataloaders[phase], 0):
                # get the inputs
                inputs, labels = data

                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                # zero the parameter gradients
                optimizer.zero_grad()
                if phase == 'train':
                    cnn.train()
                    # forward
                    outputs = cnn(inputs)
                    _, predictions = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()

                else:
                    cnn.eval()
                    # forward
                    outputs = cnn(inputs)
                    _, predictions = torch.max(outputs.data, 1)

                    loss = criterion(outputs, labels)
                    loss = loss.item()

                # statistics
                running_loss += loss
                running_corrects += torch.sum(predictions == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.item() / total
            print('{} Loss: {:.8f} Acc: {:.8}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    best_weights = copy.deepcopy(cnn.state_dict())
                    patience = patience_increase + epoch
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience:
            break
    print()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    cnn.load_state_dict(copy.deepcopy(best_weights))
    cnn.eval()
    return cnn


def load_data(get_myo, train_or_test):
    if get_myo:
        myo_or_3dc = "MYO"
    else:
        myo_or_3dc = "3DC"

    print("LOADING DATASET " + str(train_or_test) + "...")
    test_examples = []
    test_labels = []
    # Get the data for the test
    for participant in range(1, 23):
        print("Participant : ", participant)
        path = "../processed_datasets/spectrograms_processed/Spectrogram_" + myo_or_3dc + "_" + train_or_test + "_" + str(
            participant) + ".npy"

        participant_dataset_test = np.load(path)
        test_examples_participant, test_labels_participant = participant_dataset_test
        test_examples.append(test_examples_participant)
        test_labels.append(test_labels_participant)
    print("Finished loading dataset " + str(train_or_test) + "...")

    return test_examples, test_labels


def calculate_fitness_3DC():
    # Comment between here

    load_data_spectrograms.read_data("../Dataset/Participant", load_myo_data=False)

    # and here if the evaluation dataset was already processed and saved with "load_evaluation_dataset"
    import os

    print(os.listdir("../"))

    train_examples, train_labels = load_data(get_myo=False, train_or_test="train_participant")
    test_examples, test_labels = load_data(get_myo=False, train_or_test="test_participant")

    print(np.shape(train_examples[0][0]))
    for cycle in range(0, 4):  # To get cycles: 1, 2, 3 and 4
        accuracies, predictions_for_all_participants, ground_truth_for_all_participants = calculate_fitness(train_examples, train_labels, test_examples, test_labels, cycle, training_with_myo=False)
        print("ACCURACIES for " + str(cycle+1) + " cycle(s) : " + str(accuracies))
        print("Average accuracies for " + str(cycle+1) + " cycle(s) : " + str(np.mean(accuracies)))
        with open("../results/results_3DC_Spectrograms_" + str(cycle+1) + "_cycles.txt", "a") as myfile:
            myfile.write("SPECTROGRAMS Best: \n\n")
            myfile.write(str(accuracies) + '\n')
            myfile.write(str(np.mean(accuracies)) + '\n')
            myfile.write('\n\n\n\n')

        np.save("../results/predictions_3DC_Spectrograms_" + str(cycle+1) + "_cycles.npy",
                np.array(predictions_for_all_participants))
        np.save("../results/groundTruth_3DC_Spectrograms_" + str(cycle+1) + "_cycles.npy",
                np.array(ground_truth_for_all_participants))


def calculate_fitness_Myo():
    # Comment between here

    load_data_spectrograms.read_data("../Dataset/Participant", load_myo_data=True)

    # and here if the evaluation dataset was already processed and saved wit    h "load_evaluation_dataset"
    import os

    print(os.listdir("../"))

    train_examples, train_labels = load_data(get_myo=True, train_or_test="train_participant")
    test_examples, test_labels = load_data(get_myo=True, train_or_test="test_participant")
    print(np.shape(train_examples[0][0]))
    for cycle in range(0, 4):  # To get cycles: 1, 2, 3 and 4
        accuracies, predictions_for_all_participants, ground_truth_for_all_participants = calculate_fitness(train_examples, train_labels, test_examples, test_labels, cycle, training_with_myo=True)
        print("ACCURACIES for " + str(cycle+1) + " cycle(s) : " + str(accuracies))
        print("Average accuracies for " + str(cycle+1) + " cycle(s) : " + str(np.mean(accuracies)))
        with open("../results/results_MYO_Spectrograms_" + str(cycle+1) + "_cycles.txt", "a") as myfile:
            myfile.write("SPECTROGRAMS Best: \n\n")
            myfile.write(str(accuracies) + '\n')
            myfile.write(str(np.mean(accuracies)) + '\n')
            myfile.write('\n\n\n\n')

        np.save("../results/predictions_MYO_Spectrograms_" + str(cycle + 1) + "_cycles.npy",
                np.array(predictions_for_all_participants))
        np.save("../results/groundTruth_MYO_Spectrograms_" + str(cycle + 1) + "_cycles.npy",
                np.array(ground_truth_for_all_participants))

if __name__ == '__main__':
    calculate_fitness_Myo()
    calculate_fitness_3DC()

