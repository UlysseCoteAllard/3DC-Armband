import numpy as np
from RAW_emg import Raw_CNN_Myo, load_data_raw  # ,Raw_CNN_3DC
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import time
from scipy.stats import mode
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
    except FileNotFoundError:
        accuracy_for_all_participants = []

    number_of_re_run = 20 - len(accuracy_for_all_participants)
    for _ in range(number_of_re_run):
        accuracy_test = []
        for participant in range(len(labels_training)):
            X_train = []
            Y_train = []

            for cycle in range(nmbr_of_cycles_for_training+1):
                X_train.extend(examples_training[participant][cycle])
                Y_train.extend(labels_training[participant][cycle])
            X_train = np.expand_dims(X_train, axis=1)
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

            trainloader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True)
            validationloader = torch.utils.data.DataLoader(validation, batch_size=256, shuffle=True)

            cnn = Raw_CNN_Myo.Net(number_of_class=11).cuda()

            criterion = nn.CrossEntropyLoss(size_average=False)
            optimizer = optim.Adam(cnn.parameters(), lr=0.0404709)

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
            X_test = np.expand_dims(X_test, axis=1)
            test = TensorDataset(torch.from_numpy(np.array(X_test, dtype=np.float32)),
                                 torch.from_numpy(np.array(Y_test, dtype=np.int32)))

            test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)

            total = 0
            correct_prediction_test = 0
            for k, data_test_0 in enumerate(test_loader, 0):
                # get the inputs
                inputs_test, ground_truth_test = data_test_0
                inputs_test, ground_truth_test = Variable(inputs_test.cuda()), Variable(ground_truth_test.cuda())


                outputs_test_0 = cnn(inputs_test)
                _, predicted = torch.max(outputs_test_0.data, 1)
                correct_prediction_test += (mode(predicted.cpu().numpy())[0][0] ==
                                            ground_truth_test.data.cpu().numpy()).sum()
                total += ground_truth_test.size(0)
            print("ACCURACY TEST FINAL : %.3f %%" % (100 * float(correct_prediction_test) / float(total)))
            accuracy_test.append(100 * float(correct_prediction_test) / float(total))
            print("ACCURACY TEST RIGHT NOW : ", str(accuracy_test))
        accuracy_for_all_participants.append(accuracy_test)
        print("ACCURACY FOR ALL PARTICIPANTS RIGHT NOW : ", str(accuracy_for_all_participants))

        if training_with_myo:
            np.save("intermediate_results/Myo_results"+str(nmbr_of_cycles_for_training), accuracy_for_all_participants)
        else:
            np.save("intermediate_results/3DC_results"+str(nmbr_of_cycles_for_training), accuracy_for_all_participants)

    print("AVERAGE ACCURACY TEST 0 %.3f" % np.array(accuracy_for_all_participants).mean())
    return accuracy_for_all_participants


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


def calculate_fitness_CWT_3DC():
    # Comment between here

    load_data_raw.read_data("../Dataset/Participant", load_myo_data=False)

    # and here if the evaluation dataset was already processed and saved with "load_evaluation_dataset"
    import os

    print(os.listdir("../"))

    print("LOADING DATASET TRAINING...")
    datasets_training = np.load("../processed_datasets/RAW_3DC_train.npy")
    print("Finished loading dataset training...")
    print("LOADING DATASET TEST...")
    datasets_test = np.load("../processed_datasets/RAW_3DC_test.npy")
    print("Finished loading dataset test...")
    train_examples, train_labels = datasets_training
    test_examples, test_labels = datasets_test

    for cycle in range(0, 4):  # To get cycles: 1, 2, 3 and 4
        accuracies = calculate_fitness(train_examples, train_labels, test_examples, test_labels, cycle,
                                       training_with_myo=False)
        print("ACCURACIES for " + str(cycle+1) + " cycle(s) : " + str(accuracies))
        print("Average accuracies for " + str(cycle+1) + " cycle(s) : " + str(np.mean(accuracies)))
        with open("../results/results_RAW_3DC_last_cycle_training_" + str(cycle+1) + "_cycles.txt", "a") as myfile:
            myfile.write("RAW Best: \n\n")
            myfile.write(str(accuracies) + '\n')
            myfile.write(str(np.mean(accuracies)) + '\n')
            myfile.write('\n\n\n\n')


def calculate_fitness_CWT_Myo():
    # Comment between here

    load_data_raw.read_data("../Dataset/Participant", load_myo_data=True)

    # and here if the evaluation dataset was already processed and saved wit    h "load_evaluation_dataset"
    import os

    print(os.listdir("../"))
    print("LOADING DATASET TRAINING...")
    datasets_training = np.load("../processed_datasets/RAW_Myo_train.npy")
    print("Finished loading dataset training...")

    print("LOADING DATASET TEST...")
    datasets_test = np.load("../processed_datasets/RAW_Myo_test.npy")
    print("Finished loading dataset test...")

    train_examples, train_labels = datasets_training
    test_examples, test_labels = datasets_test
    #test_examples, test_labels = [], []
    for cycle in range(0, 4):  # To get cycles: 1, 2, 3 and 4
        accuracies = calculate_fitness(train_examples, train_labels, test_examples, test_labels, cycle,
                                       training_with_myo=True)
        print("ACCURACIES for " + str(cycle+1) + " cycle(s) : " + str(accuracies))
        print("Average accuracies for " + str(cycle+1) + " cycle(s) : " + str(np.mean(accuracies)))
        with open("../results/results_MYO_RAW_last_cycle_training_" + str(cycle+1) + "_cycles.txt", "a") as myfile:
            myfile.write("RAW Best: \n\n")
            myfile.write(str(accuracies) + '\n')
            myfile.write(str(np.mean(accuracies)) + '\n')
            myfile.write('\n\n\n\n')


if __name__ == '__main__':
    calculate_fitness_CWT_Myo()
    calculate_fitness_CWT_3DC()
