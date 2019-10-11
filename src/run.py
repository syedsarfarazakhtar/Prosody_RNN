import torch
from preprocessing.rose_accent import get_features
#from preprocessing.rose_phrase import get_features

import model
from torch import nn
import numpy as np
import pickle

# This function takes in the model and character as arguments and returns the next character prediction and hidden state
def predict(input):

    out, hidden = mymodel(input)
    return torch.max(out, dim=1)
    #return out.max(1, keepdim=True)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

hidden_dims = [25]#, 50, 100, 150, 200, 150, 300]
layers = [1, 2]#, 3, 4]
n_epochs = [2]
lrs = [0.0001]#, 0.0005, 0.001, 0.01]
results = {}
params = []
for i in range(len(hidden_dims)):
    for j in range(len(layers)):
        for k in range(len(n_epochs)):
            for l in range(len(lrs)):
                params.append([hidden_dims[i], layers[j], n_epochs[k], lrs[l]])

inputs, all_features = get_features()
samples, cols, features = inputs

for grid_search in params:

    hidden_dim = grid_search[0]
    n_layers = grid_search[1]
    n_epochs = grid_search[2]
    lr = grid_search[3]

    mymodel = model.Model(input_size=len(cols), output_size=2, hidden_dim=hidden_dim, n_layers=n_layers)

    labels = ["accented", "unaccented"]
    #labels = ["break", "nobreak"]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr)
    mymodel.to(device)

    training_set = []
    evaluation_set = []
    testing_set = []

    ind = 0
    for sample in samples:

        for i in sample:
            x = i
            X_train = x[0]
            X_train = np.asanyarray(X_train)
            y_train_str = np.asarray(x[1])
            if len(X_train) == 0:
                continue
            is_cuda = torch.cuda.is_available()

            if is_cuda:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            X_train = torch.from_numpy(X_train).float()
            X_train = X_train.unsqueeze(0)
            y_train = []
            for i in y_train_str:
                if i == labels[0]:
                    y_train.append([1, 0])
                else:
                    y_train.append([0, 1])

            y_train = torch.FloatTensor(y_train).float()
            y_train = y_train.unsqueeze(0)
            X_train.to(device)
            y_train.to(device)
            training_set.append([X_train, y_train])

    for sample in samples:
        for i in sample:
            x = i
            X_test = x[2]
            X_test = np.asanyarray(X_test)
            y_test_str = np.asarray(x[3])
            if len(X_test) == 0:
                continue
            is_cuda = torch.cuda.is_available()

            if is_cuda:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            X_test = torch.from_numpy(X_test).float()
            X_test = X_test.unsqueeze(0)
            y_test = []
            for i in y_test_str:
                if i == labels[0]:
                    y_test.append([1, 0])
                else:
                    y_test.append([0, 1])

            y_test = torch.FloatTensor(y_test).float()
            y_test = y_test.unsqueeze(0)
            X_test.to(device)
            y_test.to(device)
            testing_set.append([X_test, y_test])

    validation_set = testing_set[0:int(len(testing_set)/2)]
    testing_set = testing_set[int(len(testing_set)/2)::]
    print (len(training_set), len(validation_set), len(testing_set))

    for epoch in range(1, n_epochs + 1):

        if True:
            for i in training_set:
                x = i
                X_train = x[0]
                y_train = x[1]

                cross_entropy_target = []
                for i in y_train[0]:
                    cross_entropy_target.append(i[0])
                cross_entropy_target = torch.LongTensor(cross_entropy_target)
                # Training Run

                output, hidden = mymodel(X_train)
                loss = criterion(output, cross_entropy_target)

                loss.backward()  # Does backpropagation and calculates gradients
                optimizer.step()  # Updates the weights accordingly
                optimizer.zero_grad()  # Clears existing gradients

        if epoch % 1 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
            total_samples = 0.0
            total_correct = 0.0
            true_pos = 0.0
            false_neg = 0.0
            true_neg = 0.0
            false_pos = 0.0

            if True:
                for i in validation_set:
                    x = i
                    X_val = x[0]
                    Y_val = x[1]
                    O_val = predict(X_val)
                    correct_out = Y_val

                    predictions = []
                    for i in O_val:
                        predictions.append(i)
                    O_val = predictions[1]
                    le = O_val.size(0)
                    for k in range(le):
                        if int(O_val[k].item()) == int(correct_out[0][k][0].item()):
                            total_correct += 1
                            if int(correct_out[0][k][0].item()) == 1:
                                true_pos += 1
                        if int(correct_out[0][k][0].item()) == 0:
                                true_neg += 1
                        else:
                            if int(correct_out[0][k][0].item()) == 1:
                                false_neg += 1
                            if int(correct_out[0][k][0].item()) == 0:
                                false_pos += 1
                        total_samples += 1
            precision = true_pos*1.0/(true_pos+false_pos)
            recall = true_pos*1.0/(true_pos+false_neg)
            f1 = 2.0*(recall*precision)/(recall+precision)
            accuracy = total_correct*1.0/total_samples
            print("Total Samples: ", total_samples)
            print("Accuracy: ", accuracy)
            print("Precision: ", precision)
            print("Recall: ", recall)
            print("F1: ", f1)
            print()
            print(hidden_dim, n_layers, lr, epoch)
            print()
            tmp = ("validation", hidden_dim, n_layers, lr, epoch)
            results[tmp] = {}
            results[tmp]["Accuracy"] = accuracy
            results[tmp]["Precision"] = precision
            results[tmp]["Recall"] = recall
            results[tmp]["F1"] = f1

            if True:
                for i in testing_set:
                    x = i
                    X_val = x[0]
                    Y_val = x[1]
                    O_val = predict(X_val)
                    correct_out = Y_val

                    predictions = []
                    for i in O_val:
                        predictions.append(i)
                    O_val = predictions[1]
                    le = O_val.size(0)
                    for k in range(le):
                        if int(O_val[k].item()) == int(correct_out[0][k][0].item()):
                            total_correct += 1
                            if int(correct_out[0][k][0].item()) == 1:
                                true_pos += 1
                        if int(correct_out[0][k][0].item()) == 0:
                                true_neg += 1
                        else:
                            if int(correct_out[0][k][0].item()) == 1:
                                false_neg += 1
                            if int(correct_out[0][k][0].item()) == 0:
                                false_pos += 1
                        total_samples += 1
            precision = true_pos*1.0/(true_pos+false_pos)
            recall = true_pos*1.0/(true_pos+false_neg)
            f1 = 2.0*(recall*precision)/(recall+precision)
            accuracy = total_correct*1.0/total_samples
            print("Total Samples: ", total_samples)
            print("Test Accuracy: ", accuracy)
            print("Test Precision: ", precision)
            print("Test Recall: ", recall)
            print("Test F1: ", f1)
            print()
            print(hidden_dim, n_layers, lr, epoch)
            print()
            tmp = ("test", hidden_dim, n_layers, lr, epoch)
            results[tmp] = {}
            results[tmp]["Accuracy"] = accuracy
            results[tmp]["Precision"] = precision
            results[tmp]["Recall"] = recall
            results[tmp]["F1"] = f1

    for i in results:
        print(i)

with open('results.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)