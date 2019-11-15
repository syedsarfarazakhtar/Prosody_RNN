import pickle
import torch
#from preprocessing.rose_accent import get_features
#from preprocessing.rose_phrase import get_features
from preprocessing.accent_ensemble import get_features
import random
import model
from torch import nn
import numpy as np
import pickle

# This function takes in the model and character as arguments and returns the next character prediction and hidden state
def predict(input):

    out, hidden = mymodel(input)
    return torch.max(out, dim=1)
    #return out.max(1, keepdim=True)

task = "accent"
#task = "break"
architecture = "GRU"
architecture = "ensemble"
experiment = "bert"
labels = []
if task == "accent":
    labels = ["accented", "unaccented"]
else:
    labels = ["break", "nobreak"]

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

hidden_dims = [75]
layers = [3]
n_epochs = [20]
lrs = [0.00005]
results_val = {}
results_test = {}

results_meta = ["hidden_dim", "n_layers", "learning_rate", "epoch"]

params = []
for i in range(len(hidden_dims)):
    for j in range(len(layers)):
        for k in range(len(n_epochs)):
            for l in range(len(lrs)):
                params.append([hidden_dims[i], layers[j], n_epochs[k], lrs[l]])

samples, cols, features, all_features = get_features()
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def evaluate_model(mymodel, data_set, results_val):
    if True:
        if True:
            if True:
                total_samples = 0.0
                total_correct = 0.0
                true_pos = 0.0
                false_neg = 0.0
                true_neg = 0.0
                false_pos = 0.0
                with torch.no_grad():
                    mymodel.eval()
                   
                    for i in data_set:
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
                    precision = true_pos*1.0/(true_pos+false_pos+0.0001)
                    recall = true_pos*1.0/(true_pos+false_neg+0.0001)
                    f1 = 2.0*(recall*precision)/(recall+precision+0.0001)
                    accuracy = total_correct*1.0/total_samples
                    #if f1<prev_val:
                    #    #print("Early Stopping")
                    #    stopping_flag = 1
                    #prev_val=f1
                    print()
                    print()
                    print(results_meta) 
                    print(hidden_dim, n_layers, lr, epoch)
                    print(current_features_str)
                    tmp = (hidden_dim, n_layers, lr, epoch)
                    print("Total Samples: ", total_samples)
                    print("True Positive: ", true_pos)
                    print("False Positive: ", false_pos)
                    print("True Negative: ", true_neg)
                    print("False Negative: ", false_neg)
                    print("Accuracy: ", accuracy)
                    print("Precision: ", precision)
                    print("Recall: ", recall)
                    print("F1: ", f1)
                    print()
                    results_val[tmp] = {}
                    results_val[tmp]["Accuracy"] = accuracy
                    results_val[tmp]["Precision"] = precision
                    results_val[tmp]["Recall"] = recall
                    results_val[tmp]["F1"] = f1
                    results_val[tmp]["Features"] = current_features_str
                total_samples = 0.0
                total_correct = 0.0
                true_pos = 0.0
                false_neg = 0.0
                true_neg = 0.0
                false_pos = 0.0
    return results_val
   

all_data = []
if True:
    training_set = []
    evaluation_set = []
    testing_set = []
    current_features = []
    ind = 0
    for sample in samples:
        training_set = []
        evaluation_set = []
        testing_set = []
        current_features.append(features[ind])
        current_features_str = ""
        for i in current_features[0]:
            current_features_str += i
            current_features_str +=  ","
        ind+=1
        prev_val = -1
        for i in sample:
            x = i
            X_train = x[0]
            X_train = np.asanyarray(X_train)
            y_train_str = np.asarray(x[1])
            if len(X_train) == 0:
                continue

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

        for i in sample:
            x = i
            X_test = x[2]
            X_test = np.asanyarray(X_test)
            y_test_str = np.asarray(x[3])
            if len(X_test) == 0:
                continue

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

        #random.shuffle(training_set)
        #random.shuffle(testing_set)
        validation_set = testing_set[0:int(len(testing_set)/2)]
        testing_set = testing_set[int(len(testing_set)/2)::]
        
        print (len(training_set), len(validation_set), len(testing_set))
        all_data.append([training_set, validation_set, testing_set, current_features])



for sample in all_data:
    for grid_search in params:

        hidden_dim = grid_search[0]
        n_layers = grid_search[1]
        n_epochs = grid_search[2]
        lr = grid_search[3]

        training_set = []
        evaluation_set = []
        testing_set = []
        current_features = []
        ind = 0
        output_size = 2
        mymodel = model.Model(input_size=len(cols[ind]), output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers)

        criterion = nn.CrossEntropyLoss()
        #criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr)
        mymodel.to(device)

        training_set = sample[0]
        validation_set = sample[1]
        testing_set = sample[2]
        current_features_str = sample[3]

        stopping_flag = 0
        for epoch in range(1, n_epochs + 1):

            if True:
                c=0
                for i in training_set:
                    x = i
                    X_train = x[0]
                    y_train = x[1]

                    cross_entropy_target = []
                    for i in y_train[0]:
                        cross_entropy_target.append(i[0])
                    cross_entropy_target = torch.LongTensor(cross_entropy_target)
                    # Training Run
                    try:
                        mymodel.train()
                        optimizer.zero_grad()  # Clears existing gradients
                        output, hidden = mymodel(X_train)
                        #output = output.squeeze(1)
                        loss = criterion(output, cross_entropy_target)
                        loss.backward()  # Does backpropagation and calculates gradients
                        optimizer.step()  # Updates the weights accordingly
                    except:
                        c+=1
                print("Bad Data:",c)

            if epoch % 1 == 0:
                print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
                print("Loss: {:.4f}".format(loss.item()))
                total_samples = 0.0
                total_correct = 0.0
                true_pos = 0.0
                false_neg = 0.0
                true_neg = 0.0
                false_pos = 0.0
                results_val = evaluate_model(mymodel, validation_set, results_val)
                results_test = evaluate_model(mymodel, testing_set, results_test)
            
            if stopping_flag:
                pass
                #print("Model not improving... Stopping !")
                #break

        del mymodel

results = {}
results["val"] = results_val
results["test"] = results_test
results["meta"] = results_meta

with open('results_'+task+'.pickle_'+architecture+"_"+experiment, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
