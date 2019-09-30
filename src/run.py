import torch
from preprocessing.rose_accent import get_features
import model
from torch import nn
import numpy as np

# This function takes in the model and character as arguments and returns the next character prediction and hidden state
def predict(input):

    out, hidden = mymodel(input)
    char_ind = torch.max(out, dim=1)
    #char_ind = torch.max(prob, dim=0)[1].item()
    return(char_ind)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

samples, cols = get_features()
mymodel = model.Model(input_size=len(cols), output_size=2, hidden_dim=50, n_layers=2)

n_epochs = 40
lr = 0.0001

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr)
mymodel.to(device)

for epoch in range(1, n_epochs + 1):

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
                if i == "accented":
                    y_train.append([1, 0])
                else:
                    y_train.append([0, 1])

            y_train = torch.FloatTensor(y_train).float()
            y_train = y_train.unsqueeze(0)
            X_train.to(device)
            y_train.to(device)

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


for sample in samples:
    for i in sample:
        x = i
        X_test = x[2]
        if len(X_test) == 0:
            continue
        X_test = torch.from_numpy(X_test).float()
        X_test = X_test.unsqueeze(0)
        X_test.to(device)

        print (predict(X_test))


