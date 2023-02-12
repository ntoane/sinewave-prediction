import sine_wave as sine
import numpy as np
from numpy import array
import matplotlib.pyplot as plot
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import time as t

# split a univariate sequence into samples
def split_sequence(sequence, window_size):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + window_size
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Composite sine wave
sampling_rate = 10.0
sample_interval = 1/sampling_rate
time = np.arange(0, 100, sample_interval)

amplitude_y1 = sine.gen_sine(time, 1, 8, 0.25, 0)
amplitude_y2 = sine.gen_sine(time, 4, 3, 1, 0)
amplitude_y3 = sine.gen_sine(time, 8, 1, 0, 0)
amplitude_y4 = sine.gen_sine(time, 30, 6, 0.5, 0)
amplitude_y5 = sine.gen_sine(time, 50, 4, 0, 0)
composite_amplitude = np.add(
    np.add(
        np.add(
            np.add(amplitude_y1, 
            amplitude_y2), 
            amplitude_y3), 
            amplitude_y4), 
            amplitude_y5)

# Hyperparameters
WINDOW_SIZE = 5
BATCH_SIZE = 5
EPOCHS = 100
LR = 0.001
INPUT_SIZE = 1
HIDDEN_SIZE = 10
NUM_LAYERS = 1
NUM_CLASSES = 1

# Split sine wave into 80:20 train:test
amplitude_y = composite_amplitude.reshape(-1,1)
sc = MinMaxScaler()
amplitude_y = sc.fit_transform(amplitude_y)

train_data = amplitude_y[:80]
test_data = amplitude_y[80:]

# split into samples of window size of 5
trainX, trainY = split_sequence(train_data, WINDOW_SIZE)
testX, testY = split_sequence(test_data, WINDOW_SIZE)

# Convert to tensor
X_train = Variable(torch.Tensor(trainX))
y_train = Variable(torch.Tensor(trainY))
X_test = Variable(torch.Tensor(testX))
y_test = Variable(torch.Tensor(testY))

# Load the data in DataLoader for processing
train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# Define LSTM model
class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out


# Create LSTM model
lstm = LSTM(NUM_CLASSES, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)

# Define loss function and optimizer
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)

# Train and evaluate the model
train_loss = list()
test_loss = list()
before_time = t.time()
for epoch in range(1, EPOCHS+1):
    lstm.train()
    
    temp_loss = list()
    for X_train, y_train in train_dataloader:
        # Compute prediction error
        y_hat = lstm.forward(X_train)
        loss = loss_function(y_hat, y_train)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        temp_loss.append(loss.detach().cpu().numpy())
    
    train_loss.append(np.average(temp_loss))
    # if epoch % 10 == 0:
    #     print(f"Epoch {epoch} of {EPOCHS}, Training loss {train_loss[-1]:.6f}")

    # Evaluate the model on test set
    with torch.no_grad():
        # set validation mode
        lstm.eval()
        
        temp_test_loss = list()
        for X_test, y_test in test_dataloader:
            
            y_hat = lstm.forward(X_test)
            loss = loss_function(y_hat, y_test)

            temp_test_loss.append(loss.detach().cpu().numpy())
        
        test_loss.append(np.average(temp_test_loss))
        if epoch % 10 == 0:
            print(f"Epoch {epoch} of {EPOCHS}, Training loss: {train_loss[-1]:.6f}, Testing loss: {test_loss[-1]:.6f}")

train_time = t.time() - before_time
print(f"\nTraining time(s): {train_time}")
print(f"Average Training Loss: {np.average(train_loss):.6f}")
print(f"Average Testing Loss: {np.average(test_loss):.6f}")

# Plot the training and testing loss over epochs
plot.figure(figsize = (10, 6))
ep = list(range(0, EPOCHS))
plot.plot(ep, train_loss, label='train')
plot.plot(ep, test_loss, label='test')
plot.title('Training vs Testing Loss')
plot.xlabel('Epoch')
plot.ylabel('Average Loss')
plot.legend()
plot.show()
