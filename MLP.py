# Imports
import numpy as np
from numpy import array
import matplotlib.pyplot as plot
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import time as t
import sine_wave as sine


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

# Create a sine wave
sampling_rate = 10.0
sample_interval = 1/sampling_rate
time = np.arange(0, 10, sample_interval)

frequency = 2
amplitude = 1

# Generate amplitude(y)
amplitude_y = sine.gen_sine(time, frequency, amplitude)

# Hyperparameters
HIDDEN_SIZE = 20
LR = 0.0001
EPOCHS = 200
BATCH_SIZE = 5
WINDOW_SIZE = 5
# Split sine wave into 80:20 train:test
train_data = amplitude_y[:80]
test_data = amplitude_y[80:]

# split into samples of window size of 5
X_train, y_train = split_sequence(train_data, WINDOW_SIZE)
X_test, y_test = split_sequence(test_data, WINDOW_SIZE)

# Convert to tensor for processing
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)

# Load the data in DataLoader for processing
train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# Define model with default hidden size of 10 neurons
class SineMLP(nn.Module):
    def __init__(self, HIDDEN_SIZE = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_features=5, out_features=HIDDEN_SIZE)
        self.output = nn.Linear(in_features=HIDDEN_SIZE, out_features=1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.output(x))
        x = self.output(x)
        return x

# Create model
sine_model = SineMLP(HIDDEN_SIZE)

# Define loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(sine_model.parameters(), lr=LR)

train_loss = list()
test_loss = list()
before_time = t.time()
for epoch in range(1, EPOCHS+1):
    sine_model.train()
    
    temp_loss = list()
    for X_train, y_train in train_dataloader:
        # Compute prediction error
        y_hat = sine_model.forward(X_train)
        y_train = y_train.reshape(-1, 1)
        loss = loss_function(y_hat, y_train)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        temp_loss.append(loss.detach().cpu().numpy())
    
    train_loss.append(np.average(temp_loss))

    # Evaluate the model on test set
    with torch.no_grad():
        # set validation mode
        sine_model.eval()
        
        temp_test_loss = list()
        for X_test, y_test in test_dataloader:
            
            y_hat = sine_model.forward(X_test)
            y_test = y_test.reshape(-1, 1)
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
