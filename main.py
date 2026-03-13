
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import time
import random
import argparse

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Generate synthetic data
X = np.random.rand(1000, 784)
y = np.random.randint(0, 2, 1000)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a custom dataset class
class SyntheticDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# Create dataset instances
train_dataset = SyntheticDataset(X_train, y_train)
test_dataset = SyntheticDataset(X_test, y_test)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

# Define a high-performance modular neural architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 2)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.batch_norm3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm3(self.fc3(x)))
        x = self.fc4(x)
        return x

# Initialize the neural network, loss function, and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

# Initialize tensorboard writer
writer = SummaryWriter()

# Optimize the training loop for speed and accuracy
num_epochs = 100
cudnn.benchmark = True
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    writer.add_scalar("Loss/train", total_loss / len(train_loader), epoch)
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}')

    model.eval()
    with torch.no_grad():
        total_correct = 0
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
        accuracy = total_correct / len(test_loader.dataset)
        writer.add_scalar("Accuracy/test", accuracy, epoch)
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Accuracy: {accuracy:.4f}')

end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")

# Print success if the model is trained and accuracy is greater than 0
if model and accuracy > 0:
    print("Success")

# Save the trained model
torch.save(model.state_dict(), 'model.pth')

# Load the saved model
# model.load_state_dict(torch.load('model.pth'))

# Evaluate the model on the test set
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

accuracy = correct / len(test_loader.dataset)
print(f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {accuracy:.4f}')

# Define a function to get the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the number of parameters in the model
print(f"Number of parameters: {count_parameters(model)}")

# Define a function to plot the training and test loss and accuracy
import matplotlib.pyplot as plt

def plot_loss_and_accuracy(writer):
    loss_values = []
    accuracy_values = []
    for i in range(num_epochs):
        loss_values.append(writer.scalar_dict["Loss/train"][i])
        accuracy_values.append(writer.scalar_dict["Accuracy/test"][i])
    plt.plot(loss_values)
    plt.plot(accuracy_values)
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.show()

# Plot the training and test loss and accuracy
plot_loss_and_accuracy(writer)
