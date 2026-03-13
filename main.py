
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
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

X = np.random.rand(1000, 784)
y = np.random.randint(0, 2, 1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class SyntheticDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

train_dataset = SyntheticDataset(X_train, y_train)
test_dataset = SyntheticDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

writer = SummaryWriter()

num_epochs = 100
cudnn.benchmark = True
start_time = time.time()
train_loss_values = []
test_accuracy_values = []

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
    train_loss_values.append(total_loss / len(train_loader))
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
        test_accuracy_values.append(accuracy)
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Accuracy: {accuracy:.4f}')

end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")

if model and accuracy > 0:
    print("Success")

torch.save(model.state_dict(), 'model.pth')

model.load_state_dict(torch.load('model.pth'))

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of parameters: {count_parameters(model)}")

plt.plot(train_loss_values)
plt.plot(test_accuracy_values)
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.show()

class ModularNeuralNetwork(nn.Module):
    def __init__(self):
        super(ModularNeuralNetwork, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2)
        )
        self.block2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2)
        )
        self.block3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        self.fc4 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fc4(x)
        return x

modular_model = ModularNeuralNetwork().to(device)
modular_criterion = nn.CrossEntropyLoss()
modular_optimizer = optim.Adam(modular_model.parameters(), lr=0.001, weight_decay=0.001)

modular_writer = SummaryWriter()

modular_num_epochs = 100
modular_cudnn_benchmark = True
modular_start_time = time.time()
modular_train_loss_values = []
modular_test_accuracy_values = []

for epoch in range(modular_num_epochs):
    modular_model.train()
    modular_total_loss = 0
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        modular_optimizer.zero_grad()
        outputs = modular_model(inputs)
        loss = modular_criterion(outputs, labels)
        loss.backward()
        modular_optimizer.step()
        modular_total_loss += loss.item()
    modular_writer.add_scalar("Loss/train", modular_total_loss / len(train_loader), epoch)
    modular_train_loss_values.append(modular_total_loss / len(train_loader))
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {modular_total_loss / len(train_loader):.4f}')

    modular_model.eval()
    with torch.no_grad():
        modular_total_correct = 0
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = modular_model(inputs)
            _, predicted = torch.max(outputs, 1)
            modular_total_correct += (predicted == labels).sum().item()
        modular_accuracy = modular_total_correct / len(test_loader.dataset)
        modular_writer.add_scalar("Accuracy/test", modular_accuracy, epoch)
        modular_test_accuracy_values.append(modular_accuracy)
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Accuracy: {modular_accuracy:.4f}')

modular_end_time = time.time()
print(f"Modular Training time: {modular_end_time - modular_start_time} seconds")

if modular_model and modular_accuracy > 0:
    print("Modular Success")

torch.save(modular_model.state_dict(), 'modular_model.pth')

modular_model.load_state_dict(torch.load('modular_model.pth'))

modular_model.eval()
modular_test_loss = 0
modular_correct = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = modular_model(inputs)
        loss = modular_criterion(outputs, labels)
        modular_test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        modular_correct += (predicted == labels).sum().item()

modular_accuracy = modular_correct / len(test_loader.dataset)
print(f'Modular Test Loss: {modular_test_loss / len(test_loader):.4f}, Modular Test Accuracy: {modular_accuracy:.4f}')

def modular_count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Modular Number of parameters: {modular_count_parameters(modular_model)}")

plt.plot(modular_train_loss_values)
plt.plot(modular_test_accuracy_values)
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.show()
