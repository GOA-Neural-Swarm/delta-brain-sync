
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

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

class ModularNeuralNetwork(nn.Module):
    def __init__(self):
        super(ModularNeuralNetwork, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(784, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.2)
        )
        self.block2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2)
        )
        self.block3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        self.fc4 = nn.Linear(512, 2)
        self.attention = nn.MultiHeadAttention(512, 8)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x, _ = self.attention(x, x)
        x = self.fc4(x)
        return x

class ModularNeuralNetworkImproved(nn.Module):
    def __init__(self):
        super(ModularNeuralNetworkImproved, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(784, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.2)
        )
        self.block2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2)
        )
        self.block3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        self.fc4 = nn.Linear(512, 2)
        self.attention = nn.MultiHeadAttention(512, 8)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x, _ = self.attention(x, x)
        x = self.fc4(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ModularNeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

writer = SummaryWriter()

num_epochs = 200
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

model_improved = ModularNeuralNetworkImproved().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_improved.parameters(), lr=0.001, weight_decay=0.001)

writer = SummaryWriter()

num_epochs = 200
cudnn.benchmark = True
start_time = time.time()
train_loss_values_improved = []
test_accuracy_values_improved = []

for epoch in range(num_epochs):
    model_improved.train()
    total_loss = 0
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_improved(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    writer.add_scalar("Loss/train_improved", total_loss / len(train_loader), epoch)
    train_loss_values_improved.append(total_loss / len(train_loader))
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}')

    model_improved.eval()
    with torch.no_grad():
        total_correct = 0
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_improved(inputs)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
        accuracy = total_correct / len(test_loader.dataset)
        writer.add_scalar("Accuracy/test_improved", accuracy, epoch)
        test_accuracy_values_improved.append(accuracy)
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Accuracy: {accuracy:.4f}')

end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")

if model_improved and accuracy > 0:
    print("Success")

torch.save(model_improved.state_dict(), 'model_improved.pth')

model_improved.load_state_dict(torch.load('model_improved.pth'))

model_improved.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_improved(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

accuracy = correct / len(test_loader.dataset)
print(f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {accuracy:.4f}')

print(f"Number of parameters: {count_parameters(model_improved)}")

plt.plot(train_loss_values_improved)
plt.plot(test_accuracy_values_improved)
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.show()
