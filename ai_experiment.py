import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class HyperDimensionalSpace:
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def evolve(self, data):
        evolved_data = []
        for point in data:
            mutated_point = point + np.random.normal(0, 0.1, size=self.dimensions)
            evolved_data.append(mutated_point)
        return evolved_data


class UtilitarianLoss(nn.Module):
    def __init__(self):
        super(UtilitarianLoss, self).__init__()

    def forward(self, predictions, targets):
        loss = -torch.sum(predictions * targets)
        return loss


class StoicOptimizer(optim.Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(StoicOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                p.add_(d_p, alpha=-group["lr"])

        return loss


class ExistentialDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EvolutionaryTrainer:
    def __init__(self, model, optimizer, loss_fn, hyper_space, dataset):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.hyper_space = hyper_space
        self.dataset = dataset

    def train(self, epochs):
        data_loader = DataLoader(self.dataset, batch_size=10, shuffle=True)
        for epoch in range(epochs):
            for batch in data_loader:
                inputs, labels = batch
                inputs = torch.from_numpy(inputs).float()
                labels = torch.from_numpy(labels).float()

                evolved_inputs = self.hyper_space.evolve(inputs.detach().numpy())
                evolved_inputs = torch.from_numpy(np.array(evolved_inputs)).float()

                outputs = self.model(evolved_inputs)
                loss = self.loss_fn(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f"Epoch {epoch+1}, Loss: {loss.item()}")


if __name__ == "__main__":
    hyper_space = HyperDimensionalSpace(dimensions=10)
    utilitarian_loss = UtilitarianLoss()
    model = Model()
    stoic_optimizer = StoicOptimizer(params=model.parameters(), lr=0.01)

    np.random.seed(0)
    data = np.random.rand(100, 10)
    labels = np.random.rand(100, 10)
    existential_dataset = ExistentialDataset(data=data, labels=labels)

    trainer = EvolutionaryTrainer(
        model, stoic_optimizer, utilitarian_loss, hyper_space, existential_dataset
    )
    trainer.train(epochs=10)
