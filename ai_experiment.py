import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class HyperDimensionalSpace:
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def evolve(self, data):
        return np.random.normal(0, 0.1, size=(data.shape[0], self.dimensions)) + data


class UtilitarianLoss(nn.Module):
    def __init__(self):
        super(UtilitarianLoss, self).__init__()

    def forward(self, predictions, targets):
        return -torch.sum(predictions * targets)


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
        return self.data[idx], self.labels[idx]


class EvolutionaryModel(nn.Module):
    def __init__(self):
        super(EvolutionaryModel, self).__init__()
        self.fc1 = nn.Linear(20, 10)
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

                evolved_inputs = torch.from_numpy(
                    self.hyper_space.evolve(inputs.numpy())
                ).float()
                outputs = self.model(evolved_inputs)
                loss = self.loss_fn(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f"Epoch {epoch+1}, Loss: {loss.item()}")


class AdditiveEvolutionaryTrainer(EvolutionaryTrainer):
    def __init__(self, model, optimizer, loss_fn, hyper_space, dataset, alpha=0.1):
        super(AdditiveEvolutionaryTrainer, self).__init__(
            model, optimizer, loss_fn, hyper_space, dataset
        )
        self.alpha = alpha
        self.preserved_model = EvolutionaryModel()
        self.preserved_model.load_state_dict(model.state_dict())

    def evolve(self):
        for key in self.preserved_model.state_dict():
            self.model.state_dict()[key].data += self.alpha * (
                self.preserved_model.state_dict()[key].data
                - self.model.state_dict()[key].data
            )

    def train(self, epochs):
        data_loader = DataLoader(self.dataset, batch_size=10, shuffle=True)
        for epoch in range(epochs):
            self.evolve()
            self.preserved_model.load_state_dict(self.model.state_dict())
            for batch in data_loader:
                inputs, labels = batch
                inputs = torch.from_numpy(inputs).float()
                labels = torch.from_numpy(labels).float()

                evolved_inputs = torch.from_numpy(
                    self.hyper_space.evolve(inputs.numpy())
                ).float()
                outputs = self.model(evolved_inputs)
                loss = self.loss_fn(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f"Epoch {epoch+1}, Loss: {loss.item()}")


def main():
    torch.manual_seed(0)
    hyper_space = HyperDimensionalSpace(dimensions=10)
    utilitarian_loss = UtilitarianLoss()
    evolutionary_model = EvolutionaryModel()
    stoic_optimizer = StoicOptimizer(params=evolutionary_model.parameters(), lr=0.01)

    np.random.seed(0)
    data = np.random.rand(100, 20)
    labels = np.random.rand(100, 10)
    existential_dataset = ExistentialDataset(data=data, labels=labels)

    trainer = AdditiveEvolutionaryTrainer(
        evolutionary_model,
        stoic_optimizer,
        utilitarian_loss,
        hyper_space,
        existential_dataset,
    )
    trainer.train(epochs=10)


if __name__ == "__main__":
    main()