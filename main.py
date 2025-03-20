from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from NN_tutorial import *
from tqdm import tqdm


# Define a simple MLP model with 100 hidden layers, each with 4 neurons
class MLP(nn.Module):
    def __init__(self, outputdim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 4),  # Update the input size to match the number of features
            nn.ReLU(),
            # Add 99 more layers with 4 neurons each
            *[
                nn.Sequential(nn.Linear(4, 4), nn.Sigmoid())
                for _ in range(98)
            ],
            nn.Linear(4, outputdim)
        )

    def forward(self, x):
        return self.model(x)


class MLP_OPT(nn.Module):
    def __init__(self, outputdim):
        super(MLP_OPT, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 4),  # Update the input size to match the number of features
            nn.ReLU(),
            nn.BatchNorm1d(4),  # Add batch normalization after the first linear layer
            # Add 99 more layers with 4 neurons each
            *[
                nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.BatchNorm1d(4))
                for _ in range(98)
            ],
            nn.Linear(4, outputdim)
        )

    def forward(self, x):
        return self.model(x)


class MLP_sin(nn.Module):
    def __init__(self, outputdim):
        super(MLP_sin, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(20, 16),  # Update the input size to match the number of features
            nn.ReLU(),
            # Add 4 more layers with 16 neurons each
            *[
                nn.Sequential(nn.Linear(16, 16), nn.ReLU())
                for _ in range(4)
            ],
            nn.Linear(16, outputdim)
        )

    def forward(self, x):
        return self.model(x)


def train_models(depth_list, width_dict):
    best_model = None
    worst_model = None
    best_val_acc = 0.0
    worst_val_acc = 1.0

    best_train_losses = []
    best_val_losses = []
    best_test_losses = []
    worst_train_losses = []
    worst_val_losses = []
    worst_test_losses = []

    for depth, width in zip(depth_list, width_dict.values()):
        output_dim = len(train_data['country'].unique())
        model = [nn.Linear(2, width), nn.ReLU()] + [nn.Linear(width, width), nn.ReLU()] * (depth - 1) + [
            nn.Linear(width, output_dim)]
        model = nn.Sequential(*model)
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = \
            train_model(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256)

        if val_accs[-1] > best_val_acc:
            best_val_acc = val_accs[-1]
            best_model = model
            best_train_losses = train_losses
            best_val_losses = val_losses
            best_test_losses = test_losses

        if val_accs[-1] < worst_val_acc:
            worst_val_acc = val_accs[-1]
            worst_model = model
            worst_train_losses = train_losses
            worst_val_losses = val_losses
            worst_test_losses = test_losses

    plt.figure()
    plt.plot(best_train_losses, label='Train', color='red')
    plt.plot(best_val_losses, label='Val', color='blue')
    plt.plot(best_test_losses, label='Test', color='green')
    plt.title('Best Model Losses')
    plt.legend()
    plt.show()

    plot_decision_boundaries(best_model, test_data[['long', 'lat']].values, test_data['country'].values,
                             'Best Model Decision Boundaries', implicit_repr=False)

    plt.figure()
    plt.plot(worst_train_losses, label='Train', color='red')
    plt.plot(worst_val_losses, label='Val', color='blue')
    plt.plot(worst_test_losses, label='Test', color='green')
    plt.title('Worst Model Losses')
    plt.legend()
    plt.show()

    plot_decision_boundaries(worst_model, test_data[['long', 'lat']].values, test_data['country'].values,
                             'Worst Model Decision Boundaries', implicit_repr=False)


def train_depth_models(train_data, val_data, test_data, depths, width, output_dim=2, lr=0.001, epochs=100, batch_size=128):
    depth_accs = []

    for depth in depths:
        output_dim = len(train_data['country'].unique())
        model = [nn.Linear(2, width), nn.ReLU()]
        model += [nn.Linear(width, width), nn.ReLU()] * (depth - 1)
        model += [nn.Linear(width, output_dim)]
        model = nn.Sequential(*model)

        model, train_accs, val_accs, test_accs, _, _, _ = train_model(train_data, val_data, test_data, model, lr=lr, epochs=epochs, batch_size=batch_size)
        depth_accs.append((train_accs[-1], val_accs[-1], test_accs[-1]))

    depth_accs = np.array(depth_accs)

    plt.figure()
    plt.plot(depths, depth_accs[:,0], label='Train', color='red')
    plt.plot(depths, depth_accs[:,1], label='Val', color='blue')
    plt.plot(depths, depth_accs[:,2], label='Test', color='green')
    plt.xlabel('Number of Hidden Layers')
    plt.ylabel('Accuracy')
    plt.title('Effect of Network Depth on MLP Performance')
    plt.legend()
    plt.show()


def train_width_models(train_data, val_data, test_data, depth, widths, output_dim=2, lr=0.001, epochs=100, batch_size=128):
    width_accs = []

    for width in widths:
        output_dim = len(train_data['country'].unique())
        model = [nn.Linear(2, width), nn.ReLU()]
        model += [nn.Linear(width, width), nn.ReLU()] * (depth - 1)
        model += [nn.Linear(width, output_dim)]
        model = nn.Sequential(*model)

        model, train_accs, val_accs, test_accs, _, _, _ = \
            train_model(train_data, val_data, test_data, model, lr=lr, epochs=epochs, batch_size=batch_size)
        width_accs.append((train_accs[-1], val_accs[-1], test_accs[-1]))

    width_accs = np.array(width_accs)

    plt.figure()
    plt.plot(widths, width_accs[:,0], label='Train', color='red')
    plt.plot(widths, width_accs[:,1], label='Val', color='blue')
    plt.plot(widths, width_accs[:,2], label='Test', color='green')
    plt.xlabel('Width of Hidden Layers')
    plt.ylabel('Accuracy')
    plt.title('Effect of Network Width on MLP Performance')
    plt.legend()
    plt.show()


def train_model_with_gradients(train_loader, model, lr=0.001, epochs=10, layer_indices=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if layer_indices is None:
        layer_indices = np.array([0, 30, 60, 90, 95, 99])
        layer_indices = 2 * layer_indices # each layer has both matrix and bias as weights (2 parameters)

    gradients = {i: [] for i in layer_indices}
    avg_gradients_per_epoch = {i: [] for i in layer_indices}

    for epoch in tqdm(range(epochs)):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Record gradients for selected layers
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            for i, param in enumerate(model.parameters()):
                if i in layer_indices:
                    grad = torch.norm(param.grad).item()
                    print(f"Layer {i}: {grad}")
                    gradients[i].append(grad ** 2)  # squared norm

        for i in layer_indices:
            avg_gradients_per_epoch[i].append(sum(gradients[i]) / len(gradients[i]))
        gradients = {i: [] for i in layer_indices}

    return avg_gradients_per_epoch


if __name__ == '__main__':
    # seeding
    torch.manual_seed(42)
    np.random.seed(42)

    # read data
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    # Q6.2.1&2
    depth_list = [1, 2, 6, 10, 6, 6, 6]
    width_dict = {1: 16, 2: 16, 6: 16, 10: 16, 6: 8, 6: 32, 6: 64}
    train_models(depth_list, width_dict)

    # Q6.2.3 - depths
    depths = [1, 2, 6, 10]
    width = 16
    train_depth_models(train_data, val_data, test_data, depths, width, lr=0.001, epochs=100, batch_size=128)

    # Q6.2.4 - widths
    depth = 6
    widths = [8, 16, 32, 64]
    train_width_models(train_data, val_data, test_data, depth, widths, lr=0.001, epochs=100, batch_size=128)

    # Q6.2.5 - gradients
    # Read data
    data_numpy, _ = read_data_demo('train.csv')
    train_data_torch = torch.tensor(data_numpy[:, 1:-1]).float()
    labels = torch.tensor(data_numpy[:, -1]).long()

    # Define and train the model
    outputdim = len(np.unique(labels))
    model = MLP(outputdim)
    # Create a TensorDataset
    dataset = TensorDataset(train_data_torch, labels)

    # Create a DataLoader
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    gradients = train_model_with_gradients(train_loader, model, lr=0.001, epochs=10)

    # Plotting
    layer_indices = np.array([0, 30, 60, 90, 95, 99])
    layer_indices = layer_indices * 2
    for layer_index in layer_indices:
        plt.plot(range(10), gradients[layer_index], label=f'Layer {layer_index / 2}')

    plt.xlabel('Epoch')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Magnitude vs. Epoch for Selected Layers')
    plt.yscale('log')
    plt.legend()
    plt.show()

    # Q6.2.6 - gradients optimization
    # Read data
    data_numpy, _ = read_data_demo('train.csv')
    train_data_torch = torch.tensor(data_numpy[:, 1:-1]).float()
    labels = torch.tensor(data_numpy[:, -1]).long()

    # Define and train the model
    outputdim = len(np.unique(labels))
    model = MLP_OPT(outputdim)
    # Create a TensorDataset
    dataset = TensorDataset(train_data_torch, labels)

    # Create a DataLoader
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    gradients = train_model_with_gradients(train_loader, model, lr=0.001, epochs=10)

    # Plotting
    layer_indices = np.array([0, 30, 60, 90, 95, 99])
    layer_indices = layer_indices * 2
    for layer_index in layer_indices:
        plt.plot(range(10), gradients[layer_index], label=f'Layer {layer_index / 2}')

    plt.xlabel('Epoch')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Magnitude vs. Epoch for Selected Layers With BatchNorm')
    plt.yscale('log')
    plt.legend()
    plt.show()

    # #Q6.2.7
    # Read data
    data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values

    # Add sine transformations
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    X_with_sin = []
    for alpha in alphas:
        sin_data = np.sin(alpha * X)
        X_with_sin.append(sin_data)
    X_with_sin = np.concatenate(X_with_sin, axis=1)

    # Split the data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_with_sin, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())

    # Define and train the model
    model = MLP_sin(len(np.unique(y)))
    trained_model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = \
        train_model(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=50, batch_size=256)

    # Print final test accuracy
    print(f"Final Test Accuracy: {test_accs[-1]}")

    plt.figure()
    plt.plot(train_losses, label='Train', color='red')
    plt.plot(val_losses, label='Val', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.title('Losses - Preprocessed')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_accs, label='Train', color='red')
    plt.plot(val_accs, label='Val', color='blue')
    plt.plot(test_accs, label='Test', color='green')
    plt.title('Accs. - Preprocessed')
    plt.legend()
    plt.show()

    plot_decision_boundaries(
        trained_model,
        X, y,
        'Decision Boundaries- Implicit Representation',
        implicit_repr=True)

