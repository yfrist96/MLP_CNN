import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time

from helpers import *


def train_model(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Using device:', device)

    trainset = torch.utils.data.TensorDataset(torch.tensor(train_data[['long', 'lat']].values).float(), torch.tensor(train_data['country'].values).long())
    valset = torch.utils.data.TensorDataset(torch.tensor(val_data[['long', 'lat']].values).float(), torch.tensor(val_data['country'].values).long())
    testset = torch.utils.data.TensorDataset(torch.tensor(test_data[['long', 'lat']].values).float(), torch.tensor(test_data['country'].values).long())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []

    for ep in range(epochs):
        model.train()
        pred_correct = 0
        ep_loss = 0.
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            #### YOUR CODE HERE ####

            # perform a training iteration

            # move the inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the gradients
            optimizer.zero_grad()
            # forward pass
            outputs = model(inputs)
            # calculate the loss
            loss = criterion(outputs, labels)
            # backward pass
            loss.backward()
            # update the weights
            optimizer.step()

            # name the model outputs "outputs"
            # and the loss "loss"

            #### END OF YOUR CODE ####

            pred_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            ep_loss += loss.item()

        train_accs.append(pred_correct / len(trainset))
        train_losses.append(ep_loss / len(trainloader))

        model.eval()
        with torch.no_grad():
            for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
                correct = 0
                total = 0
                ep_loss = 0.
                for inputs, labels in loader:
                    #### YOUR CODE HERE ####

                    # perform an evaluation iteration

                    # move the inputs and labels to the device
                    inputs, labels = inputs.to(device), labels.to(device)
                    # forward pass
                    outputs = model(inputs)
                    # calculate the loss
                    loss = criterion(outputs, labels)

                    #### END OF YOUR CODE ####

                    ep_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accs.append(correct / total)
                losses.append(ep_loss / len(loader))

        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1], test_accs[-1]))

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses


if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    output_dim = len(train_data['country'].unique())
    model = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
             nn.Linear(16, output_dim)  # output layer
             ]
    model = nn.Sequential(*model)

    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = \
        train_model(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256)

    plt.figure()
    plt.plot(train_losses, label='Train', color='red')
    plt.plot(val_losses, label='Val', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.title('Losses')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_accs, label='Train', color='red')
    plt.plot(val_accs, label='Val', color='blue')
    plt.plot(test_accs, label='Test', color='green')
    plt.title('Accs.')
    plt.legend()
    plt.show()

    plot_decision_boundaries(model, test_data[['long', 'lat']].values, test_data['country'].values,
                             'Decision Boundaries', implicit_repr=False)

    # Q6.1.2.1 - learning rates
    learning_rates = [1., 0.01, 0.001, 0.00001]
    for lr in learning_rates:
        model_copy = nn.Sequential(*[layer for layer in model])
        model_copy, _, val_accs, _, _, val_losses, _ = train_model(train_data, val_data, test_data, model_copy, lr=lr,
                                                                   epochs=50, batch_size=256)
        plt.plot(val_losses, label=f'LR={lr}')

    plt.title('Validation Losses for Different Learning Rates')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Q6.1.2.2 - epochs
    epochs_list = [1, 5, 10, 20, 50, 100]
    colors = ['b', 'g', 'r', 'c', 'm', 'y']  # Define colors for each epoch

    model_copy = nn.Sequential(*[layer for layer in model])
    _, _, val_accs, _, _, val_losses, _ = train_model(train_data, val_data, test_data, model_copy,
                                                      lr=0.001, epochs=max(epochs_list), batch_size=256)

    plt.figure(figsize=(10, 6))
    current_color = colors[0]  # Start with the color for the first epoch
    current_epoch = 0
    for i, epochs in enumerate(epochs_list):
        plt.plot(range(current_epoch + 1, epochs + 1), val_losses[current_epoch:epochs], label=f'Epochs={epochs}',
                 color=current_color)
        if i < len(colors) - 1:  # Use the next color for the next epoch range
            current_color = colors[i + 1]
        current_epoch = epochs

    plt.title('Validation Losses for Different Numbers of Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Q6.1.2.3 - Batch norm
    model_bn = [nn.Linear(2, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 1
                nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 2
                nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 3
                nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 4
                nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 5
                nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 6
                nn.Linear(16, output_dim)  # output layer
                ]
    model_bn = nn.Sequential(*model_bn)

    models = {'Regular': model, 'BatchNorm': model_bn}
    for name, model in models.items():
        model_copy = nn.Sequential(*[layer for layer in model])
        model_copy, _, val_accs, _, _, val_losses, _ = train_model(train_data, val_data, test_data, model_copy,
                                                                   lr=0.001, epochs=50, batch_size=256)
        plt.plot(val_losses, label=name)

    plt.title('Validation Losses for Regular and BatchNorm Models')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Q6.1.2.4 - Batch size
    batch_sizes = [1, 16, 128, 1024]
    epochs_dict = {1: 1, 16: 10, 128: 50, 1024: 50}
    loss_values = []

    for batch_size in batch_sizes:
        epochs = epochs_dict[batch_size]

        start_time = time.time()
        model_copy = nn.Sequential(*[layer for layer in model])
        _, _, val_accs, _, _, val_losses, _ = train_model(train_data, val_data, test_data, model_copy, lr=0.001, epochs=epochs,
                                                   batch_size=batch_size)
        end_time = time.time()
        print(f"Batch Size: {batch_size}, Time: {end_time - start_time:.2f} seconds")

        # Pad the val_losses list with None values to match the longest list
        val_losses += [None] * (max(epochs_dict.values()) - epochs)

        loss_values.append(val_losses)

    # Plotting
    plt.figure(figsize=(10, 6))
    for i, batch_size in enumerate(batch_sizes):
        plt.plot(range(1, max(epochs_dict.values()) + 1), loss_values[i], label=f'Batch Size={batch_size}')

    plt.title('Validation Loss over Epochs for Different Batch Sizes')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.yscale('log')
    plt.legend()
    plt.show()



