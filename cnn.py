import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from sklearn.linear_model import LogisticRegression

class ResNet18(nn.Module):
    def __init__(self, pretrained=False, probing=False):
        super(ResNet18, self).__init__()
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
            self.resnet18 = resnet18(weights=weights)
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            self.resnet18 = resnet18()
        in_features_dim = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()
        if probing:
            for name, param in self.resnet18.named_parameters():
                    param.requires_grad = False
        self.logistic_regression = nn.Linear(in_features_dim, 1)

    def forward(self, x):
        features = self.resnet18(x)
        ### YOUR CODE HERE ###
        return self.logistic_regression(features)


def get_loaders(path, transform, batch_size):
    """
    Get the data loaders for the train, validation and test sets.
    :param path: The path to the 'whichfaceisreal' directory.
    :param transform: The transform to apply to the images.
    :param batch_size: The batch size.
    :return: The train, validation and test data loaders.
    """
    train_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'train'), transform=transform)
    val_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'val'), transform=transform)
    test_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def compute_accuracy(model, data_loader, device):
    """
    Compute the accuracy of the model on the data in data_loader
    :param model: The model to evaluate.
    :param data_loader: The data loader.
    :param device: The device to run the evaluation on.
    :return: The accuracy of the model on the data in data_loader
    """
    model.eval()
    ### YOUR CODE HERE ###
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = (outputs > 0).long().squeeze(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def run_training_epoch(model, criterion, optimizer, train_loader, device):
    """
    Run a single training epoch
    :param model: The model to train
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param train_loader: The data loader
    :param device: The device to run the training on
    :return: The average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    for (imgs, labels) in tqdm(train_loader, total=len(train_loader)):
        ### YOUR CODE HERE ###
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def linear_probe(model, train_loader, val_loader, test_loader):
    """
    Perform linear probing using a pre-trained ResNet18 model.
    :param model: The pre-trained ResNet18 model.
    :param train_loader: The training data loader.
    :param val_loader: The validation data loader.
    :param test_loader: The test data loader.
    """
    # Extract features from the pre-trained ResNet18 model
    model.eval()
    with torch.no_grad():
        train_features = []
        train_labels = []
        for images, labels in train_loader:
            features = model(images)
            train_features.extend(features.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

    # Train a Logistic Regression model
    logistic_regression = LogisticRegression(max_iter=1000)
    logistic_regression.fit(train_features, train_labels)

    # Evaluate the model
    def evaluate(loader, name):
        correct = 0
        total = 0
        for images, labels in loader:
            features = model(images)
            predicted = logistic_regression.predict(features.detach().cpu().numpy())
            correct += (predicted == labels.numpy()).sum().item()
            total += labels.size(0)
        accuracy = correct / total
        print(f"{name} Accuracy: {accuracy:.4f}")

    evaluate(train_loader, "Train")
    evaluate(val_loader, "Val")
    evaluate(test_loader, "Test")


# Set the random seed for reproducibility
torch.manual_seed(0)

### UNCOMMENT THE FOLLOWING LINES TO TRAIN THE MODEL ###
# From Scratch
# model = ResNet18(pretrained=False, probing=False)
# Linear probing
# model = ResNet18(pretrained=True, probing=True)
# Fine-tuning
model = ResNet18(pretrained=True, probing=False)

transform = model.transform
batch_size = 32
num_of_epochs = 30
learning_rate = 0.001
path = 'C:/Users/Yehuda Frist/ML Methods/exercise_4/whichfaceisreal'
train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
### Define the loss function and the optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
### Train the model

# Train the model
for epoch in range(num_of_epochs):
    # Run a training epoch
    loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
    # Compute the accuracy
    train_acc = compute_accuracy(model, train_loader, device)
    # Compute the validation accuracy
    val_acc = compute_accuracy(model, val_loader, device)
    print(f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')
    # Stopping condition
    if val_acc > 0.99:
        break

# Compute the test accuracy
test_acc = compute_accuracy(model, test_loader, device)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Learning Rate: {learning_rate:.5f}")

# # Q4 - linear probing with sklearn
# linear_probe(model, train_loader, val_loader, test_loader)