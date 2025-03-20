import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

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
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = (outputs > 0).long().squeeze(1)
            predictions.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy, predictions


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
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# Set the random seed for reproducibility
torch.manual_seed(0)

### UNCOMMENT THE FOLLOWING LINES TO TRAIN THE MODEL ###
# From Scratch
model_scratch = ResNet18(pretrained=False, probing=False)
# Linear probing
# model_linear_probe = ResNet18(pretrained=True, probing=True)
# Fine-tuning
model_finetune = ResNet18(pretrained=True, probing=False)

transform = model_scratch.transform
batch_size = 32
num_of_epochs = 1
learning_rate = 0.01
path = 'C:/Users/Yehuda Frist/ML Methods/exercise_4/whichfaceisreal'
train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_scratch = model_scratch.to(device)
model_finetune = model_finetune.to(device)

### Define the loss function and the optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_scratch.parameters(), lr=learning_rate)
### Train the model

# Train the model
for epoch in range(num_of_epochs):
    # Run a training epoch
    loss_scratch = run_training_epoch(model_scratch, criterion, optimizer, train_loader, device)
    loss_ft = run_training_epoch(model_finetune, criterion, optimizer, train_loader, device)
    # Compute the accuracy
    train_acc_scratch = compute_accuracy(model_scratch, train_loader, device)
    train_acc_ft = compute_accuracy(model_finetune, train_loader, device)
    # Compute the validation accuracy
    val_acc_scratch, _ = compute_accuracy(model_scratch, val_loader, device)
    val_acc_ft, _ = compute_accuracy(model_finetune, val_loader, device)
    print(f'Model scratch: Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss_scratch:.4f}, Val accuracy: {val_acc_scratch:.4f}')
    print(f'Model FT: Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss_ft:.4f}, Val accuracy: {val_acc_ft:.4f}')
_, scratch_pred = compute_accuracy(model_scratch, test_loader, device)
_, ft_pred = compute_accuracy(model_finetune, test_loader, device)


# Find samples correctly classified by linear probing but misclassified by training from scratch
misclassified_indices = \
    [i for i, (true_label, pred_ft, pred_scratch)
     in enumerate(zip(test_loader.dataset.targets, ft_pred, scratch_pred))
     if pred_ft == true_label and pred_scratch != true_label]
print("Number of misclassified indices:", len(misclassified_indices))
print("Misclassified indices:", misclassified_indices)

# Visualize 5 samples
for idx in misclassified_indices[:5]:
    print("Index:", idx)
    true_label = val_loader.dataset.targets[idx]
    pred_ft = ft_pred[idx]
    pred_scratch = scratch_pred[idx]
    print("True Label:", true_label)
    print("Predicted by Fine tuning:", pred_ft)
    print("Predicted by Scratch:", pred_scratch)
    image, label = val_loader.dataset[idx]
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.title(f"True Label: {label}, "
              f"Predicted by Fine tuning: {pred_ft}, "
              f"Predicted by Scratch: {int(pred_scratch)}")
    plt.show()

