import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


#################################################################################
#################################################################################
#################################################################################
#################################################################################

# Define paths and parameters
data_dir = '/content/drive/MyDrive/Vaay/Newset/Data/'
output_dir = '/content/drive/MyDrive/Train_test_val'
categories = {
    "airport_terminal": "Airport",
    "bakery/shop": "Bakery",
    "bowling_alley": "Bowling",
    "classroom": "classroom",
    "music_studio": "music_studio"
}
num_images_per_class = 500  # Total images per class
train_size = 350
val_size = 75
test_size = 75
resize_dim = (224, 224)

# Create output directories
for split in ['train', 'val', 'test']:
    for category in categories.values():
        os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)

# Function to resize and save images
def resize_and_save(img_path, output_path):
    img = Image.open(img_path)
    img = img.resize(resize_dim, Image.ANTIALIAS)
    img.save(output_path)

# Process each category
for category_path, category_name in categories.items():
    full_category_path = os.path.join(data_dir, category_path)
    images = [os.path.join(full_category_path, img) for img in os.listdir(full_category_path) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Shuffle and select a subset of images
    random.shuffle(images)
    images = images[:num_images_per_class]

    # Split images into training, validation, and test sets
    train_images, test_images = train_test_split(images, test_size=(val_size + test_size) / num_images_per_class, random_state=42)
    val_images, test_images = train_test_split(test_images, test_size=test_size / (val_size + test_size), random_state=42)

    # Print the number of images in each set
    print(f"{category_name}:")
    print(f"  Training: {len(train_images)}")
    print(f"  Validation: {len(val_images)}")
    print(f"  Testing: {len(test_images)}")

    # Resize and save images to respective directories
    for img_path in train_images:
        resize_and_save(img_path, os.path.join(output_dir, 'train', category_name, os.path.basename(img_path)))
    for img_path in val_images:
        resize_and_save(img_path, os.path.join(output_dir, 'val', category_name, os.path.basename(img_path)))
    for img_path in test_images:
        resize_and_save(img_path, os.path.join(output_dir, 'test', category_name, os.path.basename(img_path)))

print("Dataset preparation completed.")

#################################################################################
#################################################################################
#################################################################################
#################################################################################
import numpy as np

output_dir = '/content/drive/MyDrive/Train_test_val'
categories = {
    "airport_terminal": "Airport",
    "bakery/shop": "Bakery",
    "bowling_alley": "Bowling",
    "classroom": "classroom",
    "music_studio": "music_studio"
}

#################################################################################
#################################################################################
#################################################################################
#################################################################################


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
# Calculate the number of images in each category for each split
data_counts = {'Category': [], 'Split': [], 'Count': []}

for split in ['train', 'val', 'test']:
    for category in categories.values():
        category_path = os.path.join(output_dir, split, category)
        count = len(os.listdir(category_path))
        data_counts['Category'].append(category)
        data_counts['Split'].append(split)
        data_counts['Count'].append(count)

# Convert to DataFrame for easier plotting
data_counts_df = pd.DataFrame(data_counts)

# Plot the distribution of data
plt.figure(figsize=(12, 6))
sns.barplot(x='Category', y='Count', hue='Split', data=data_counts_df)
plt.title('Data Distribution Across Train, Validation, and Test Sets')
plt.xlabel('Category')
plt.ylabel('Number of Images')
plt.legend(title='Split')
plt.show()

#################################################################################
#################################################################################
#################################################################################
#################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define categories
categories = {
    "airport_terminal": "Airport",
    "bakery/shop": "Bakery",
    "bowling_alley": "Bowling",
    "classroom": "classroom",
    "music_studio": "music_studio"
}

# Load datasets
train_dataset = ImageFolder(root='/content/drive/MyDrive/Train_test_val/train', transform=transform)
val_dataset = ImageFolder(root='/content/drive/MyDrive/Train_test_val/val', transform=transform)
test_dataset = ImageFolder(root='/content/drive/MyDrive/Train_test_val/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size= 32, shuffle=True)  # Reduced batch size
val_loader = DataLoader(val_dataset, batch_size= 32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size= 32, shuffle=False)

# Print the number of images in each dataset
print(f'Number of training images: {len(train_dataset)}')
print(f'Number of validation images: {len(val_dataset)}')
print(f'Number of testing images: {len(test_dataset)}')

# Print the number of images per category in each dataset
for split in ['train', 'val', 'test']:
    for category in categories.values():
        category_path = os.path.join('/content/drive/MyDrive/Train_test_val', split, category)
        count = len(os.listdir(category_path))
        print(f'Number of {split} images in {category}: {count}')


#################################################################################
#################################################################################
#################################################################################
#################################################################################

class CUSTOMCNN(nn.Module):
    def __init__(self):
        super(CUSTOMCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 5)  # 5 classes

    def forward(self, x):
        x = self.pool(self.batch_norm1(torch.relu(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(self.batch_norm2(torch.relu(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool(self.batch_norm3(torch.relu(self.conv3(x))))
        x = self.dropout3(x)
        x = self.pool(self.batch_norm4(torch.relu(self.conv4(x))))
        x = x.view(-1, 256 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
model = CUSTOMCNN()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduced learning rate for better performance

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)



def plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'r', label='Training loss')
    plt.plot(epochs, val_losses, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_accuracies, 'b', label='Validation accuracy')
    plt.title('Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_precisions, 'b', label='Validation precision')
    plt.plot(epochs, val_recalls, 'g', label='Validation recall')
    plt.plot(epochs, val_f1_scores, 'r', label='Validation F1 score')
    plt.title('Validation precision, recall, and F1 score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.show()



#################################################################################
#################################################################################
#################################################################################
#################################################################################




def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25):
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        scheduler.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        val_precision = precision_score(all_labels, all_preds, average='weighted')
        val_recall = recall_score(all_labels, all_preds, average='weighted')
        val_f1 = f1_score(all_labels, all_preds, average='weighted')

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1_scores.append(val_f1)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%, Val Precision: {val_precision:.4f}, '
              f'Val Recall: {val_recall:.4f}, Val F1 Score: {val_f1:.4f}')

    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores)

# Use the updated train_model function
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)

#################################################################################
#################################################################################
#################################################################################
#################################################################################


import torch
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model_v2(model, data_loader, device='cuda'):
    model.to(device).eval()
    total_correct = 0
    total_samples = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            total_samples += labels.size(0)
            total_correct += (preds == labels).sum().item()
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = (total_correct / total_samples) * 100
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)

# Example usage
# Assuming `model` and `test_loader` are already defined and loaded
evaluate_model_v2(model, test_loader)

# Ensure the save directory exists
save_path = '/content/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
#################################################################################
#################################################################################
#################################################################################
#################################################################################