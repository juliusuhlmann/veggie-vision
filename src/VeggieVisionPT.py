import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os


# Dataset class for images
class VeggieDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # Images should have shape (batch, channels, height, width)
        self.labels = labels
        self.transform = transform 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.fromarray((image * 255).astype(np.uint8))  # Rescale and convert to a PIL Image
        if self.transform:
            image = self.transform(image)  
        label = torch.tensor(self.labels[idx], dtype=torch.float32).squeeze()  # Convert label to tensor
        return image, label

# Model class with MobileNetV2 as base model for weight prediction
class VeggieVision(nn.Module):
    def __init__(self):
        super(VeggieVision, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        self.base_model.features.requires_grad = False  # Freeze feature extractor layers (MobileNetV2)
        self.fc1 = nn.Linear(self.base_model.last_channel * 7 * 7, 128)  
        self.dropout = nn.Dropout(0.3)  
        self.fc2 = nn.Linear(128, 1)  

    def forward(self, x):
        x = self.base_model.features(x)  # Pass through pretrained feature extractor
        x = x.reshape(x.size(0), -1)  # Flatten features
        x = torch.relu(self.fc1(x))  # Apply first linear layer with ReLU
        x = self.dropout(x)
        x = self.fc2(x)  # Output layer for regression
        return x

# Augmentation
train_transform = transforms.Compose([
    transforms.RandomRotation(360),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

# Function to prepare data loaders
def prepare_data_loaders(X, y, validation_split=0.2, batch_size=32):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
    train_dataset = VeggieDataset(X_train, y_train, transform=train_transform)
    val_dataset = VeggieDataset(X_val, y_val, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader

# Initialize model, loss, optimizer, and scheduler
def initialize_model(device, learning_rate=1e-4):
    model = VeggieVision().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)  # Adjust LR on plateau
    return model, criterion, optimizer, scheduler

# Training loop for the model
def training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=20, print_freq=1):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train(model, train_loader, criterion, optimizer, device, print_freq=print_freq)
        val_loss = validate(model, val_loader, criterion, device, print_freq=print_freq)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

# Training function for a single epoch
def train(model, dataloader, criterion, optimizer, device, print_freq=1):
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        
        # Print every few batches based on print frequency
        if (batch_idx + 1) % print_freq == 0:
            print(f"Batch {batch_idx+1}/{total_batches}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# Function to fine-tune model by unfreezing layers
def fine_tune(model, n_unfreeze_layers=1):
    layers = list(model.base_model.features.children())
    for layer in layers[-n_unfreeze_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

def setup_fine_tuning(model, optimizer, learning_rate=1e-5, n_unfreeze_layers=1):
    fine_tune(model, n_unfreeze_layers)  # Unfreeze layers for training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Update optimizer with new LR
    return optimizer

# Fine-tuning loop for model
def fine_tuning_loop(model, train_loader, val_loader, criterion, optimizer, device, fine_tune_epochs=2):
    for epoch in range(fine_tune_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Fine-tuning Epoch {epoch+1}/{fine_tune_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Validation function
def validate(model, dataloader, criterion, device, print_freq=1):
    model.eval()
    running_loss = 0.0
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            running_loss += loss.item() * images.size(0)
            
            # Print every few batches based on print frequency
            if (batch_idx + 1) % print_freq == 0:
                print(f"Validation Batch {batch_idx+1}/{total_batches}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# Predict on new data
def predict(model, X, device):
    model.eval()
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if len(X.shape) == 3:
        X = X.unsqueeze(0)  # Add batch dimension if single image
    X = X.to(device)
    with torch.no_grad():
        outputs = model(X)
        predictions = outputs.squeeze().cpu().numpy()  # Return as numpy array for easier handling
    return predictions

# Evaluate model on test data
def evaluate(model, X_test, y_test, device):
    predictions = predict(model, X_test, device)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error on Test Set: {mse:.4f}')
    return mse, np.array(predictions)



if __name__ == "__main__":
    # Set the working directory to the current file's directory
    os.chdir(os.path.dirname(__file__))

    
    # Load the data
    X = np.load("../data/processed_data/X.npy")
    y = np.load("../data/processed_data/y.npy")

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Prepare data loaders
    train_loader, val_loader = prepare_data_loaders(X_train, y_train, 0.2, 32)
        
    # Initialize model, loss function, optimizer and scheduler
    model, criterion, optimizer, scheduler = initialize_model(device, 1e-3)

    # Training phase
    training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs = 4)

    # Fine-tuning phase
    optimizer = setup_fine_tuning(model, optimizer, learning_rate=1e-5, n_unfreeze_layers=1)
    fine_tuning_loop(model, train_loader, val_loader, criterion, optimizer, device, epochs = 1)

    # Evaluation phase
    evaluate(model, X_test, y_test, device)

    # Save model
    torch.save(model.state_dict(), '../models/veggie_vision_pt.pth')