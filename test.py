import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
import random

# Define the model architecture
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(768, 1024)  # Input layer: 768 -> 1024
        self.fc2 = nn.Linear(1024, 10)   # Output layer: 1024 -> 10 (digits)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        output = self.fc2(hidden)
        return output

# Load MNIST dataset
print("Loading MNIST dataset...")
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.01

# Initialize model and loss function
model = MNISTModel()
criterion = nn.MSELoss()

# Collect model parameters for manual SGD
params = list(model.parameters())

# Prepare training data
num_train_samples = 5000  # Using more samples for better training
train_indices = list(range(num_train_samples))
test_indices = list(range(1000))  # Using more test samples

print(f"Training with {num_train_samples} examples")
start_time = time.time()

# Training loop
for epoch in range(num_epochs):
    # Shuffle indices for this epoch
    random.shuffle(train_indices)
    
    total_loss = 0.0
    num_correct = 0
    num_batches = len(train_indices) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = batch_start + batch_size
        batch_indices = train_indices[batch_start:batch_end]
        
        # Zero gradients
        for param in params:
            if param.grad is not None:
                param.grad.zero_()
        
        batch_loss = 0
        batch_correct = 0
        
        # Process each example in the batch
        for idx in batch_indices:
            # Get data
            x, label = train_dataset[idx]
            # Flatten and resize to 768 (from 784) by dropping some pixels
            x = x.view(-1)[:768].unsqueeze(0)
            
            # Forward pass
            output = model(x)
            
            # Create one-hot encoded target
            target = torch.zeros(1, 10)
            target[0, label] = 1.0
            
            # Compute loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            batch_correct += (predicted == label).sum().item()
            batch_loss += loss.item()
        
        # Manual SGD update
        with torch.no_grad():
            for param in params:
                param -= learning_rate * param.grad
        
        # Statistics for this batch
        avg_loss = batch_loss / batch_size
        total_loss += avg_loss
        num_correct += batch_correct
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{num_batches} - "
                  f"Loss: {avg_loss:.4f} - Batch Accuracy: {100.0 * batch_correct/batch_size:.2f}%")
    
    # Epoch statistics
    epoch_accuracy = 100.0 * num_correct / num_train_samples
    epoch_loss = total_loss / num_batches
    elapsed = time.time() - start_time
    
    print(f"Epoch {epoch+1}/{num_epochs} completed - "
          f"Loss: {epoch_loss:.4f} - "
          f"Accuracy: {epoch_accuracy:.2f}% - "
          f"Time: {elapsed:.2f}s")

# Evaluate on test set
print("\nEvaluating on test set...")
test_correct = 0

with torch.no_grad():
    for test_idx in test_indices:
        # Get test example
        x, label = test_dataset[test_idx]
        x = x.view(-1)[:768].unsqueeze(0)  # Flatten and resize to 768
        
        # Forward pass
        output = model(x)
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        test_correct += (predicted == label).item()

test_accuracy = 100.0 * test_correct / len(test_indices)
print(f"Test Accuracy: {test_accuracy:.2f}% ({test_correct}/{len(test_indices)})")