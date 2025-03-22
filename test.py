import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
import random

# Define the same model architecture as in Rust
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        hidden = self.fc1(x)
        output = self.fc2(hidden)
        return output

# Load MNIST dataset
print("Loading MNIST dataset...")
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

print("Using 100 training examples and 10 test examples")

# Hyperparameters - same as in Rust
batch_size = 1
num_epochs = 5
learning_rate = 0.001

# Initialize model
model = SimpleNN()
criterion = nn.MSELoss()

# Collect model parameters for manual optimization
params = list(model.parameters())

# Prepare for training
indices = list(range(100))  # Only use first 100 examples
start_time = time.time()

# Training loop
for epoch in range(num_epochs):
    # Shuffle indices for this epoch
    random.shuffle(indices)
    
    total_loss = 0.0
    num_correct = 0
    num_batches = len(indices) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_indices = indices[batch_start:batch_start + batch_size]
        
        # Get batch data
        x, label = train_dataset[batch_indices[0]]
        x = x.view(-1, 784)  # Flatten
        
        # Zero gradients before forward pass
        for param in params:
            if param.grad is not None:
                param.grad.zero_()
        
        # Forward pass
        output = model(x)
        
        # Create one-hot encoded target tensors
        target = torch.zeros(batch_size, 10)
        target[0, label] = 1.0
        
        # Compute loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Manual parameter update - direct gradient descent
        with torch.no_grad():
            for param in params:
                param -= learning_rate * param.grad
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        num_correct += (predicted == label).sum().item()
        
        loss_val = loss.item()
        total_loss += loss_val
        
        print(
            f"Epoch {epoch+1}/{num_epochs} - Example {batch_idx+1}/{num_batches} - "
            f"Loss: {loss_val:.4f} - Accuracy: {100.0 * num_correct/(batch_idx+1):.2f}%"
        )
    
    elapsed = time.time() - start_time
    print(
        f"Epoch {epoch+1}/{num_epochs} completed - "
        f"Loss: {total_loss/num_batches:.4f} - "
        f"Accuracy: {100.0 * num_correct/num_batches:.2f}% - "
        f"Time: {elapsed:.2f}s"
    )

# Evaluate on test set
print("\nEvaluating on test set...")

test_indices = list(range(10))  # Only use first 10 test examples
test_correct = 0

for idx, test_idx in enumerate(test_indices):
    # Get test example
    x, label = test_dataset[test_idx]
    x = x.view(-1, 784)  # Flatten
    
    # Forward pass (without tracking gradients)
    with torch.no_grad():
        output = model(x)
    
    # Calculate accuracy
    _, predicted = torch.max(output.data, 1)
    
    if predicted.item() == label:
        test_correct += 1
    
    print(f"Test example {idx+1}: Predicted: {predicted.item()}, Actual: {label}")

test_accuracy = 100.0 * test_correct / len(test_indices)
print(f"Test Accuracy: {test_accuracy:.2f}% ({test_correct}/{len(test_indices)})")