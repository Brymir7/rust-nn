import torch

# Input tensors
t1 = torch.tensor([1.0]) 
t2 = torch.tensor([2.0])

# Trainable parameters
w1 = torch.tensor([0.5], requires_grad=True)
w2 = torch.tensor([0.5], requires_grad=True)
b1 = torch.tensor([0.5], requires_grad=True)
b2 = torch.tensor([0.5], requires_grad=True)

# Target value
wanted_res = torch.tensor([5.0])

# Create parameter list and optimizer
params = [ b1, b2]
learning_rate = 0.1
optimizer = torch.optim.SGD(params, lr=learning_rate)

# Initial prediction
res = t1  + t2 + b1 + b2
print(f"Initial prediction: {res.item()}")

# Training loop
for i in range(5):
    # Forward pass
    res = t1  + t2 + b1 + b2
    
    # Compute loss (using MSE instead of abs)
    loss = ((res - wanted_res) ** 2)
    
    # Print current state
    print(f"Iteration {i}, Loss: {loss.item()}, Prediction: {res.item()}")
    
    # Backward pass and optimize
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()
    optimizer.step()       # Update parameters

# Final prediction
res = t1 + t2  + b1 + b2
print(f"Final prediction: {res.item()}")
print(f"Parameters: w1={w1.item()}, w2={w2.item()}, b1={b1.item()}, b2={b2.item()}")