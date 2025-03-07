import torch

# Initialize tensors with requires_grad=True for parameters we want to optimize
t1 = torch.tensor([1.0], requires_grad=False)
t2 = torch.tensor([1.0], requires_grad=False)
w1 = torch.tensor([0.5], requires_grad=True)
w2 = torch.tensor([0.5], requires_grad=True)
b1 = torch.tensor([0.5], requires_grad=True)
b2 = torch.tensor([0.5], requires_grad=True)
wanted = torch.tensor([4.0])

# Compute forward pass
res = t1 * w1 + b1 + t2 * w2 + b2
loss = (wanted - res).abs()

# Zero gradients before backward pass
if w1.grad is not None:
    w1.grad.zero_()
if w2.grad is not None:
    w2.grad.zero_()
if b1.grad is not None:
    b1.grad.zero_()
if b2.grad is not None:
    b2.grad.zero_()

# Compute gradients
loss.backward()

# Print gradients
print("Gradients after backward pass:")
print(f"w1.grad: {w1.grad.item()}")
print(f"w2.grad: {w2.grad.item()}")
print(f"b1.grad: {b1.grad.item()}")
print(f"b2.grad: {b2.grad.item()}")