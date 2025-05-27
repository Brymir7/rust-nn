# Rust Neural Network Library

A basic neural network library implemented in Rust that provides automatic differentiation and tensor operations on CPU.

## Features

- Automatic differentiation through dynamic computational graph construction
- Tensor operations with broadcasting support 
- Basic neural network layers (Linear/Dense)
- Common activation functions (ReLU, Softmax)
- Loss functions (MSE, Cross Entropy)
- SGD optimizer with momentum
- MNIST dataset loader and example training (trains MNIST to 99%, mnist_test::run_mnist_training())

## Implementation Details

The library implements reverse-mode automatic differentiation by:

Ops = Add(Tensor, Tensor), Sub(Tensor, Tensor)...
1. Storing what Ops created the Tensor in Tensor.graph
2. Visiting all nodes of the graph starting from Loss.graph and applying gradient based on OpType, then visiting OpType's tensors

## Usage

Basic example of training a model:

```rust
let a = Tensor::new_f32(vec![1.0, 2.0, 3.0], None, true);
let w = Tensor::new_f32(vec![0.5, 0.5, 0.5], None, true);
let target = Tensor::new_f32(vec![2.0, 4.0, 6.0], None, false);
let learning_rate = 0.1;
let momentum = 0.90;
let mut optimizer = SGD::new(learning_rate, momentum);
let num_epochs = 50;
println!("Starting training for {} epochs...", num_epochs);
let params = vec![a, w];
let flamer = Flamer::new();
for _ in 0..num_epochs {
    flamer.training_step(&mut optimizer, &params, || {
        let output = a * w;
        let loss = output.mse(&target);
        loss.backward();
        println!("Loss {loss}");
    });
}
```