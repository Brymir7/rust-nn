use crate::data_loader::{check_mnist_dataset, load_mnist_dataset};
use crate::tensor::{get_tensor, Tensor, TensorHandle, TENSOR_CONTEXT};
use crate::LinearLayer;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::time::Instant;

pub fn train_mnist() -> Result<(), Box<dyn std::error::Error>> {
    // Check for MNIST dataset
    println!("Checking for MNIST dataset...");
    let data_dir = check_mnist_dataset()?;

    println!("Loading MNIST dataset...");
    let (train_dataset, test_dataset) = load_mnist_dataset(&data_dir)?;

    println!("Training set size: {}", train_dataset.len());
    println!("Test set size: {}", test_dataset.len());

    // Create a simple model with two linear layers
    // Input size: 784 (28x28 images)
    // Output size: 10 (digits 0-9)
    let fc1 = LinearLayer::new(784, 10);

    // Collect model parameters for optimization
    let params = vec![fc1.weights, fc1.bias];

    let batch_size = 1;
    let num_epochs = 5;
    let learning_rate = 0.01;

    // Prepare for training
    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..train_dataset.len()).collect();
    let start_time = Instant::now();

    // Training loop
    for epoch in 0..num_epochs {
        // Shuffle indices for this epoch
        indices.shuffle(&mut rng);

        let mut total_loss = 0.0;
        let mut num_correct = 0;
        let num_batches = train_dataset.len() / batch_size;

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;
            let batch_indices = &indices[batch_start..batch_start + batch_size];

            let (batch_x, batch_labels) = train_dataset.get_batch(batch_indices);

            // Reset operation tracking
            TENSOR_CONTEXT.with_borrow_mut(|ctx| {
                ctx.tensor_cache.start_op_index = ctx.tensor_cache.op_result_pointers.len();
            });

            // Forward pass
            let output = fc1.forward(&batch_x.flatten());

            // Compute loss (using MSE)
            // Create one-hot encoded target tensors
            let mut target_data = vec![0.0; batch_size * 10];
            for (i, &label) in batch_labels.iter().enumerate() {
                target_data[i * 10 + label as usize] = 1.0;
            }
            let target = Tensor::with_shape_f32(target_data, vec![batch_size, 10], false).flatten();

            let loss = output.mse(&target);

            // Reset gradients
            TENSOR_CONTEXT.with_borrow_mut(|ctx| {
                ctx.tensor_cache.next_iteration();
                for param in &params {
                    if let Some(tensor) = ctx.get_mut_tensor(*param) {
                        match tensor {
                            Tensor::F32 { grad, .. } => {
                                *grad = None;
                            }
                            _ => {}
                        }
                    }
                }
            });

            // Backward pass
            loss.backward();

            // Update parameters
            TENSOR_CONTEXT.with_borrow_mut(|ctx| {
                for param in &params {
                    if let Some(tensor) = ctx.get_mut_tensor(*param) {
                        match tensor {
                            Tensor::F32 { data, grad, .. } => {
                                if let Some(grad) = grad {
                                    let grad_data = grad.data_f32();
                                    let data = data.mut_data_f32();

                                    for i in 0..data.len() {
                                        data[i] -= grad_data[i] * learning_rate;
                                    }
                                }
                            }
                            _ => todo!(),
                        }
                    }
                }
            });

            // Calculate accuracy
            let output_data = get_tensor(output).unwrap().data_f32().to_vec();
            for i in 0..batch_size {
                let mut max_idx = 0;
                let mut max_val = output_data[i * 10];
                for j in 1..10 {
                    let idx = i * 10 + j;
                    if output_data[idx] > max_val {
                        max_val = output_data[idx];
                        max_idx = j;
                    }
                }

                if max_idx == batch_labels[i] as usize {
                    num_correct += 1;
                }
            }

            let loss_val = get_tensor(loss).unwrap().data_f32()[0];
            total_loss += loss_val;

            if (batch_idx + 1) % 100 == 0 {
                println!(
                    "Epoch {}/{} - Batch {}/{} - Loss: {:.4} - Accuracy: {:.2}%",
                    epoch + 1,
                    num_epochs,
                    batch_idx + 1,
                    num_batches,
                    total_loss / (batch_idx + 1) as f32,
                    100.0 * num_correct as f32 / ((batch_idx + 1) * batch_size) as f32
                );
            }
        }

        let elapsed = start_time.elapsed().as_secs_f32();
        println!(
            "Epoch {}/{} completed - Loss: {:.4} - Accuracy: {:.2}% - Time: {:.2}s",
            epoch + 1,
            num_epochs,
            total_loss / num_batches as f32,
            100.0 * num_correct as f32 / (num_batches * batch_size) as f32,
            elapsed
        );
    }

    // Evaluate on test set
    println!("\nEvaluating on test set...");

    let test_batch_size = 100;
    let num_test_batches = test_dataset.len() / test_batch_size;
    let mut test_correct = 0;

    for batch_idx in 0..num_test_batches {
        let batch_indices: Vec<usize> = (0..test_batch_size)
            .map(|i| batch_idx * test_batch_size + i)
            .collect();

        let (batch_x, batch_labels) = test_dataset.get_batch(&batch_indices);

        // Forward pass (without tracking gradients)
        let output = fc1.forward(&batch_x);

        // Calculate accuracy
        let output_data = get_tensor(output).unwrap().data_f32().to_vec();
        for i in 0..test_batch_size {
            let mut max_idx = 0;
            let mut max_val = output_data[i * 10];
            for j in 1..10 {
                let idx = i * 10 + j;
                if output_data[idx] > max_val {
                    max_val = output_data[idx];
                    max_idx = j;
                }
            }

            if max_idx == batch_labels[i] as usize {
                test_correct += 1;
            }
        }
    }

    let test_accuracy = 100.0 * test_correct as f32 / (num_test_batches * test_batch_size) as f32;
    println!("Test Accuracy: {:.2}%", test_accuracy);

    println!(
        "Total used storage: {}",
        TENSOR_CONTEXT.with(|ctx| ctx.borrow().all_tensors.len())
    );

    Ok(())
}
