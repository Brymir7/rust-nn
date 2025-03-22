use tensor::{get_tensor, Tensor, TensorHandle, TENSOR_CONTEXT};
pub mod tensor;
pub mod utils;
pub mod mnist_example;
pub mod data_loader;
pub mod optimizer;
struct LinearLayer {
    weights: TensorHandle,
    bias: TensorHandle,
}
impl LinearLayer {
    fn new(in_dim: usize, out_dim: usize) -> Self {
        let weights = Tensor::random_f32(vec![in_dim, out_dim], true);
        let bias = Tensor::random_f32(vec![out_dim], true);
        Self {
            weights: weights,
            bias: bias,
        }
    }

    fn forward(&self, input: &TensorHandle) -> TensorHandle {
        let weights_tensor = get_tensor(self.weights).unwrap();
        let weights_shape = weights_tensor.shape();
        let in_features = weights_shape[0];
        let out_features = weights_shape[1];

        let mut result: Option<TensorHandle> = None;
        for j in 0..out_features {
            let weights_j = Tensor::with_shape_f32(
                (0..in_features)
                    .map(|i| weights_tensor.data_f32()[i * out_features + j])
                    .collect(),
                vec![in_features],
                false,
            );
            let mul_result = input * &weights_j;
            let sum_result = mul_result.sum();
            if let Some(prev_result) = result {
                result = Some(prev_result.concat(&sum_result, Some(0)));
            } else {
                result = Some(sum_result);
            }
        }

        result.unwrap() + self.bias
    }
}
fn main() {
    println!("Starting one-hot overfitting example...");

    // Create a random input (simulating an MNIST image with 784 features)
    let input_size = 784;
    let hidden_size = 128;
    let output_size = 10;
    
    // Generate random input vector
    let mut random_input = Vec::with_capacity(input_size);
    for _ in 0..input_size {
        random_input.push(rand::random::<f32>());
    }
    
    let input_tensor = Tensor::with_shape_f32(random_input, vec![input_size], false);
    
    // Create a target tensor with one-hot encoding for digit 5
    let mut target_data = vec![0.0; output_size];
    target_data[5] = 1.0; // Set the 6th element (index 5) to 1.0
    let target = Tensor::with_shape_f32(target_data.clone(), vec![output_size], false);
    
    println!("Target one-hot vector: {:?}", target_data);
    
    // Create a simple network with two layers
    let fc1 = LinearLayer::new(input_size, hidden_size);
    let fc2 = LinearLayer::new(hidden_size, output_size);
    
    // Collect parameters for optimization
    let params = vec![fc1.weights, fc1.bias, fc2.weights, fc2.bias];
    
    // Set learning rate
    let learning_rate = 0.01;
    
    // Initialize tensor context
    TENSOR_CONTEXT.with_borrow_mut(|ctx| {
        ctx.tensor_cache.start_op_index = ctx.tensor_cache.op_result_pointers.len();
    });
    
    // Training loop
    for i in 0..2000 {
        // Forward pass
        let hidden = fc1.forward(&input_tensor);
        let output = fc2.forward(&hidden);
        let loss = output.mse(&target);
        
        // Reset gradients and prepare for backprop
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

                                debug_assert!(grad_data.len() == data.len());
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

        // Print debug information every 50 epochs
        if (i + 1) % 50 == 0 || i < 5 {
            let loss_val = get_tensor(loss).unwrap().data_f32()[0];
            let output_data = get_tensor(output).unwrap().data_f32().to_vec();
            
            // Find predicted class
            let mut max_idx = 0;
            let mut max_val = output_data[0];
            for j in 1..output_size {
                if output_data[j] > max_val {
                    max_val = output_data[j];
                    max_idx = j;
                }
            }
            
            let is_correct = max_idx == 5; // Target is digit 5
            
            println!(
                "Epoch {} - Loss: {:.6} - Prediction: {} (Correct: {})",
                i + 1,
                loss_val,
                max_idx,
                is_correct
            );
            
            // Calculate and print MSE manually
            let mut squared_error_sum = 0.0;
            for j in 0..output_size {
                let error = target_data[j] - output_data[j];
                squared_error_sum += error * error;
            }
            let manual_mse = squared_error_sum / output_size as f32;
            
            println!("Target vector: {:?}", target_data);
            println!("Output values: {:?}", output_data);
            println!("MSE calculated: {:.6}, Framework MSE: {:.6}", manual_mse, loss_val);
            
            // Early stopping if we've successfully overfit
            if loss_val < 0.0001 {
                println!("Successfully overfit the example at epoch {}!", i + 1);
                break;
            }
        }
    }

    let final_output = fc2.forward(&fc1.forward(&input_tensor));
    let final_output_data = get_tensor(final_output).unwrap().data_f32().to_vec();
    
    println!("Final output: {:?}", final_output_data);
    println!(
        "Total used storage: {}",
        TENSOR_CONTEXT.with(|ctx| ctx.borrow().all_tensors.len())
    );
}