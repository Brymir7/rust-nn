use crate::data_loader::{check_mnist_dataset, load_mnist_dataset, MnistDataset};
use crate::optimizer::{Optimizer, SGD};
use crate::tensor::{get_tensor, Tensor, TensorHandle, TensorOperation, TENSOR_CONTEXT};
use crate::LinearLayer;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashSet;
use std::time::Instant;

pub fn run_mnist_training() {
    println!("MNIST Training Example");

    let data_dir = match check_mnist_dataset() {
        Ok(dir) => dir,
        Err(e) => {
            println!("Error: {}", e);
            return;
        }
    };

    let (train_dataset, test_dataset) = match load_mnist_dataset(&data_dir, 100) {
        Ok(data) => data,
        Err(e) => {
            println!("Error loading dataset: {}", e);
            return;
        }
    };

    println!(
        "Dataset loaded: {} training examples, {} test examples",
        train_dataset.len(),
        test_dataset.len()
    );

    let input_size = 784; // 28x28 pixels
    let hidden_size_1 = 1024;
    let num_classes = 10; // 10 digits (0-9)

    let fc1 = LinearLayer::new(input_size, hidden_size_1);
    let fc2 = LinearLayer::new(hidden_size_1, num_classes);

    let params = vec![fc1.weights, fc1.bias, fc2.weights, fc2.bias];

    let learning_rate = 0.01;
    let momentum = 0.90;
    let optimizer = SGD::new(learning_rate, momentum);

    let num_epochs = 10;
    let mut rng = thread_rng();

    let mut indices: Vec<usize> = (0..train_dataset.len()).collect();

    println!("Starting training for {} epochs...", num_epochs);
    let start_time = Instant::now();

    let flatten_batch = |tensor: &TensorHandle| -> TensorHandle {
        let t = get_tensor(*tensor).unwrap();
        let data = t.data_f32();
        Tensor::with_shape_f32(data.iter().map(|&x| x).collect(), vec![t.shape()[1]], true)
    };
    TENSOR_CONTEXT.with_borrow_mut(|ctx| {
        ctx.tensor_cache.start_op_index = ctx.tensor_cache.op_result_pointers.len();
    });
    for epoch in 0..num_epochs {
        indices.shuffle(&mut rng);

        let mut epoch_loss = 0.0;
        let mut correct_predictions = 0;
        let mut total_predictions = 0;

        for (i, &idx) in indices.iter().enumerate() {
            optimizer.zero_grad(&params);
            optimizer.prepare_next_iteration();

            let single_index = &[idx];
            let (batch_image, batch_label) = train_dataset.get_batch(single_index);
            let image = flatten_batch(&batch_image);
            let label = batch_label[0];
            let mut target_data = vec![0.0; num_classes];
            target_data[label as usize] = 1.0;
            let target = Tensor::with_shape_f32(target_data, vec![num_classes], false);

            let hidden1 = fc1.forward(&image);
            let hidden1_activated = hidden1.relu();
            let output = fc2.forward(&hidden1_activated);
            let loss = output.mse(&target);
            // print_computation_graph(loss, &mut HashSet::new(), 0);
            loss.backward();

            optimizer.step(&params);

            let loss_val = get_tensor(loss).unwrap().data_f32()[0];
            epoch_loss += loss_val;

            if i % 100 == 0 {
                println!("Epoch {}, Sample {}: Loss = {:.6}", epoch + 1, i, loss_val);
            }

            let output_data = get_tensor(output).unwrap();
            let output_data = output_data.data_f32();
            let mut max_idx = 0;
            let mut max_val = output_data[0];
            for j in 1..num_classes {
                if output_data[j] > max_val {
                    max_val = output_data[j];
                    max_idx = j;
                }
            }

            if max_idx == label as usize {
                correct_predictions += 1;
            }
            total_predictions += 1;
        }

        let avg_loss = epoch_loss / (train_dataset.len() as f32);
        let train_accuracy = 100.0 * (correct_predictions as f32) / (total_predictions as f32);

        println!(
            "Epoch {}/{} completed - Avg Loss: {:.6} - Train Acc: {:.2}%",
            epoch + 1,
            num_epochs,
            avg_loss,
            train_accuracy
        );

        evaluate_model_1d(&fc1, &fc2, &test_dataset, &flatten_batch);
    }

    let training_duration = start_time.elapsed();
    println!("Training completed in {:.2?}", training_duration);
}

/// Print the computation graph starting from a specific tensor
fn print_computation_graph(handle: TensorHandle, visited: &mut HashSet<usize>, indent: usize) {
    if !visited.insert(handle.0) {
        println!(
            "{:indent$}Tensor({}) [already visited]",
            "",
            handle.0,
            indent = indent
        );
        return;
    }

    let tensor = match get_tensor(handle) {
        Some(t) => t,
        None => {
            println!(
                "{:indent$}Tensor({}) [not found]",
                "",
                handle.0,
                indent = indent
            );
            return;
        }
    };

    let shape_str = tensor
        .shape()
        .iter()
        .map(|&d| d.to_string())
        .collect::<Vec<_>>()
        .join("×");
    let requires_grad = tensor.requires_grad();

    println!(
        "{:indent$}Tensor({}) [shape={}, requires_grad={}]",
        "",
        handle.0,
        shape_str,
        requires_grad,
        indent = indent
    );

    // Print gradient info if available
    match tensor.clone() {
        Tensor::F32 { grad, .. } => {
            if let Some(_) = grad {
                println!("{:indent$}     Has gradient", "", indent = indent + 2);
            } else {
                println!("{:indent$}     No gradient", "", indent = indent + 2);
            }
        }
        _ => {}
    }

    // Check what operation created this tensor
    match tensor.graph() {
        TensorOperation::Add { left, right } => {
            println!("{:indent$}     Operation: Add", "", indent = indent + 2);
            println!("{:indent$}         Left:", "", indent = indent + 2);
            print_computation_graph(left, visited, indent + 4);
            println!("{:indent$}         Right:", "", indent = indent + 2);
            // print_computation_graph(right, visited, indent+4);
        }
        TensorOperation::Sub { left, right } => {
            println!(
                "{:indent$}     Operation: Subtract",
                "",
                indent = indent + 2
            );
            println!("{:indent$}         Left:", "", indent = indent + 2);
            print_computation_graph(left, visited, indent + 4);
            println!("{:indent$}         Right:", "", indent = indent + 2);
            // print_computation_graph(right, visited, indent+4);
        }
        TensorOperation::Mul { left, right } => {
            println!(
                "{:indent$}     Operation: Multiply",
                "",
                indent = indent + 2
            );
            println!("{:indent$}         Left:", "", indent = indent + 2);
            print_computation_graph(left, visited, indent + 4);
            println!("{:indent$}         Right:", "", indent = indent + 2);
            // print_computation_graph(right, visited, indent+4);
        }
        TensorOperation::Div { left, right } => {
            println!("{:indent$}     Operation: Divide", "", indent = indent + 2);
            println!("{:indent$}         Left:", "", indent = indent + 2);
            print_computation_graph(left, visited, indent + 4);
            println!("{:indent$}         Right:", "", indent = indent + 2);
            // print_computation_graph(right, visited, indent+4);
        }
        TensorOperation::MatMul { input, weights } => {
            println!("{:indent$}     Operation: MatMul", "", indent = indent + 2);
            println!("{:indent$}         Input:", "", indent = indent + 2);
            print_computation_graph(input, visited, indent + 4);
            println!("{:indent$}         Weights:", "", indent = indent + 2);
            print_computation_graph(weights, visited, indent + 4);
        }
        TensorOperation::Max {
            input,
            threshold,
            mask,
        } => {
            println!("{:indent$}     Operation: Max", "", indent = indent + 2);
            println!("{:indent$}         Input:", "", indent = indent + 2);
            print_computation_graph(input, visited, indent + 4);
            println!(
                "{:indent$}         Threshold: {}",
                "",
                threshold,
                indent = indent + 2
            );

            // println!("{:indent$}         Mask:", "", indent = indent + 2);
            // print_computation_graph(mask, visited, indent + 4);
        }
        TensorOperation::Sum { input } => {
            println!("{:indent$}     Operation: Sum", "", indent = indent + 2);
            println!("{:indent$}         Input:", "", indent = indent + 2);
            print_computation_graph(input, visited, indent + 4);
        }
        _ => {
            println!("otrher op {:?}", tensor.graph());
        }
    }
}

fn evaluate_model_1d(
    fc1: &LinearLayer,
    fc2: &LinearLayer,
    dataset: &MnistDataset,
    flatten_batch: &dyn Fn(&TensorHandle) -> TensorHandle,
) -> f32 {
    let mut correct_predictions = 0;
    let mut total_predictions = 0;

    for i in 0..dataset.len() {
        let single_index = &[i];
        let (batch_image, batch_label) = dataset.get_batch(single_index);
        let image = flatten_batch(&batch_image);
        let label = batch_label[0];

        let hidden1 = fc1.forward(&image);
        let hidden1_activated = hidden1.relu();
        let output = fc2.forward(&hidden1_activated);

        let output_data = get_tensor(output).unwrap();
        let output_data = output_data.data_f32();
        let mut max_idx = 0;
        let mut max_val = output_data[0];
        for j in 1..10 {
            if output_data[j] > max_val {
                max_val = output_data[j];
                max_idx = j;
            }
        }

        if max_idx == label as usize {
            correct_predictions += 1;
        }
        total_predictions += 1;
    }

    let accuracy = 100.0 * (correct_predictions as f32) / (total_predictions as f32);
    println!(
        "Test accuracy: {}/{} correct ({:.2}%)",
        correct_predictions, total_predictions, accuracy
    );

    accuracy
}
