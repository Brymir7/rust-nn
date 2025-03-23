use crate::data_loader::{check_mnist_dataset, load_mnist_dataset, MnistDataset};
use crate::optimizer::{Optimizer, SGD};
use crate::tensor::{get_tensor, Tensor, TensorHandle, TensorOperation, TENSOR_CONTEXT};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::time::Instant;

fn one_hot_encode(labels: &[u8], num_classes: usize) -> TensorHandle {
    let batch_size = labels.len();
    let mut encoded = vec![0.0; batch_size * num_classes];

    for (i, &label) in labels.iter().enumerate() {
        encoded[i * num_classes + label as usize] = 1.0;
    }

    Tensor::with_shape_f32(encoded, vec![batch_size, num_classes], false)
}

fn evaluate_model(
    fc1: &LinearLayer,
    fc2: &LinearLayer,

    dataset: &MnistDataset,
    batch_size: usize,
) -> f32 {
    let mut correct_predictions = 0;
    let mut total_predictions = 0;
    let num_batches = dataset.len() / batch_size;
    let indices: Vec<usize> = (0..dataset.len()).collect();

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * batch_size;
        let batch_end = std::cmp::min(batch_start + batch_size, dataset.len());
        let batch_indices = &indices[batch_start..batch_end];
        let actual_batch_size = batch_indices.len();

        let (batch_images, batch_labels) = dataset.get_batch(batch_indices);

        let hidden1 = fc1.forward(&batch_images);
        let hidden1_activated = hidden1.relu();

        let output = fc2.forward(&hidden1_activated);
        let output_data = get_tensor(output).unwrap();
        let output_data = output_data.data_f32();
        for b in 0..actual_batch_size {
            let mut max_idx = 0;
            let mut max_val = output_data[[b, 0]];
            for j in 1..10 {
                if output_data[[b, j]] > max_val {
                    max_val = output_data[[b, j]];
                    max_idx = j;
                }
            }

            if max_idx == batch_labels[b] as usize {
                correct_predictions += 1;
            }
            total_predictions += 1;
        }
    }

    let accuracy = 100.0 * (correct_predictions as f32) / (total_predictions as f32);
    println!(
        "Test accuracy: {}/{} correct ({:.2}%)",
        correct_predictions, total_predictions, accuracy
    );

    accuracy
}

struct LinearLayer {
    weights: TensorHandle,
    bias: TensorHandle,
}

impl LinearLayer {
    fn new(in_dim: usize, out_dim: usize) -> Self {
        let weights = Tensor::random_f32(vec![in_dim, out_dim], true);
        let bias = Tensor::random_f32(vec![out_dim], true);
        Self { weights, bias }
    }

    fn forward(&self, input: &TensorHandle) -> TensorHandle {
        let input_tensor = get_tensor(*input).unwrap();
        let weights_tensor = get_tensor(self.weights).unwrap();

        let input_data = input_tensor.data_f32();
        let weights_data = weights_tensor.data_f32();

        let input_shape = input_data.shape();

        let result = if input_shape.len() == 1 {
            let a_mat_base = weights_data
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            let a_mat_t = a_mat_base.t();
            let x_vec = input_data
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .unwrap();

            let mut y_vec = ndarray::Array1::zeros(a_mat_t.shape()[0]);
            ndarray::linalg::general_mat_vec_mul(1.0, &a_mat_t, &x_vec, 0.0, &mut y_vec);

            y_vec.into_dyn()
        } else {
            let a_mat = input_data
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            let b_mat = weights_data
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();

            let mut c_mat = ndarray::Array2::zeros((a_mat.shape()[0], b_mat.shape()[1]));
            ndarray::linalg::general_mat_mul(1.0, &a_mat, &b_mat, 0.0, &mut c_mat);
            c_mat.into_dyn()
        };
        let result_handle = Tensor::from_op(result, TensorOperation::None);

        result_handle + self.bias
    }
}

pub fn run_mnist_training() {
    println!("MNIST Training Example");

    let data_dir = match check_mnist_dataset() {
        Ok(dir) => dir,
        Err(e) => {
            println!("Error: {}", e);
            return;
        }
    };

    let (train_dataset, test_dataset) = match load_mnist_dataset(&data_dir, 5) {
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
    let hidden_size_1 = 32;
    let num_classes = 10; // 10 digits (0-9)
    let batch_size = 32;

    let fc1 = LinearLayer::new(input_size, hidden_size_1);
    let fc2 = LinearLayer::new(hidden_size_1, num_classes);

    let params = vec![fc1.weights, fc1.bias, fc2.weights, fc2.bias];

    let learning_rate = 0.0001;
    let momentum = 0.99;
    let optimizer = SGD::new(learning_rate, momentum);

    let num_epochs = 100000;
    let mut rng = thread_rng();
    let num_batches = train_dataset.len() / batch_size;

    let mut indices: Vec<usize> = (0..train_dataset.len()).collect();

    TENSOR_CONTEXT.with_borrow_mut(|ctx| {
        ctx.tensor_cache.start_op_index = ctx.tensor_cache.op_result_pointers.len();
    });

    println!("Starting training for {} epochs...", num_epochs);
    let start_time = Instant::now();

    for epoch in 0..num_epochs {
        indices.shuffle(&mut rng);

        let mut epoch_loss = 0.0;
        let mut correct_predictions = 0;
        let mut total_predictions = 0;

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;
            let batch_indices = &indices[batch_start..batch_start + batch_size];

            let (batch_images, batch_labels) = train_dataset.get_batch(batch_indices);

            optimizer.zero_grad(&params);
            optimizer.prepare_next_iteration();

            let hidden1 = fc1.forward(&batch_images);
            let hidden1_activated = hidden1.relu();
            let output = fc2.forward(&hidden1_activated);
            let fixed_targets = one_hot_encode(&batch_labels, num_classes);

            let loss = output.mse(&fixed_targets);

            loss.backward();
            optimizer.step(&params);

            let loss_val = get_tensor(loss).unwrap().data_f32()[0];
            epoch_loss += loss_val;

            let output_data = get_tensor(output).unwrap();
            let output_data = output_data.data_f32();
            for b in 0..batch_size {
                let mut max_idx = 0;
                let mut max_val = output_data[[b, 0]];
                for j in 1..num_classes {
                    if output_data[[b, j]] > max_val {
                        max_val = output_data[[b, j]];
                        max_idx = j;
                    }
                }

                if max_idx == batch_labels[b] as usize {
                    correct_predictions += 1;
                }
                total_predictions += 1;
            }
        }

        let avg_loss = epoch_loss / (num_batches as f32);
        let train_accuracy = 100.0 * (correct_predictions as f32) / (total_predictions as f32);

        println!(
            "Epoch {}/{} completed - Avg Loss: {:.4} - Train Acc: {:.2}%",
            epoch + 1,
            num_epochs,
            avg_loss,
            train_accuracy
        );

        evaluate_model(&fc1, &fc2, &test_dataset, batch_size);
    }

    let training_duration = start_time.elapsed();
    println!("Training completed in {:.2?}", training_duration);

    println!("\nFinal evaluation:");
    evaluate_model(&fc1, &fc2, &test_dataset, batch_size);

    println!(
        "Total used storage: {}",
        TENSOR_CONTEXT.with(|ctx| ctx.borrow().all_tensors.len())
    );
}
