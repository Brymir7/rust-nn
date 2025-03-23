use ndarray::Axis;
use optimizer::{Optimizer, SGD};
use tensor::{get_tensor, Tensor, TensorHandle, TensorOperation, TENSOR_CONTEXT};
pub mod data_loader;
pub mod mnist_test;
pub mod optimizer;
pub mod tensor;
// pub mod utils;
pub struct LinearLayer {
    pub weights: TensorHandle,
    pub bias: TensorHandle,
}
impl LinearLayer {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let weights = Tensor::random_f32(vec![in_dim, out_dim], true);
        let bias = Tensor::random_f32(vec![out_dim], true);
        Self { weights, bias }
    }

    pub fn forward(&self, input: &TensorHandle) -> TensorHandle {
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

            let mut y_vec = ndarray::Array1::zeros(a_mat_t.shape()[0]); // todo use Tensor / tensordata to cache alloc
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

            let mut c_mat = ndarray::Array2::zeros((a_mat.shape()[0], b_mat.shape()[1])); // todo use Tensor / tensordata to cache alloc
            ndarray::linalg::general_mat_mul(1.0, &a_mat, &b_mat, 0.0, &mut c_mat);
            c_mat.into_dyn()
        };
        let result_handle = Tensor::from_op(result, TensorOperation::None);

        result_handle + self.bias
    }
}
fn main() {
    // println!("Starting batch overfitting example...");

    // let input_size = 784;
    // let hidden_size = 128;
    // let output_size = 10;
    // let batch_size = 5;

    // let mut random_inputs = Vec::with_capacity(batch_size * input_size);
    // for _ in 0..(batch_size * input_size) {
    //     random_inputs.push(rand::random::<f32>());
    // }
    // let input_tensor = Tensor::with_shape_f32(random_inputs, vec![batch_size, input_size], false);

    // let mut target_data = vec![0.0; batch_size * output_size];
    // let target_digits = [0, 2, 5, 7, 9]; // Using digits 0, 2, 5, 7, 9

    // for i in 0..batch_size {
    //     let digit = target_digits[i];
    //     target_data[i * output_size + digit] = 1.0;
    // }

    // let target = Tensor::with_shape_f32(target_data.clone(), vec![batch_size, output_size], false);
    // println!("Target digits: {:?}", target_digits);

    // let fc1 = LinearLayer::new(input_size, hidden_size);
    // let fc2 = LinearLayer::new(hidden_size, output_size);
    // let params = vec![fc1.weights, fc1.bias, fc2.weights, fc2.bias];

    // let learning_rate = 0.001;
    // let momentum = 0.95;
    // let optimizer = SGD::new(learning_rate, momentum);

    // // avoid caching these vec allocs as they never get rebuilt
    // TENSOR_CONTEXT.with_borrow_mut(|ctx| {
    //     ctx.tensor_cache.start_op_index = ctx.tensor_cache.op_result_pointers.len();
    // });

    // for i in 0..100000 {
    //     optimizer.zero_grad(&params);
    //     optimizer.prepare_next_iteration();
    //     let hidden = fc1.forward(&input_tensor);
    //     let output = fc2.forward(&hidden);
    //     let loss = output.softmax().mse(&target);

    //     loss.backward();

    //     optimizer.step(&params);

    //     if (i + 1) % 100 == 0 || i < 5 {
    //         let loss_val = get_tensor(loss).unwrap().data_f32()[0];
    //         let output_data = get_tensor(output).unwrap();
    //         let output_data = output_data.data_f32();

    //         let mut correct_count = 0;
    //         println!("Epoch {} - Loss: {:.6}", i + 1, loss_val);

    //         for b in 0..batch_size {
    //             let mut max_idx = 0;
    //             let mut max_val = output_data[[b, 0]];
    //             for j in 1..output_size {
    //                 if output_data[[b, j]] > max_val {
    //                     max_val = output_data[[b, j]];
    //                     max_idx = j;
    //                 }
    //             }

    //             let target_digit = target_digits[b];
    //             let is_correct = max_idx == target_digit;
    //             if is_correct {
    //                 correct_count += 1;
    //             }

    //             println!(
    //                 "  Example {} - Target: {}, Prediction: {} ({})",
    //                 b,
    //                 target_digit,
    //                 max_idx,
    //                 if is_correct { "✓" } else { "✗" }
    //             );
    //         }

    //         println!("Accuracy: {}/{} correct", correct_count, batch_size);

    //         if correct_count == batch_size {
    //             println!("Successfully achieved 100% accuracy at epoch {}!", i + 1);
    //             break;
    //         }

    //         if loss_val < 0.0001 {
    //             println!("Successfully overfit all examples at epoch {}!", i + 1);
    //             break;
    //         }
    //     }
    // }

    // let final_output = fc2.forward(&fc1.forward(&input_tensor));
    // let final_output_data = get_tensor(final_output).unwrap();
    // let output_data = final_output_data.data_f32();

    // println!("\nFinal predictions:");
    // for b in 0..batch_size {
    //     let mut predictions = Vec::with_capacity(output_size);
    //     for j in 0..output_size {
    //         predictions.push(output_data[[b, j]]);
    //     }
    //     println!(
    //         "Example {} (target digit {}): {:?}",
    //         b, target_digits[b], predictions
    //     );
    // }

    // println!(
    //     "Total used storage: {}",
    //     TENSOR_CONTEXT.with(|ctx| ctx.borrow().all_tensors.len())
    // );
    mnist_test::run_mnist_training();
}
