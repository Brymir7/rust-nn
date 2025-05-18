use std::time::Instant;

use flamer::Flamer;
use optimizer::SGD;
use tensor::{get_tensor, Tensor, TensorHandle};

pub mod data_loader;
pub mod flamer;
pub mod mnist_test;
pub mod optimizer;
pub mod tensor;
fn main() {
    // mnist_test::run_mnist_training();
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
}
