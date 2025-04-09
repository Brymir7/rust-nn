pub mod data_loader;
pub mod flamer;
pub mod mnist_test;
pub mod optimizer;
pub mod tensor;
fn main() {
    mnist_test::run_mnist_training();
}
