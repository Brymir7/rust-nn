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
            // fix this is reallocating every loop
            let mut c_mat = ndarray::Array2::zeros((a_mat.shape()[0], b_mat.shape()[1]));
            ndarray::linalg::general_mat_mul(1.0, &a_mat, &b_mat, 0.0, &mut c_mat);
            c_mat.into_dyn()
        };

        let result_handle = Tensor::from_op(
            result.clone(),
            TensorOperation::MatMul {
                input: input.clone(),
                weights: self.weights,
            },
        );

        // Add bias
        result_handle + self.bias
    }
}
fn main() {
    mnist_test::run_mnist_training();
}
