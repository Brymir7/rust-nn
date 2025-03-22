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
    // let t1 = Tensor::with_shape_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4], false);
    // let lin_l = LinearLayer::new(4, 4);
    // let lin_2 = LinearLayer::new(4, 2);
    // let wanted = Tensor::with_shape_f32(vec![5.0, 4.0], vec![2], false);
    // let params = vec![lin_l.weights, lin_l.bias, lin_2.weights, lin_2.bias];
    // TENSOR_CONTEXT.with_borrow_mut(|ctx| {
    //     ctx.tensor_cache.start_op_index = ctx.tensor_cache.op_result_pointers.len();
    // });
    // for i in 0..20000 {
    //     let res = lin_l.forward(&t1);
    //     let res = lin_2.forward(&res);
    //     let loss = res.mse(&wanted);
    //     TENSOR_CONTEXT.with_borrow_mut(|ctx| {
    //         ctx.tensor_cache.next_iteration();
    //         for param in &params {
    //             if let Some(tensor) = ctx.get_mut_tensor(*param) {
    //                 match tensor {
    //                     Tensor::F32 { grad, .. } => {
    //                         *grad = None;
    //                     }
    //                     _ => {}
    //                 }
    //             }
    //         }
    //     });
    //     loss.backward();
    //     TENSOR_CONTEXT.with_borrow_mut(|ctx| {
    //         for param in &params {
    //             if let Some(tensor) = ctx.get_mut_tensor(*param) {
    //                 match tensor {
    //                     Tensor::F32 { id, data, grad, .. } => {
    //                         if let Some(grad) = grad {
    //                             let grad_data = grad.data_f32();
    //                             let data = data.mut_data_f32();

    //                             debug_assert!(grad_data.len() == data.len());
    //                             for i in 0..data.len() {
    //                                 data[i] -= grad_data[i] * 0.001;
    //                             }
    //                         }
    //                     }
    //                     _ => todo!(),
    //                 }
    //             }
    //         }
    //     });

    //     if (i + 1) % 100 == 0 {
    //         println!("Loss: {:?}", get_tensor(loss).unwrap().data_f32());
    //     }
    // }

    // let final_result = lin_2.forward(&lin_l.forward(&t1));
    // println!(
    //     "Final result: {:?}",
    //     get_tensor(final_result).unwrap().data_f32()
    // );
    // println!(
    //     "Total used storage: {}",
    //     TENSOR_CONTEXT.with(|ctx| ctx.borrow().all_tensors.len())
    // );
    mnist_example::train_mnist();
}
