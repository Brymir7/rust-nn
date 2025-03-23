use crate::tensor::{Tensor, TensorData, TensorHandle, TENSOR_CONTEXT};

pub trait Optimizer {
    fn step(&self, params: &[TensorHandle]);
    fn zero_grad(&self, params: &[TensorHandle]);
    fn prepare_next_iteration(&self);
}

pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    velocities: std::cell::RefCell<std::collections::HashMap<TensorHandle, TensorData>>,
}

impl SGD {
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        SGD {
            learning_rate,
            momentum,
            velocities: std::cell::RefCell::new(std::collections::HashMap::new()),
        }
    }
}

impl Optimizer for SGD {
    fn zero_grad(&self, params: &[TensorHandle]) {
        TENSOR_CONTEXT.with_borrow_mut(|ctx| {
            for param in params {
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
    }

    fn prepare_next_iteration(&self) {
        TENSOR_CONTEXT.with_borrow_mut(|ctx| {
            ctx.tensor_cache.next_iteration();
        });
    }

    fn step(&self, params: &[TensorHandle]) {
        let mut velocities = self.velocities.borrow_mut();

        TENSOR_CONTEXT.with_borrow_mut(|ctx| {
            for &param in params {
                if let Some(tensor) = ctx.get_mut_tensor(param) {
                    match tensor {
                        Tensor::F32 { data, grad, .. } => {
                            if let Some(grad) = grad {
                                let velocity = velocities.entry(param).or_insert_with(|| {
                                    let shape = data.shape();
                                    let zeros = ndarray::ArrayD::zeros(ndarray::IxDyn(&shape));
                                    TensorData::F32 { data: zeros }
                                });
                                let momentum_term = &*velocity * self.momentum;
                                let grad_term = &*grad * self.learning_rate;
                                let new_velocity = momentum_term - grad_term;
                                *data = data.clone() + new_velocity.clone(); // todo fix this reference stuff
                                *velocity = new_velocity;
                            }
                        }
                        _ => {}
                    }
                }
            }
        });
    }
}
