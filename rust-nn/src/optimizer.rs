use crate::tensor::{Tensor, TensorHandle, TENSOR_CONTEXT};

pub trait Optimizer {
    fn step(&self, params: &[TensorHandle]);
    fn zero_grad(&self, params: &[TensorHandle]);
}

pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    velocities: std::cell::RefCell<std::collections::HashMap<TensorHandle, Vec<f32>>>,
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

    fn step(&self, params: &[TensorHandle]) {
        let mut velocities = self.velocities.borrow_mut();
        
        TENSOR_CONTEXT.with_borrow_mut(|ctx| {
            for &param in params {
                if let Some(tensor) = ctx.get_mut_tensor(param) {
                    match tensor {
                        Tensor::F32 { data, grad, .. } => {
                            if let Some(grad) = grad {
                                let grad_data = grad.data_f32();
                                let data_mut = data.mut_data_f32();
                                
                                // Get or initialize velocity for this parameter
                                let velocity = velocities
                                    .entry(param)
                                    .or_insert_with(|| vec![0.0; data_mut.len()]);
                                
                                for i in 0..data_mut.len() {
                                    // Update velocity with momentum
                                    velocity[i] = self.momentum * velocity[i] - 
                                                 self.learning_rate * grad_data[i];
                                    // Apply update
                                    data_mut[i] += velocity[i];
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        });
    }
}