use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

trait BinaryOp<T> {
    fn apply(&self, a: T, b: T) -> T;
    fn identity(&self) -> T;
}

struct AddOp;
impl BinaryOp<f32> for AddOp {
    fn apply(&self, a: f32, b: f32) -> f32 {
        a + b
    }
    fn identity(&self) -> f32 {
        0.0
    }
}

impl BinaryOp<f64> for AddOp {
    fn apply(&self, a: f64, b: f64) -> f64 {
        a + b
    }
    fn identity(&self) -> f64 {
        0.0
    }
}

struct MulOp;
impl BinaryOp<f32> for MulOp {
    fn apply(&self, a: f32, b: f32) -> f32 {
        a * b
    }
    fn identity(&self) -> f32 {
        1.0
    }
}

impl BinaryOp<f64> for MulOp {
    fn apply(&self, a: f64, b: f64) -> f64 {
        a * b
    }
    fn identity(&self) -> f64 {
        1.0
    }
}

#[derive(Debug)]
enum Tensor {
    F32 { data: Vec<f32>, shape: Vec<usize> },
    F64 { data: Vec<f64>, shape: Vec<usize> },
}

impl Tensor {
    fn new_f32(data: Vec<f32>, shape: Option<Vec<usize>>) -> Self {
        let shape = match shape {
            Some(shape) => {
                let expected_size = shape.iter().product::<usize>();
                assert_eq!(
                    data.len(),
                    expected_size,
                    "Data length {} doesn't match specified shape {:?} with product {}",
                    data.len(),
                    shape,
                    expected_size
                );
                shape
            }
            None => {
                vec![data.len()]
            }
        };

        Tensor::F32 { data, shape }
    }

    fn new_f64(data: Vec<f64>, shape: Option<Vec<usize>>) -> Self {
        let shape = match shape {
            Some(shape) => {
                let expected_size = shape.iter().product::<usize>();
                assert_eq!(
                    data.len(),
                    expected_size,
                    "Data length {} doesn't match specified shape {:?} with product {}",
                    data.len(),
                    shape,
                    expected_size
                );
                shape
            }
            None => {
                vec![data.len()]
            }
        };

        Tensor::F64 { data, shape }
    }

    fn with_shape_f32(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::new_f32(data, Some(shape))
    }

    fn with_shape_f64(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Self::new_f64(data, Some(shape))
    }

    fn from_vec_f32(data: Vec<f32>) -> Self {
        Self::new_f32(data, None)
    }

    fn from_vec_f64(data: Vec<f64>) -> Self {
        Self::new_f64(data, None)
    }

    fn sum(&self) -> Tensor {
        match self {
            Tensor::F32 { data, shape } => {
                let sum = data.iter().sum();
                let new_data = vec![sum];
                Tensor::with_shape_f32(new_data, vec![1])
            }
            _ => todo!(),
        }
    }

    fn product(&self) -> f64 {
        match self {
            Tensor::F32 { data, .. } => data.iter().fold(1.0f32, |acc, &val| acc * val) as f64,
            Tensor::F64 { data, .. } => data.iter().fold(1.0f64, |acc, &val| acc * val),
        }
    }
    fn concat(&self, other: &Tensor) -> Tensor {
        match &self {
            Tensor::F32 { data, shape } => match &other {
                Tensor::F32 {
                    data: other_data,
                    shape: _,
                } => {
                    let new_data = [data.as_slice(), other_data.as_slice()].concat();
                    Tensor::with_shape_f32(new_data, shape.clone())
                }
                _ => panic!("Cannot concatenate tensors of different types (f32 vs f64)"),
            },
            &Tensor::F64 { data, shape } => todo!("Implement for f64"),
        }
    }
}

macro_rules! impl_tensor_op {
    ($trait:ident, $method:ident, $op:tt, $op_name:expr) => {
        impl $trait for &Tensor {
            type Output = Tensor;

            fn $method(self, other: &Tensor) -> Tensor {
                match (self, other) {
                    (
                        Tensor::F32 {
                            data: data1,
                            shape: shape1,
                        },
                        Tensor::F32 {
                            data: data2,
                            shape: shape2,
                        },
                    ) => {
                        assert_eq!(shape1, shape2, concat!("Shapes must match for ", $op_name));
                        let new_data = data1.iter().zip(data2.iter()).map(|(a, b)| a $op b).collect();
                        Tensor::with_shape_f32(new_data, shape1.clone())
                    }
                    (
                        Tensor::F64 {
                            data: data1,
                            shape: shape1,
                        },
                        Tensor::F64 {
                            data: data2,
                            shape: shape2,
                        },
                    ) => {
                        assert_eq!(shape1, shape2, concat!("Shapes must match for ", $op_name));
                        let new_data = data1.iter().zip(data2.iter()).map(|(a, b)| a $op b).collect();
                        Tensor::with_shape_f64(new_data, shape1.clone())
                    }
                    _ => panic!(concat!("Cannot ", $op_name, " tensors of different types (f32 vs f64)")),
                }
            }
        }

        impl $trait for Tensor {
            type Output = Tensor;

            fn $method(self, other: Tensor) -> Tensor {
                &self $op &other
            }
        }

        // Implementation for tensor op scalar
        impl $trait<f32> for &Tensor {
            type Output = Tensor;

            fn $method(self, scalar: f32) -> Tensor {
                match self {
                    Tensor::F32 { data, shape } => {
                        let new_data = data.iter().map(|&x| x $op scalar).collect();
                        Tensor::with_shape_f32(new_data, shape.clone())
                    }
                    Tensor::F64 { data, shape } => {
                        let new_data = data.iter().map(|&x| x $op scalar as f64).collect();
                        Tensor::with_shape_f64(new_data, shape.clone())
                    }
                }
            }
        }

        impl $trait<f64> for &Tensor {
            type Output = Tensor;

            fn $method(self, scalar: f64) -> Tensor {
                match self {
                    Tensor::F32 { data, shape } => {
                        let new_data = data.iter().map(|&x| x $op scalar as f32).collect();
                        Tensor::with_shape_f32(new_data, shape.clone())
                    }
                    Tensor::F64 { data, shape } => {
                        let new_data = data.iter().map(|&x| x $op scalar).collect();
                        Tensor::with_shape_f64(new_data, shape.clone())
                    }
                }
            }
        }

        impl $trait<f32> for Tensor {
            type Output = Tensor;

            fn $method(self, scalar: f32) -> Tensor {
                &self $op scalar
            }
        }

        impl $trait<f64> for Tensor {
            type Output = Tensor;

            fn $method(self, scalar: f64) -> Tensor {
                &self $op scalar
            }
        }

        // For the reverse operation (scalar op Tensor)
        impl $trait<&Tensor> for f32 {
            type Output = Tensor;

            fn $method(self, tensor: &Tensor) -> Tensor {
                match tensor {
                    Tensor::F32 { data, shape } => {
                        let new_data = data.iter().map(|&x| self $op x).collect();
                        Tensor::with_shape_f32(new_data, shape.clone())
                    }
                    Tensor::F64 { data, shape } => {
                        let new_data = data.iter().map(|&x| self as f64 $op x).collect();
                        Tensor::with_shape_f64(new_data, shape.clone())
                    }
                }
            }
        }

        impl $trait<&Tensor> for f64 {
            type Output = Tensor;

            fn $method(self, tensor: &Tensor) -> Tensor {
                match tensor {
                    Tensor::F32 { data, shape } => {
                        let new_data = data.iter().map(|&x| self as f32 $op x).collect();
                        Tensor::with_shape_f32(new_data, shape.clone())
                    }
                    Tensor::F64 { data, shape } => {
                        let new_data = data.iter().map(|&x| self $op x).collect();
                        Tensor::with_shape_f64(new_data, shape.clone())
                    }
                }
            }
        }

        impl $trait<Tensor> for f32 {
            type Output = Tensor;

            fn $method(self, tensor: Tensor) -> Tensor {
                self $op &tensor
            }
        }

        impl $trait<Tensor> for f64 {
            type Output = Tensor;

            fn $method(self, tensor: Tensor) -> Tensor {
                self $op &tensor
            }
        }
    };
}
impl_tensor_op!(Add, add, +, "add");
impl_tensor_op!(Sub, sub, -, "subtract");
impl_tensor_op!(Mul, mul, *, "multiply");
impl_tensor_op!(Div, div, /, "divide");

fn main() {
    let tensor = Tensor::with_shape_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let res = tensor.sum();
    println!("{:?}", res);
}

fn tests() {}
