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

enum Tensor {
    F32 { data: Vec<f32>, shape: Vec<usize> },
    F64 { data: Vec<f64>, shape: Vec<usize> },
}
impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tensor::F32 { data, shape } => {
                write!(f, "Tensor::F32 {{\n")?;
                format_tensor(f, data, shape, 0)?;
                write!(f, "\nshape: {:?}\n}}", shape)
            }
            Tensor::F64 { data, shape } => {
                write!(f, "Tensor::F64 {{\n")?;
                format_tensor(f, data, shape, 0)?;
                write!(f, "\nshape: {:?}\n}}", shape)
            }
        }
    }
}

fn format_tensor<T: Debug>(
    f: &mut std::fmt::Formatter<'_>,
    data: &[T],
    shape: &[usize],
    depth: usize,
) -> std::fmt::Result {
    let indent = "  ".repeat(depth + 1);

    if shape.len() <= 1 {
        return write!(f, "{}[{:?}]", indent, data);
    }
    write!(f, "{}[", indent)?;

    let dim_size = shape[0];
    let sub_dim_size: usize = shape[1..].iter().product();

    for i in 0..dim_size {
        if i > 0 {
            write!(f, ",\n")?;
        } else {
            write!(f, "\n")?;
        }

        let start = i * sub_dim_size;
        let end = start + sub_dim_size;
        format_tensor(f, &data[start..end], &shape[1..], depth + 1)?;
    }

    write!(f, "\n{}]", indent)
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

    fn concat(&self, other: &Tensor, dim: Option<usize>) -> Tensor {
        match &self {
            Tensor::F32 { data, shape } => match &other {
                Tensor::F32 {
                    data: other_data,
                    shape: other_shape,
                } => {
                    match dim {
                        // dim = 0
                        None => {
                            assert_eq!(
                                shape[0], other_shape[0],
                                "Shapes must match for concatenation"
                            );
                            let mut new_data = data.clone();
                            new_data.extend(other_data);
                            let mut new_shape = shape.clone();
                            new_shape[0] += other_shape[0];
                            Tensor::with_shape_f32(new_data, new_shape)
                        }
                        Some(dim) => {
                            for (i, (a, b)) in shape.iter().zip(other_shape.iter()).enumerate() {
                                if i == dim {
                                    continue;
                                } else {
                                    assert_eq!(a, b, "Shapes must match for concatenation");
                                }
                            }
                            let mut new_shape = shape.clone();
                            new_shape[dim] += other_shape[dim];
                            let mut new_data = Vec::with_capacity(new_shape.iter().product());
                            let outer_dim = shape[..dim].iter().product();
                            let inner_dim: usize = shape[dim + 1..].iter().product();
                            let self_dim = shape[dim];
                            let other_dim = other_shape[dim];

                            for outer in 0..outer_dim {
                                let outer_offset = outer * self_dim * inner_dim;
                                for d1 in 0..self_dim {
                                    let start = outer_offset + d1 * inner_dim;
                                    for k in 0..inner_dim {
                                        new_data.push(data[start + k]);
                                    }
                                }
                                let outer_offset = outer * other_dim * inner_dim;
                                for d2 in 0..other_dim {
                                    let start = outer_offset + d2  * inner_dim;
                                    for k in 0..inner_dim {
                                        new_data.push(other_data[start + k]);
                                    }
                                }
                            }

                            Tensor::with_shape_f32(new_data, new_shape)
                        }
                    }
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
    let tensor = Tensor::with_shape_f32(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3, 2],
    );
    let tensor2 = Tensor::with_shape_f32(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3, 2],
    );
    let res = tensor.concat(&tensor2, Some(1));
    println!("{:?}", res);

    tests();
}

fn tests() {
    {
        let t1 = Tensor::with_shape_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::with_shape_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let result = t1.concat(&t2, None);

        match result {
            Tensor::F32 { data, shape } => {
                assert_eq!(
                    shape,
                    vec![4, 2],
                    "Shape should be [4, 2] when concatenating along default dimension"
                );
                assert_eq!(
                    data,
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    "Data should be concatenated correctly"
                );
            }
            _ => panic!("Expected F32 tensor result"),
        }
    }

    {
        let t1 = Tensor::with_shape_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::with_shape_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let result = t1.concat(&t2, Some(0));

        match result {
            Tensor::F32 { data, shape } => {
                assert_eq!(
                    shape,
                    vec![4, 2],
                    "Shape should be [4, 2] when concatenating along dim 0"
                );
                assert_eq!(
                    data,
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    "Data should be concatenated correctly"
                );
            }
            _ => panic!("Expected F32 tensor result"),
        }
    }

    {
        let t1 = Tensor::with_shape_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::with_shape_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let result = t1.concat(&t2, Some(1));

        match result {
            Tensor::F32 { data, shape } => {
                assert_eq!(
                    shape,
                    vec![2, 4],
                    "Shape should be [2, 4] when concatenating along dim 1"
                );
                assert_eq!(
                    data,
                    vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0],
                    "Data should be interleaved correctly when concatenating along dim 1"
                );
            }
            _ => panic!("Expected F32 tensor result"),
        }
    }

    {
        let t1 =
            Tensor::with_shape_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
        let t2 = Tensor::with_shape_f32(
            vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            vec![2, 2, 2],
        );

        let result0 = t1.concat(&t2, Some(0));
        match result0 {
            Tensor::F32 {
                data: data0,
                shape: shape0,
            } => {
                assert_eq!(
                    shape0,
                    vec![4, 2, 2],
                    "Shape should be [4, 2, 2] when concatenating along dim 0"
                );
                assert_eq!(
                    data0,
                    vec![
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                        15.0, 16.0
                    ],
                    "Data should be concatenated correctly along dim 0"
                );
            }
            _ => panic!("Expected F32 tensor result"),
        }

        let result1 = t1.concat(&t2, Some(1));
        match result1 {
            Tensor::F32 {
                data: data1,
                shape: shape1,
            } => {
                assert_eq!(
                    shape1,
                    vec![2, 4, 2],
                    "Shape should be [2, 4, 2] when concatenating along dim 1"
                );
                assert_eq!(
                    data1,
                    vec![
                        1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0, 5.0, 6.0, 7.0, 8.0, 13.0, 14.0,
                        15.0, 16.0
                    ],
                    "Data should be interleaved correctly when concatenating along dim 1"
                );
            }
            _ => panic!("Expected F32 tensor result"),
        }

        let result2 = t1.concat(&t2, Some(2));
        match result2 {
            Tensor::F32 {
                data: data2,
                shape: shape2,
            } => {
                assert_eq!(
                    shape2,
                    vec![2, 2, 4],
                    "Shape should be [2, 2, 4] when concatenating along dim 2"
                );
                assert_eq!(
                    data2,
                    vec![
                        1.0, 2.0, 9.0, 10.0, 3.0, 4.0, 11.0, 12.0, 5.0, 6.0, 13.0, 14.0, 7.0, 8.0,
                        15.0, 16.0
                    ],
                    "Data should be interleaved correctly when concatenating along dim 2"
                );
            }
            _ => panic!("Expected F32 tensor result"),
        }
    }
}
