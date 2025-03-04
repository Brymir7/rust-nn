use once_cell::unsync::Lazy;
use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Mutex;

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
#[derive(Clone, Debug)]
struct TensorHandle(usize);
struct TensorContext {
    all_tensors: Vec<Tensor>,
}

impl TensorContext {
    fn get_next_id(&self) -> usize {
        return self.all_tensors.len();
    }
    fn register_tensor(&mut self, tensor: Tensor) -> usize {
        let id = self.all_tensors.len();
        self.all_tensors.push(tensor);
        id
    }

    fn get_tensor(&self, id: TensorHandle) -> Option<&Tensor> {
        self.all_tensors.get(id.0)
    }
}

thread_local! {
    static TENSOR_CONTEXT: RefCell<TensorContext> = RefCell::new(TensorContext {
        all_tensors: Vec::new(),
    });
}

fn register_tensor(tensor: Tensor) -> TensorHandle {
    TensorHandle(TENSOR_CONTEXT.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        ctx.register_tensor(tensor)
    }))
}

fn get_tensor(id: TensorHandle) -> Option<Tensor> {
    TENSOR_CONTEXT.with(|ctx| {
        let ctx = ctx.borrow();
        ctx.get_tensor(id).cloned()
    })
}

fn get_next_tensor_id() -> usize {
    TENSOR_CONTEXT.with(|ctx| {
        let ctx = ctx.borrow();
        ctx.get_next_id()
    })
}
#[derive(Clone, Debug)]
enum OperatorHandle {
    Tensor(TensorHandle),
    Scalar,
}
#[derive(Clone, Debug)]
enum TensorOperation {
    Add {
        left: TensorHandle,
        right: OperatorHandle, // can also add with scalar
    },
    Mul {
        left: TensorHandle,
        right: OperatorHandle, // can also mul with scalar
    },
    Sub {
        left: TensorHandle,
        right: OperatorHandle, // can also add with scalar
    },
    Div {
        left: TensorHandle,
        right: OperatorHandle, // can also mul with scalar
    },
    Sum {
        input: TensorHandle,
    },
    Concat {
        inputs: Vec<TensorHandle>,
        dim: Option<usize>,
    },
    None,
}

#[derive(Clone)]
enum Tensor {
    F32 {
        id: TensorHandle,
        data: Vec<f32>,
        shape: Vec<usize>,
        graph: TensorOperation,
    },
    F64 {
        data: Vec<f64>,
        shape: Vec<usize>,
    },
}
impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tensor::F32 {
                id,
                data,
                shape,
                graph,
            } => {
                write!(f, "Tensor::F32 {:?}{{\n", id)?;
                format_tensor(f, data, shape, 0)?;
                write!(f, "\nshape: {:?}\n}}", shape)?;
                write!(f, "\ngraph: {:?}\n}}", graph)
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
        let tensor = Tensor::F32 {
            id: TensorHandle(get_next_tensor_id()),
            data,
            shape,
            graph: TensorOperation::None,
        };
        let id = register_tensor(tensor);
        return get_tensor(id).unwrap();
    }
    fn from_op(data: Vec<f32>, shape: Vec<usize>, op: TensorOperation) -> Self {
        let tensor = Tensor::F32 {
            id: TensorHandle(get_next_tensor_id()),
            data: data,
            shape: shape,
            graph: op,
        };
        let id = register_tensor(tensor);
        return get_tensor(id).unwrap();
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
    // todo should be along axis
    fn sum(&self) -> Tensor {
        match self {
            Tensor::F32 { id, data, .. } => {
                let sum = data.iter().sum();
                let new_data = vec![sum];
                Tensor::from_op(
                    new_data,
                    vec![1],
                    TensorOperation::Sum { input: id.clone() },
                )
            }
            _ => todo!(),
        }
    }

    fn concat(&self, other: &Tensor, dim: Option<usize>) -> Tensor {
        match &self {
            Tensor::F32 {
                id, data, shape, ..
            } => match &other {
                Tensor::F32 {
                    id: other_id,
                    data: other_data,
                    shape: other_shape,
                    ..
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
                            let inner_dim = shape[dim + 1..].iter().product(); // this is equal for both
                            let self_dim = shape[dim];
                            let other_dim = other_shape[dim];

                            for outer_dim in 0..outer_dim {
                                let outer_offset = outer_dim * self_dim * inner_dim;
                                for i in 0..self_dim {
                                    let start = outer_offset + i * inner_dim;
                                    for k in 0..inner_dim {
                                        new_data.push(data[start + k])
                                    }
                                }

                                for i in 0..other_dim {
                                    let start = outer_offset + i * inner_dim;
                                    for k in 0..inner_dim {
                                        new_data.push(other_data[start + k]);
                                    }
                                }
                            }

                            Tensor::from_op(
                                new_data,
                                new_shape,
                                TensorOperation::Concat {
                                    inputs: vec![id.clone(), other_id.clone()],
                                    dim: Some(dim),
                                },
                            )
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
    ($trait:ident, $method:ident, $op:tt, $op_name:expr, $op_enum:ident) => {
        impl $trait for &Tensor {
            type Output = Tensor;

            fn $method(self, other: &Tensor) -> Tensor {
                match (self, other) {
                    (
                        Tensor::F32 {
                            id: id1,
                            data: data1,
                            shape: shape1,
                            ..
                        },
                        Tensor::F32 {
                            id: id2,
                            data: data2,
                            shape: shape2,
                            ..
                        },
                    ) => {
                        assert_eq!(shape1, shape2, concat!("Shapes must match for ", $op_name));
                        let new_data = data1.iter().zip(data2.iter()).map(|(a, b)| a $op b).collect();
                        Tensor::from_op(
                            new_data,
                            shape1.clone(),
                            TensorOperation::$op_enum {
                                left: id1.clone(),
                                right: OperatorHandle::Tensor(id2.clone()),
                            }
                        )
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
                    Tensor::F32 { id, data, shape, .. } => {
                        let new_data = data.iter().map(|&x| x $op scalar).collect();
                        Tensor::from_op(
                            new_data,
                            shape.clone(),
                            TensorOperation::$op_enum {
                                left: id.clone(),
                                right: OperatorHandle::Scalar,
                            }
                        )
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
                    Tensor::F32 { id, data, shape, .. } => {
                        let new_data = data.iter().map(|&x| x $op scalar as f32).collect();
                        Tensor::from_op(
                            new_data,
                            shape.clone(),
                            TensorOperation::$op_enum {
                                left: id.clone(),
                                right: OperatorHandle::Scalar,
                            }
                        )
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
                    Tensor::F32 { id, data, shape, .. } => {
                        let new_data = data.iter().map(|&x| self $op x).collect();
                        // For scalars operating on tensors, we still record the operation
                        // but with right and left swapped according to the operation
                        Tensor::from_op(
                            new_data,
                            shape.clone(),
                            TensorOperation::$op_enum {
                                left: id.clone(), // Note: semantically this may need adjustment based on operation
                                right: OperatorHandle::Scalar,
                            }
                        )
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
                    Tensor::F32 { id, data, shape, .. } => {
                        let new_data = data.iter().map(|&x| self as f32 $op x).collect();
                        Tensor::from_op(
                            new_data,
                            shape.clone(),
                            TensorOperation::$op_enum {
                                left: id.clone(),
                                right: OperatorHandle::Scalar,
                            }
                        )
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
impl_tensor_op!(Add, add, +, "add", Add);
impl_tensor_op!(Sub, sub, -, "subtract", Sub);
impl_tensor_op!(Mul, mul, *, "multiply", Mul);
impl_tensor_op!(Div, div, /, "divide", Div);

fn tests() {
    {
        let t1 = Tensor::with_shape_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::with_shape_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let result = t1.concat(&t2, None);

        match result {
            Tensor::F32 {
                id,
                data,
                shape,
                graph,
            } => {
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
            Tensor::F32 {
                id,
                data,
                shape,
                graph,
            } => {
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
            Tensor::F32 {
                id,
                data,
                shape,
                graph,
            } => {
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
                id,
                data: data0,
                shape: shape0,
                graph: graph0,
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
                id,
                data: data1,
                shape: shape1,
                graph: graph1,
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
                id,
                data: data2,
                shape: shape2,
                graph: graph1,
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
