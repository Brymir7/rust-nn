use once_cell::unsync::Lazy;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;
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
    fn get_mut_tensor(&mut self, id: TensorHandle) -> Option<&mut Tensor> {
        self.all_tensors.get_mut(id.0)
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
fn with_mut_tensor<F, R>(id: TensorHandle, f: F) -> Option<R>
where
    F: FnOnce(&mut Tensor) -> R,
{
    TENSOR_CONTEXT.with(|ctx| {
        let mut ctx_ref = ctx.borrow_mut();
        let tensor = ctx_ref.get_mut_tensor(id)?;
        Some(f(tensor))
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
    Abs {
        input: TensorHandle,
    },
    None,
}

#[derive(Clone, Debug)]
enum TensorData {
    F32 {
        // id: TensorHandle,
        data: Vec<f32>,
        shape: Vec<usize>,
        // graph: TensorOperation,
        // grad: Option<Vec<f32>>,
    },
    F64 {
        data: Vec<f64>,
        shape: Vec<usize>,
    },
}
impl TensorData {
    fn shape(&self) -> &Vec<usize> {
        match self {
            TensorData::F32 { shape, .. } => shape,
            TensorData::F64 { shape, .. } => shape,
        }
    }

    fn data_f32(&self) -> &Vec<f32> {
        match self {
            TensorData::F32 { data, .. } => data,
            _ => panic!("Not an f32 tensor"),
        }
    }

    fn data_f64(&self) -> &Vec<f64> {
        match self {
            TensorData::F64 { data, .. } => data,
            _ => panic!("Not an f64 tensor"),
        }
    }

    fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            TensorData::F32 { data, .. } => data.clone(),
            TensorData::F64 { data, .. } => data.iter().map(|&x| x as f32).collect(),
        }
    }

    fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            TensorData::F32 { data, .. } => data.iter().map(|&x| x as f64).collect(),
            TensorData::F64 { data, .. } => data.clone(),
        }
    }

    fn is_f32(&self) -> bool {
        matches!(self, TensorData::F32 { .. })
    }

    fn is_f64(&self) -> bool {
        matches!(self, TensorData::F64 { .. })
    }

    fn size(&self) -> usize {
        match self {
            TensorData::F32 { data, .. } => data.len(),
            TensorData::F64 { data, .. } => data.len(),
        }
    }
}

#[derive(Clone)]
enum Tensor {
    F32 {
        id: TensorHandle,
        data: TensorData,
        graph: TensorOperation,
        grad: Option<TensorData>,
    },
    F64 {
        id: TensorHandle,
        data: TensorData,
        graph: TensorOperation,
        grad: Option<TensorData>,
    },
}
impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tensor::F32 {
                id,
                data,
                graph,
                grad,
            } => {
                write!(f, "Tensor::F32 {:?}{{\n", id)?;
                format_tensor(f, data.data_f32(), data.shape(), 0)?;
                // write!(f, "\nshape: {:?}\n}}", shape)?;
                write!(f, "\ngraph: {:?}\n}}", graph);
                write!(f, "\ngrad:  {:?}\n}}", grad)
            }
            _ => todo!(),
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
        // Validate shape and create proper shape vec
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

        let tensor_data: TensorData = TensorData::F32 { data, shape };
        let tensor = Tensor::F32 {
            id: TensorHandle(get_next_tensor_id()),
            data: tensor_data,
            graph: TensorOperation::None,
            grad: None,
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
        let tensor_data: TensorData = TensorData::F64 { data, shape };
        let tensor = Tensor::F32 {
            id: TensorHandle(get_next_tensor_id()),
            data: tensor_data,
            graph: TensorOperation::None,
            grad: None,
        };
        let id = register_tensor(tensor);
        return get_tensor(id).unwrap();
    }

    fn from_op(data: Vec<f32>, shape: Vec<usize>, op: TensorOperation) -> Self {
        let tensor_data = TensorData::F32 { data, shape };
        let tensor = Tensor::F32 {
            id: TensorHandle(get_next_tensor_id()),
            data: tensor_data,
            graph: op,
            grad: None,
        };
        let id = register_tensor(tensor);
        return get_tensor(id).unwrap();
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

    fn backward(&self) {
        let mut queue: VecDeque<(TensorOperation, TensorData)> = VecDeque::new(); // op and grad on result // prev grad
        match self {
            Tensor::F32 { graph, .. } => {
                queue.push_back((graph.clone(), TensorData::from_vec_f32(vec![1.0])));
                // loss has 1.0 grad
            }
            _ => todo!(),
        }

        while let Some((tensor_h, prev_grad)) = queue.pop_front() {
            match tensor_h {
                TensorOperation::Add { left, right } => {
                    with_mut_tensor(left.clone(), |tensor| match tensor {
                        Tensor::F32 { grad, .. } => {
                            let mut new_grad: TensorData = prev_grad.clone() * 1.0;
                            match grad {
                                Some(ref existing_grad) => {
                                    new_grad = new_grad + existing_grad.clone();
                                }
                                None => {}
                            }
                            *grad = Some(new_grad.clone());
                            queue.push_back((tensor.graph(), new_grad));
                        }
                        _ => todo!(),
                    });
                    match right {
                        OperatorHandle::Tensor(right) => {
                            with_mut_tensor(right, |tensor| match tensor {
                                Tensor::F32 { grad, .. } => {
                                    let mut new_grad: TensorData = prev_grad * 1.0;
                                    match grad {
                                        Some(ref existing_grad) => {
                                            new_grad = new_grad + existing_grad.clone();
                                        }
                                        None => {}
                                    }
                                    *grad = Some(new_grad.clone());
                                    queue.push_back((tensor.graph(), new_grad));
                                }
                                _ => todo!(),
                            });
                        }
                        OperatorHandle::Scalar => {}
                    }
                }
                TensorOperation::Sub { left, right } => {
                    // For subtraction: gradient flows to left unchanged, but negated to right
                    with_mut_tensor(left.clone(), |tensor| match tensor {
                        Tensor::F32 { grad, .. } => {
                            let mut new_grad: TensorData = prev_grad.clone() * 1.0;
                            match grad {
                                Some(ref existing_grad) => {
                                    new_grad = new_grad + existing_grad.clone();
                                }
                                None => {}
                            }
                            *grad = Some(new_grad.clone());
                            queue.push_back((tensor.graph(), new_grad));
                        }
                        _ => todo!(),
                    });
                    match right {
                        OperatorHandle::Tensor(right) => {
                            with_mut_tensor(right, |tensor| match tensor {
                                Tensor::F32 { grad, .. } => {
                                    // Note the negative gradient for right operand of subtraction
                                    let mut new_grad = prev_grad * -1.0;
                                    match grad {
                                        Some(ref existing_grad) => {
                                            new_grad = new_grad + existing_grad.clone();
                                        }
                                        None => {}
                                    }
                                    *grad = Some(new_grad);
                                }
                                _ => todo!(),
                            });
                        }
                        OperatorHandle::Scalar => {}
                    }
                }
                TensorOperation::Sum { input } => {
                    with_mut_tensor(input.clone(), |tensor| match tensor {
                        Tensor::F32 { grad, .. } => {
                            let mut new_grad: TensorData = prev_grad.clone() * 1.0;
                            match grad {
                                Some(grad) => {
                                    new_grad = new_grad + grad.clone();
                                }
                                None => {
                                    new_grad = new_grad;
                                }
                            }
                            *grad = Some(new_grad.clone());
                            queue.push_back((tensor.graph(), new_grad));
                        }
                        _ => todo!(),
                    });
                }
                TensorOperation::Mul { left, right } => {
                    todo!()
                }

                TensorOperation::Abs { input } => {
                    with_mut_tensor(input.clone(), |tensor| match tensor {
                        Tensor::F32 { grad, .. } => {
                            let mut new_grad: TensorData = prev_grad.clone() * 1.0;
                            match grad {
                                Some(grad) => {
                                    new_grad = new_grad + grad.clone();
                                }
                                None => {
                                    new_grad = new_grad;
                                }
                            }
                            *grad = Some(new_grad.clone());
                            queue.push_back((tensor.graph(), new_grad));
                        }
                        _ => todo!(),
                    });
                }
                TensorOperation::None => {}

                _ => {
                    println!("Operation not implemented {:?}", tensor_h);
                    todo!()
                }
            }
        }
    }
    fn grad_f32(&self) -> f32 {
        match self {
            Tensor::F32 { grad, .. } => match grad {
                Some(grad) => grad.data_f32()[0],
                None => 0.0,
            },
            _ => todo!(),
        }
    }
    fn graph(&self) -> TensorOperation {
        match self {
            Tensor::F32 { graph, .. } => graph.clone(),
            _ => todo!(),
        }
    }
    fn abs(&self) -> Tensor {
        match self {
            Tensor::F32 {
                id,
                data,
                graph,
                grad,
            } => {
                let new_data = data.data_f32().iter().map(|x| x.abs()).collect();
                Tensor::from_op(
                    new_data,
                    data.shape().clone(),
                    TensorOperation::Abs { input: id.clone() },
                )
            }
            _ => todo!(),
        }
    }
    // todo should be along axis
    fn sum(&self) -> Tensor {
        match self {
            Tensor::F32 { id, data, .. } => {
                let sum = data.data_f32().iter().sum();
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
            Tensor::F32 { id, data, .. } => match &other {
                Tensor::F32 {
                    id: other_id,
                    data: other_data,
                    ..
                } => {
                    let other_shape = other_data.shape();
                    let shape = data.shape();
                    let data = data.data_f32();
                    let other_data = other_data.data_f32();
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
            _ => todo!(),
        }
    }
}

impl TensorData {
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
        TensorData::F32 { data, shape }
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

        TensorData::F64 { data, shape }
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
}

macro_rules! impl_tensordata_op {
    ($trait:ident, $method:ident, $op:tt, $op_name:expr) => {
        impl $trait for &TensorData {
            type Output = TensorData;

            fn $method(self, other: &TensorData) -> TensorData {
                match (self, other) {
                    (
                        TensorData::F32 { data: data1, shape: shape1 },
                        TensorData::F32 { data: data2, shape: shape2 },
                    ) => {
                        assert_eq!(shape1, shape2, concat!("Shapes must match for ", $op_name));
                        let new_data = data1.iter().zip(data2.iter()).map(|(a, b)| a $op b).collect();
                        TensorData::with_shape_f32(new_data, shape1.clone())
                    }
                    (
                        TensorData::F64 { data: data1, shape: shape1 },
                        TensorData::F64 { data: data2, shape: shape2 },
                    ) => {
                        assert_eq!(shape1, shape2, concat!("Shapes must match for ", $op_name));
                        let new_data = data1.iter().zip(data2.iter()).map(|(a, b)| a $op b).collect();
                        TensorData::with_shape_f64(new_data, shape1.clone())
                    }
                    _ => panic!(concat!("Cannot ", $op_name, " tensor data of different types (f32 vs f64)")),
                }
            }
        }

        impl $trait for TensorData {
            type Output = TensorData;

            fn $method(self, other: TensorData) -> TensorData {
                &self $op &other
            }
        }

        // Implementation for TensorData op scalar
        impl $trait<f32> for &TensorData {
            type Output = TensorData;

            fn $method(self, scalar: f32) -> TensorData {
                match self {
                    TensorData::F32 { data, shape } => {
                        let new_data = data.iter().map(|&x| x $op scalar).collect();
                        TensorData::with_shape_f32(new_data, shape.clone())
                    }
                    TensorData::F64 { data, shape } => {
                        let new_data = data.iter().map(|&x| x $op scalar as f64).collect();
                        TensorData::with_shape_f64(new_data, shape.clone())
                    }
                }
            }
        }

        impl $trait<f64> for &TensorData {
            type Output = TensorData;

            fn $method(self, scalar: f64) -> TensorData {
                match self {
                    TensorData::F32 { data, shape } => {
                        let new_data = data.iter().map(|&x| x $op scalar as f32).collect();
                        TensorData::with_shape_f32(new_data, shape.clone())
                    }
                    TensorData::F64 { data, shape } => {
                        let new_data = data.iter().map(|&x| x $op scalar).collect();
                        TensorData::with_shape_f64(new_data, shape.clone())
                    }
                }
            }
        }

        impl $trait<f32> for TensorData {
            type Output = TensorData;

            fn $method(self, scalar: f32) -> TensorData {
                &self $op scalar
            }
        }

        impl $trait<f64> for TensorData {
            type Output = TensorData;

            fn $method(self, scalar: f64) -> TensorData {
                &self $op scalar
            }
        }

        // For the reverse operation (scalar op TensorData)
        impl $trait<&TensorData> for f32 {
            type Output = TensorData;

            fn $method(self, tensor_data: &TensorData) -> TensorData {
                match tensor_data {
                    TensorData::F32 { data, shape } => {
                        let new_data = data.iter().map(|&x| self $op x).collect();
                        TensorData::with_shape_f32(new_data, shape.clone())
                    }
                    TensorData::F64 { data, shape } => {
                        let new_data = data.iter().map(|&x| self as f64 $op x).collect();
                        TensorData::with_shape_f64(new_data, shape.clone())
                    }
                }
            }
        }

        impl $trait<&TensorData> for f64 {
            type Output = TensorData;

            fn $method(self, tensor_data: &TensorData) -> TensorData {
                match tensor_data {
                    TensorData::F32 { data, shape } => {
                        let new_data = data.iter().map(|&x| self as f32 $op x).collect();
                        TensorData::with_shape_f32(new_data, shape.clone())
                    }
                    TensorData::F64 { data, shape } => {
                        let new_data = data.iter().map(|&x| self $op x).collect();
                        TensorData::with_shape_f64(new_data, shape.clone())
                    }
                }
            }
        }

        impl $trait<TensorData> for f32 {
            type Output = TensorData;

            fn $method(self, tensor_data: TensorData) -> TensorData {
                self $op &tensor_data
            }
        }

        impl $trait<TensorData> for f64 {
            type Output = TensorData;

            fn $method(self, tensor_data: TensorData) -> TensorData {
                self $op &tensor_data
            }
        }
    };
}

// Now, let's modify the Tensor macro to use the TensorData operations
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
                            ..
                        },
                        Tensor::F32 {
                            id: id2,
                            data: data2,
                            ..
                        },
                    ) => {
                        // Use the TensorData operation
                        let new_data = data1 $op data2;

                        Tensor::from_op(
                            new_data.data_f32().clone(),
                            new_data.shape().clone(),
                            TensorOperation::$op_enum {
                                left: id1.clone(),
                                right: OperatorHandle::Tensor(id2.clone()),
                            }
                        )
                    }
                    (
                        Tensor::F64 {
                            id: id1,
                            data: data1,
                            ..
                        },
                        Tensor::F64 {
                            id: id2,
                            data: data2,
                            ..
                        },
                    ) => {
                        // Use the TensorData operation
                        let new_data = data1 $op data2;

                        // This is a placeholder - you'll need to implement from_op for F64
                        todo!()
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
                    Tensor::F32 { id, data, .. } => {
                        // Use the TensorData operation
                        let new_data = data $op scalar;

                        Tensor::from_op(
                            new_data.data_f32().clone(),
                            new_data.shape().clone(),
                            TensorOperation::$op_enum {
                                left: id.clone(),
                                right: OperatorHandle::Scalar,
                            }
                        )
                    }
                    Tensor::F64 { id, data, .. } => {
                        todo!()
                    }
                }
            }
        }

        impl $trait<f64> for &Tensor {
            type Output = Tensor;

            fn $method(self, scalar: f64) -> Tensor {
                match self {
                    Tensor::F32 { id, data, .. } => {
                        // Use the TensorData operation
                        let new_data = data $op scalar;

                        Tensor::from_op(
                            new_data.data_f32().clone(),
                            new_data.shape().clone(),
                            TensorOperation::$op_enum {
                                left: id.clone(),
                                right: OperatorHandle::Scalar,
                            }
                        )
                    }
                    Tensor::F64 { id, data, .. } => {
                        todo!()
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
                    Tensor::F32 { id, data, .. } => {
                        // Use the TensorData operation
                        let new_data = self $op data;

                        Tensor::from_op(
                            new_data.data_f32().clone(),
                            new_data.shape().clone(),
                            TensorOperation::$op_enum {
                                left: id.clone(),
                                right: OperatorHandle::Scalar,
                            }
                        )
                    }
                    Tensor::F64 { data, .. } => {
                        todo!()
                    }
                }
            }
        }

        impl $trait<&Tensor> for f64 {
            type Output = Tensor;

            fn $method(self, tensor: &Tensor) -> Tensor {
                match tensor {
                    Tensor::F32 { id, data, .. } => {
                        // Use the TensorData operation
                        let new_data = self $op data;

                        Tensor::from_op(
                            new_data.data_f32().clone(),
                            new_data.shape().clone(),
                            TensorOperation::$op_enum {
                                left: id.clone(),
                                right: OperatorHandle::Scalar,
                            }
                        )
                    }
                    Tensor::F64 { data, .. } => {
                        todo!()
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

// Implement operations for TensorData
impl_tensordata_op!(Add, add, +, "add");
impl_tensordata_op!(Sub, sub, -, "subtract");
impl_tensordata_op!(Mul, mul, *, "multiply");
impl_tensordata_op!(Div, div, /, "divide");

// Implement operations for Tensor (now using TensorData operations internally)
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

                graph,
                grad: None,
            } => {
                assert_eq!(
                    *data.shape(),
                    vec![4, 2],
                    "Shape should be [4, 2] when concatenating along default dimension"
                );
                assert_eq!(
                    *data.data_f32(),
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    "Data should be concatenated correctly"
                );
            }
            _ => panic!("Expected F32 tensor result"),
        }
    }

    {
        let t1: Tensor = Tensor::with_shape_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::with_shape_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let result = t1.concat(&t2, Some(0));

        match result {
            Tensor::F32 {
                id,
                data,

                graph,
                grad: None,
            } => {
                assert_eq!(
                    *data.shape(),
                    vec![4, 2],
                    "Shape should be [4, 2] when concatenating along dim 0"
                );
                assert_eq!(
                    *data.data_f32(),
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

                graph,
                grad: None,
            } => {
                assert_eq!(
                    *data.shape(),
                    vec![2, 4],
                    "Shape should be [2, 4] when concatenating along dim 1"
                );
                assert_eq!(
                    *data.data_f32(),
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

                graph: graph0,
                grad: None,
            } => {
                assert_eq!(
                    *data0.shape(),
                    vec![4, 2, 2],
                    "Shape should be [4, 2, 2] when concatenating along dim 0"
                );
                assert_eq!(
                    *data0.data_f32(),
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

                graph: graph1,
                grad: None,
            } => {
                assert_eq!(
                    *data1.shape(),
                    vec![2, 4, 2],
                    "Shape should be [2, 4, 2] when concatenating along dim 1"
                );
                assert_eq!(
                    *data1.data_f32(),
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
                graph: graph1,
                grad: None,
            } => {
                assert_eq!(
                    *data2.shape(),
                    vec![2, 2, 4],
                    "Shape should be [2, 2, 4] when concatenating along dim 2"
                );
                assert_eq!(
                    *data2.data_f32(),
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
fn get_computation_graph(curr_tensor: &Tensor, graph: &mut Vec<TensorOperation>) {
    match curr_tensor {
        Tensor::F32 {
            id,
            data,

            graph: curr_graph,
            grad: None,
        } => match curr_graph {
            TensorOperation::None => {
                return;
            }
            _ => {
                graph.push(curr_graph.clone());

                let mut queue = Vec::new();

                match curr_graph {
                    TensorOperation::Add { left, right }
                    | TensorOperation::Mul { left, right }
                    | TensorOperation::Sub { left, right }
                    | TensorOperation::Div { left, right } => {
                        queue.push(left.clone());
                        if let OperatorHandle::Tensor(right_handle) = right {
                            queue.push(right_handle.clone());
                        }
                    }
                    TensorOperation::Sum { input } => {
                        queue.push(input.clone());
                    }
                    TensorOperation::Concat { inputs, dim: _ } => {
                        for input_handle in inputs {
                            queue.push(input_handle.clone());
                        }
                    }
                    TensorOperation::Abs { input } => {
                        queue.push(input.clone());
                    }
                    TensorOperation::None => {}
                }

                let mut i = 0;
                while i < queue.len() {
                    if let Some(tensor) = get_tensor(queue[i].clone()) {
                        if let Tensor::F32 { graph: op, .. } = &tensor {
                            match op {
                                TensorOperation::None => {}
                                _ => {
                                    graph.push(op.clone());

                                    match op {
                                        TensorOperation::Add { left, right }
                                        | TensorOperation::Mul { left, right }
                                        | TensorOperation::Sub { left, right }
                                        | TensorOperation::Div { left, right } => {
                                            queue.push(left.clone());
                                            if let OperatorHandle::Tensor(right_handle) = right {
                                                queue.push(right_handle.clone());
                                            }
                                        }
                                        TensorOperation::Sum { input } => {
                                            queue.push(input.clone());
                                        }
                                        TensorOperation::Concat { inputs, dim: _ } => {
                                            for input_handle in inputs {
                                                queue.push(input_handle.clone());
                                            }
                                        }
                                        TensorOperation::Abs { input } => {
                                            queue.push(input.clone());
                                        }
                                        TensorOperation::None => {}
                                    }
                                }
                            }
                        }
                    }
                    i += 1;
                }
            }
        },
        Tensor::F64 { data, .. } => {
            todo!()
        }
        _ => todo!(),
    }
}

fn print_computation_graph(tensor: &Tensor) {
    let mut graph = Vec::new();
    get_computation_graph(tensor, &mut graph);

    println!("Computation Graph:");
    if graph.is_empty() {
        println!("  Empty (no operations)");
        return;
    }

    for (i, op) in graph.iter().enumerate() {
        let indent = "  ";
        println!("{}[{}]: {}", indent, i, format_operation(op));
    }
}
// todo cleanup
fn print_tensor_data(tensor: &Tensor) {
    match tensor {
        Tensor::F32 {
            id,
            data,

            graph,
            grad,
        } => {
            println!("Tensor::F32 {{");
            println!("  data: {:?}", data);
            println!("}}");
        }
        _ => todo!(),
    }
}
fn format_operation(op: &TensorOperation) -> String {
    match op {
        TensorOperation::Add { left, right } => match right {
            OperatorHandle::Tensor(right_id) => {
                format!("Add(Tensor({}) + Tensor({}))", left.0, right_id.0)
            }
            OperatorHandle::Scalar => format!("Add(Tensor({}) + Scalar)", left.0),
        },
        TensorOperation::Mul { left, right } => match right {
            OperatorHandle::Tensor(right_id) => {
                format!("Mul(Tensor({}) * Tensor({}))", left.0, right_id.0)
            }
            OperatorHandle::Scalar => format!("Mul(Tensor({}) * Scalar)", left.0),
        },
        TensorOperation::Sub { left, right } => match right {
            OperatorHandle::Tensor(right_id) => {
                format!("Sub(Tensor({}) - Tensor({}))", left.0, right_id.0)
            }
            OperatorHandle::Scalar => format!("Sub(Tensor({}) - Scalar)", left.0),
        },
        TensorOperation::Div { left, right } => match right {
            OperatorHandle::Tensor(right_id) => {
                format!("Div(Tensor({}) / Tensor({}))", left.0, right_id.0)
            }
            OperatorHandle::Scalar => format!("Div(Tensor({}) / Scalar)", left.0),
        },
        TensorOperation::Sum { input } => {
            format!("Sum(Tensor({}))", input.0)
        }
        TensorOperation::Concat { inputs, dim } => {
            let ids = inputs
                .iter()
                .map(|id| id.0.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            match dim {
                Some(d) => format!("Concat(Tensors([{}]), dim={})", ids, d),
                None => format!("Concat(Tensors([{}]))", ids),
            }
        }
        TensorOperation::Abs { input } => {
            format!("Abs(Tensor({}))", input.0)
        }
        TensorOperation::None => "None".to_string(),
    }
}

fn main() {
    // tests();
    let t1 = Tensor::with_shape_f32(vec![3.0], vec![1]);
    let t2 = Tensor::with_shape_f32(vec![1.5], vec![1]);
    let res = t1 - t2;
    let wanted: f32 = 3.0;
    let loss = (wanted - res).abs();
    print_tensor_data(&loss);
    loss.backward();
    TENSOR_CONTEXT.with_borrow(|ctx| {
        for t in ctx.all_tensors.iter() {
            println!("tensor {:?}", t.grad_f32());
        }
    });
    print_computation_graph(&loss);
    // println!("tensor 4 {:?}", t4);
}
