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
#[derive(Clone, Debug, Copy)]
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
    Scalar(f32),
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
        forward_positive: bool, // whether the forward pass val was pos / neg before abs
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
    fn mut_data_f32(&mut self) -> &mut Vec<f32> {
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
        requires_grad: bool,
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
                ..
            } => {
                write!(f, "Tensor::F32 {:?}{{\n", id)?;
                format_tensor(f, data.data_f32(), data.shape(), 0)?;
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
    fn new_f32(data: Vec<f32>, shape: Option<Vec<usize>>, requires_grad: bool) -> TensorHandle {
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
            requires_grad,
        };
        register_tensor(tensor)
    }
    fn new_f64(data: Vec<f64>, shape: Option<Vec<usize>>, requires_grad: bool) -> TensorHandle {
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
            requires_grad,
        };
        register_tensor(tensor)
    }

    fn from_op(data: Vec<f32>, shape: Vec<usize>, op: TensorOperation) -> TensorHandle {
        let tensor_data = TensorData::F32 { data, shape };
        let tensor = Tensor::F32 {
            id: TensorHandle(get_next_tensor_id()),
            data: tensor_data,
            graph: op,
            grad: None,
            requires_grad: true,
        };
        register_tensor(tensor)
    }

    fn with_shape_f32(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> TensorHandle {
        Self::new_f32(data, Some(shape), requires_grad)
    }
    fn with_shape_f64(data: Vec<f64>, shape: Vec<usize>, requires_grad: bool) -> TensorHandle {
        Self::new_f64(data, Some(shape), requires_grad)
    }
    fn from_vec_f32(data: Vec<f32>, requires_grad: bool) -> TensorHandle {
        Self::new_f32(data, None, requires_grad)
    }
    fn random_f32(shape: Vec<usize>, requires_grad: bool) -> TensorHandle {
        let data = (0..shape.iter().product())
            .map(|_| rand::random::<f32>())
            .collect();
        Self::new_f32(data, Some(shape), requires_grad)
    }
    fn backward(&self) {
        let mut queue: VecDeque<(TensorOperation, TensorData)> = VecDeque::new(); // op and grad on result // prev grad
        match self {
            Tensor::F32 { graph, data, .. } => {
                // Initialize with ones matching the loss tensor shape
                let shape = data.shape();
                let size = shape.iter().product::<usize>();
                let ones = vec![1.0; size];
                queue.push_back((
                    graph.clone(),
                    TensorData::with_shape_f32(ones, shape.clone()),
                ));
            }
            _ => todo!(),
        }

        while let Some((tensor_h, prev_grad)) = queue.pop_front() {
            match tensor_h {
                TensorOperation::Add { left, right } => {
                    // Process left operand gradient
                    with_mut_tensor(left, |tensor| {
                        if !tensor.requires_grad() {
                            return;
                        }

                        match tensor {
                            Tensor::F32 { grad, data, .. } => {
                                // Preserve the original shape of the tensor for the gradient
                                let shape = data.shape().clone();
                                let mut new_grad = TensorData::F32 {
                                    data: prev_grad.data_f32().clone(), // Use the entire previous gradient
                                    shape: shape.clone(),
                                };

                                // Accumulate with existing gradient if present
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
                        }
                    });

                    // Process right operand gradient
                    match right {
                        OperatorHandle::Tensor(right) => {
                            with_mut_tensor(right, |tensor| {
                                if !tensor.requires_grad() {
                                    return;
                                }

                                match tensor {
                                    Tensor::F32 { grad, data, .. } => {
                                        // Preserve the original shape of the tensor for the gradient
                                        let shape = data.shape().clone();
                                        let mut new_grad = TensorData::F32 {
                                            data: prev_grad.data_f32().clone(), // Use the entire previous gradient
                                            shape: shape.clone(),
                                        };

                                        // Accumulate with existing gradient if present
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
                                }
                            });
                        }
                        OperatorHandle::Scalar(_) => {} // Scalar gradients are not tracked
                    }
                }
                TensorOperation::Sub { left, right } => {
                    // For subtraction: gradient flows to left unchanged, but negated to right
                    with_mut_tensor(left, |tensor| {
                        if !tensor.requires_grad() {
                            return;
                        }
                        match tensor {
                            Tensor::F32 { grad, .. } => {
                                let mut new_grad: TensorData = prev_grad.clone() * 1.0;
                                match grad {
                                    Some(ref existing_grad) => {
                                        new_grad = new_grad + existing_grad.clone();
                                    }
                                    None => {}
                                }
                                *grad = Some(new_grad.clone());
                                if tensor.requires_grad() {
                                    queue.push_back((tensor.graph(), new_grad));
                                }
                            }
                            _ => todo!(),
                        }
                    });
                    match right {
                        OperatorHandle::Tensor(right) => {
                            with_mut_tensor(right, |tensor| {
                                if !tensor.requires_grad() {
                                    return;
                                }
                                match tensor {
                                    Tensor::F32 { grad, .. } => {
                                        // Note the negative gradient for right operand of subtraction
                                        let mut new_grad: TensorData = prev_grad * -1.0;
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
                                }
                            });
                        }
                        OperatorHandle::Scalar(_) => {}
                    }
                }
                TensorOperation::Sum { input } => {
                    with_mut_tensor(input, |tensor| {
                        if !tensor.requires_grad() {
                            return;
                        }
                        match tensor {
                            Tensor::F32 { grad, data, .. } => {
                                let input_shape = data.shape();
                                let input_size = input_shape.iter().product::<usize>();
                                let scalar_grad_value = prev_grad.data_f32()[0];
                                let broadcast_grad = vec![scalar_grad_value; input_size];
                                let mut new_grad =
                                    TensorData::with_shape_f32(broadcast_grad, input_shape.clone());

                                match grad {
                                    Some(grad) => {
                                        new_grad = new_grad + grad.clone();
                                    }
                                    None => {}
                                }

                                *grad = Some(new_grad.clone());
                                if tensor.requires_grad() {
                                    queue.push_back((tensor.graph(), new_grad));
                                }
                            }
                            _ => todo!(),
                        }
                    });
                }
                TensorOperation::Mul { left, right } => {
                    let right_data = match right {
                        OperatorHandle::Tensor(right) => {
                            let right_tensor = get_tensor(right).unwrap();
                            right_tensor.data_f32().clone()
                        }
                        OperatorHandle::Scalar(_) => vec![1.0],
                    };
                    with_mut_tensor(left, |tensor| {
                        if !tensor.requires_grad() {
                            return;
                        }
                        match tensor {
                            Tensor::F32 { data, grad, .. } => {
                                let new_grad = TensorData::from_vec_f32(right_data);
                                let mut new_grad: TensorData = prev_grad.clone() * new_grad;
                                match grad {
                                    Some(grad) => {
                                        new_grad = new_grad + grad.clone();
                                    }
                                    None => {}
                                }
                                *grad = Some(new_grad.clone());
                                if tensor.requires_grad() {
                                    queue.push_back((tensor.graph(), new_grad));
                                }
                            }
                            _ => todo!(),
                        }
                    });
                }
                TensorOperation::Div { left, right } => {
                    let right_data = match right {
                        OperatorHandle::Tensor(right) => {
                            let right_tensor = get_tensor(right).unwrap();
                            right_tensor.data_f32().clone()
                        }
                        OperatorHandle::Scalar(scalar_val) => {
                            // For scalar denominator, create a vector with the scalar value
                            // that can be used for gradient calculations
                            vec![scalar_val]
                        }
                    };

                    // Gradient for the numerator (left operand): dz/dx = 1/y
                    with_mut_tensor(left, |tensor| {
                        if !tensor.requires_grad() {
                            return;
                        }
                        match tensor {
                            Tensor::F32 { grad, data, .. } => {
                                let mut new_grad = match right {
                                    OperatorHandle::Tensor(_) => {
                                        // For tensor division
                                        let divisor = TensorData::from_vec_f32(right_data);
                                        prev_grad.clone() / divisor
                                    }
                                    OperatorHandle::Scalar(scalar_val) => {
                                        // For scalar division, scale gradient by 1/scalar
                                        prev_grad.clone() * (1.0 / scalar_val)
                                    }
                                };

                                match grad {
                                    Some(existing_grad) => {
                                        new_grad = new_grad + existing_grad.clone();
                                    }
                                    None => {}
                                }

                                *grad = Some(new_grad.clone());
                                if tensor.requires_grad() {
                                    queue.push_back((tensor.graph(), new_grad));
                                }
                            }
                            _ => todo!(),
                        }
                    });

                    // Gradient for the denominator (right operand): dz/dy = -x/y²
                    // For scalar denominators, we don't track gradients since they're not tensors
                    if let OperatorHandle::Tensor(right) = right {
                        let left_tensor = get_tensor(left).unwrap();
                        let left_data = left_tensor.data_f32();

                        with_mut_tensor(right, |tensor| {
                            if !tensor.requires_grad() {
                                return;
                            }
                            match tensor {
                                Tensor::F32 { grad, data, .. } => {
                                    let right_data = data.data_f32();

                                    // Calculate gradient: dz/dy = -x/y²
                                    let mut denom_grad = Vec::with_capacity(right_data.len());
                                    for i in 0..right_data.len() {
                                        denom_grad.push(
                                            -left_data[i] / (right_data[i] * right_data[i])
                                                * prev_grad.data_f32()[0],
                                        );
                                    }

                                    let mut new_grad = TensorData::with_shape_f32(
                                        denom_grad,
                                        data.shape().clone(),
                                    );

                                    match grad {
                                        Some(existing_grad) => {
                                            new_grad = new_grad + existing_grad.clone();
                                        }
                                        None => {}
                                    }

                                    *grad = Some(new_grad.clone());
                                    if tensor.requires_grad() {
                                        queue.push_back((tensor.graph(), new_grad));
                                    }
                                }
                                _ => todo!(),
                            }
                        });
                    }
                }
                TensorOperation::Abs {
                    input,
                    forward_positive,
                } => {
                    with_mut_tensor(input, |tensor| {
                        if !tensor.requires_grad() {
                            return;
                        }
                        match tensor {
                            Tensor::F32 { grad, .. } => {
                                let dir = if forward_positive { 1.0 } else { -1.0 };
                                let mut new_grad: TensorData = prev_grad.clone() * dir;
                                match grad {
                                    Some(grad) => {
                                        new_grad = new_grad + grad.clone();
                                    }
                                    None => {
                                        new_grad = new_grad;
                                    }
                                }
                                *grad = Some(new_grad.clone());
                                if tensor.requires_grad() {
                                    queue.push_back((tensor.graph(), new_grad));
                                }
                            }
                            _ => todo!(),
                        }
                    });
                }
                TensorOperation::None => {}
                TensorOperation::Concat { inputs, dim } => {
                    let mut offset = 0;
                    let mut input_grads = Vec::new();
                    for input in &inputs {
                        let tensor = get_tensor(*input).unwrap();
                        let data = tensor.data_f32();
                        let shape = tensor.shape();
                        let size = shape.iter().product::<usize>();
                        let input_grad = prev_grad.data_f32()[offset..offset + size].to_vec();
                        offset += size;
                        input_grads.push(input_grad);
                    }

                    for (input, input_grad) in inputs.iter().zip(input_grads) {
                        with_mut_tensor(*input, |tensor| {
                            if !tensor.requires_grad() {
                                return;
                            }
                            match tensor {
                                Tensor::F32 { grad, .. } => {
                                    let mut new_grad = TensorData::from_vec_f32(input_grad);
                                    match grad {
                                        Some(existing_grad) => {
                                            new_grad = new_grad + existing_grad.clone();
                                        }
                                        None => {}
                                    }
                                    *grad = Some(new_grad.clone());
                                    if tensor.requires_grad() {
                                        queue.push_back((tensor.graph(), new_grad));
                                    }
                                }
                                _ => todo!(),
                            }
                        });
                    }
                }
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
    fn requires_grad(&self) -> bool {
        match self {
            Tensor::F32 { requires_grad, .. } => *requires_grad,
            _ => todo!(),
        }
    }
    fn graph(&self) -> TensorOperation {
        match self {
            Tensor::F32 { graph, .. } => graph.clone(),
            _ => todo!(),
        }
    }
    fn data_f32(&self) -> Vec<f32> {
        match self {
            Tensor::F32 { data, .. } => data.data_f32().clone(),
            _ => todo!(),
        }
    }
    fn shape(&self) -> Vec<usize> {
        match self {
            Tensor::F32 { data, .. } => data.shape().clone(),
            _ => todo!(),
        }
    }
    fn abs(&self) -> TensorHandle {
        match self {
            Tensor::F32 {
                id,
                data,
                graph,
                grad,
                ..
            } => {
                debug_assert!(data.data_f32().len() > 0);
                let forward_positive = data.data_f32()[0] >= 0.0;
                let new_data = data.data_f32().iter().map(|x| x.abs()).collect();
                Tensor::from_op(
                    new_data,
                    data.shape().clone(),
                    TensorOperation::Abs {
                        input: id.clone(),
                        forward_positive: forward_positive,
                    },
                )
            }
            _ => todo!(),
        }
    }
    // todo should be along axis
    fn sum(&self) -> TensorHandle {
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

    fn concat(&self, other: &Tensor, dim: Option<usize>) -> TensorHandle {
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
                            Tensor::from_op(
                                new_data,
                                new_shape,
                                TensorOperation::Concat {
                                    inputs: vec![id.clone(), other_id.clone()],
                                    dim: Some(0),
                                },
                            )
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
                        if shape1 == shape2 {
                            // Shapes match exactly - perform element-wise operation
                            let new_data = data1.iter().zip(data2.iter()).map(|(a, b)| a $op b).collect();
                            TensorData::with_shape_f32(new_data, shape1.clone())
                        } else if shape1.len() == 1 && shape1[0] == 1 {
                            // Self is a scalar - broadcast to other's shape
                            let scalar_value = data1[0];
                            let new_data = data2.iter().map(|&b| scalar_value $op b).collect();
                            TensorData::with_shape_f32(new_data, shape2.clone())
                        } else if shape2.len() == 1 && shape2[0] == 1 {
                            // Other is a scalar - broadcast to self's shape
                            let scalar_value = data2[0];
                            let new_data = data1.iter().map(|&a| a $op scalar_value).collect();
                            TensorData::with_shape_f32(new_data, shape1.clone())
                        } else {
                            panic!(concat!("Cannot ", $op_name, " tensors of different shapes"))
                        }
                    }
                    // Similar cases for F64...
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
            type Output = TensorHandle;

            fn $method(self, other: &Tensor) -> TensorHandle {
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
            type Output = TensorHandle;

            fn $method(self, other: Tensor) -> TensorHandle {
                &self $op &other
            }
        }

        // Implementation for tensor op scalar
        impl $trait<f32> for &Tensor {
            type Output = TensorHandle;

            fn $method(self, scalar: f32) -> TensorHandle {
                match self {
                    Tensor::F32 { id, data, .. } => {
                        // Use the TensorData operation
                        let new_data = data $op scalar;

                        Tensor::from_op(
                            new_data.data_f32().clone(),
                            new_data.shape().clone(),
                            TensorOperation::$op_enum {
                                left: id.clone(),
                                right: OperatorHandle::Scalar(scalar),
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
            type Output = TensorHandle;

            fn $method(self, scalar: f64) -> TensorHandle {
                match self {
                    Tensor::F32 { id, data, .. } => {
                        // Use the TensorData operation
                        let new_data = data $op scalar;

                        Tensor::from_op(
                            new_data.data_f32().clone(),
                            new_data.shape().clone(),
                            TensorOperation::$op_enum {
                                left: id.clone(),
                                right: OperatorHandle::Scalar(scalar as f32),
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
            type Output = TensorHandle;

            fn $method(self, scalar: f32) -> TensorHandle {
                &self $op scalar
            }
        }

        impl $trait<f64> for Tensor {
            type Output = TensorHandle;

            fn $method(self, scalar: f64) -> TensorHandle {
                &self $op scalar
            }
        }

        // For the reverse operation (scalar op Tensor)
        impl $trait<&Tensor> for f32 {
            type Output = TensorHandle;

            fn $method(self, tensor: &Tensor) -> TensorHandle {
                match tensor {
                    Tensor::F32 { id, data, .. } => {
                        // Use the TensorData operation
                        let new_data = self $op data;

                        Tensor::from_op(
                            new_data.data_f32().clone(),
                            new_data.shape().clone(),
                            TensorOperation::$op_enum {
                                left: id.clone(),
                                right: OperatorHandle::Scalar(self),
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
            type Output = TensorHandle;

            fn $method(self, tensor: &Tensor) -> TensorHandle {
                match tensor {
                    Tensor::F32 { id, data, .. } => {
                        // Use the TensorData operation
                        let new_data = self $op data;

                        Tensor::from_op(
                            new_data.data_f32().clone(),
                            new_data.shape().clone(),
                            TensorOperation::$op_enum {
                                left: id.clone(),
                                right: OperatorHandle::Scalar(self as f32),
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
            type Output = TensorHandle;

            fn $method(self, tensor: Tensor) -> TensorHandle {
                self $op &tensor
            }
        }

        impl $trait<Tensor> for f64 {
            type Output = TensorHandle;

            fn $method(self, tensor: Tensor) -> TensorHandle {
                self $op &tensor
            }
        }
    };
}
macro_rules! impl_tensorhandle_op {
    ($trait:ident, $method:ident, $op:tt, $op_name:expr, $op_enum:ident) => {
        impl $trait for &TensorHandle {
            type Output = TensorHandle;

            fn $method(self, other: &TensorHandle) -> TensorHandle {
                // Get tensors from context
                let tensor1 = get_tensor(self.clone()).unwrap();
                let tensor2 = get_tensor(other.clone()).unwrap();

                let new_tensor = tensor1 $op tensor2;
                new_tensor
            }
        }

        impl $trait for TensorHandle {
            type Output = TensorHandle;

            fn $method(self, other: TensorHandle) -> TensorHandle {
                &self $op &other
            }
        }

        // Implementation for TensorHandle op scalar
        impl $trait<f32> for &TensorHandle {
            type Output = TensorHandle;

            fn $method(self, scalar: f32) -> TensorHandle {
                let tensor = get_tensor(self.clone()).unwrap();
                let new_tensor = tensor $op scalar;
                new_tensor

            }
        }

        impl $trait<f64> for &TensorHandle {
            type Output = TensorHandle;

            fn $method(self, scalar: f64) -> TensorHandle {
                let tensor = get_tensor(self.clone()).unwrap();
                let new_tensor = tensor $op scalar;
                new_tensor
            }
        }

        impl $trait<f32> for TensorHandle {
            type Output = TensorHandle;

            fn $method(self, scalar: f32) -> TensorHandle {
                &self $op scalar
            }
        }

        impl $trait<f64> for TensorHandle {
            type Output = TensorHandle;

            fn $method(self, scalar: f64) -> TensorHandle {
                &self $op scalar
            }
        }

        // For the reverse operation (scalar op TensorHandle)
        impl $trait<&TensorHandle> for f32 {
            type Output = TensorHandle;

            fn $method(self, handle: &TensorHandle) -> TensorHandle {
                let tensor = get_tensor(handle.clone()).unwrap();
                let new_tensor = self $op &tensor;
                new_tensor
            }
        }

        impl $trait<&TensorHandle> for f64 {
            type Output = TensorHandle;

            fn $method(self, handle: &TensorHandle) -> TensorHandle {
                let tensor = get_tensor(handle.clone()).unwrap();
                let new_tensor = self $op &tensor;
                new_tensor

            }
        }

        impl $trait<TensorHandle> for f32 {
            type Output = TensorHandle;

            fn $method(self, handle: TensorHandle) -> TensorHandle {
                self $op &handle
            }
        }

        impl $trait<TensorHandle> for f64 {
            type Output = TensorHandle;

            fn $method(self, handle: TensorHandle) -> TensorHandle {
                self $op &handle
            }
        }
    };
}
impl TensorHandle {
    fn concat(&self, other: &TensorHandle, dim: Option<usize>) -> TensorHandle {
        let tensor1 = get_tensor(self.clone()).unwrap();
        let tensor2 = get_tensor(other.clone()).unwrap();
        tensor1.concat(&tensor2, dim)
    }
    fn sum(&self) -> TensorHandle {
        let tensor = get_tensor(self.clone()).unwrap();
        tensor.sum()
    }
    fn mse(&self, target: &TensorHandle) -> TensorHandle {
        let diff = self - target;
        let squared = diff * diff;
        let sum_squared = squared.sum();
        let n = get_tensor(squared)
            .unwrap()
            .shape()
            .iter()
            .product::<usize>() as f32;
        sum_squared / n
    }
    fn abs(&self) -> TensorHandle {
        let tensor = get_tensor(self.clone()).unwrap();
        tensor.abs()
    }
    fn backward(&self) {
        let tensor = get_tensor(self.clone()).unwrap();
        tensor.backward();
    }
}
// Implement operations for TensorData
impl_tensordata_op!(Add, add, +, "add");
impl_tensordata_op!(Sub, sub, -, "subtract");
impl_tensordata_op!(Mul, mul, *, "multiply");
impl_tensordata_op!(Div, div, /, "divide");

// Implement operations for TensorHandle which operates (now using TensorData operations internally)
impl_tensor_op!(Add, add, +, "add", Add);
impl_tensor_op!(Sub, sub, -, "subtract", Sub);
impl_tensor_op!(Mul, mul, *, "multiply", Mul);
impl_tensor_op!(Div, div, /, "divide", Div);

impl_tensorhandle_op!(Add, add, +, "add", Add);
impl_tensorhandle_op!(Sub, sub, -, "subtract", Sub);
impl_tensorhandle_op!(Mul, mul, *, "multiply", Mul);
impl_tensorhandle_op!(Div, div, /, "divide", Div);

//
fn get_computation_graph(curr_tensor: &Tensor, graph: &mut Vec<TensorOperation>) {
    match curr_tensor {
        Tensor::F32 {
            id,
            data,
            graph: curr_graph,
            grad: None,
            ..
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
                    TensorOperation::Abs { input, .. } => {
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
                                        TensorOperation::Abs { input, .. } => {
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

            requires_grad,
        } => {
            println!(
                "ID: {:?}, data: {:?}, grad: {:?}, requires_grad: {:?}",
                id,
                data.data_f32(),
                grad,
                requires_grad
            );
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
            OperatorHandle::Scalar(scalar) => format!("Add(Tensor({}) + Scalar{})", left.0, scalar),
        },
        TensorOperation::Mul { left, right } => match right {
            OperatorHandle::Tensor(right_id) => {
                format!("Mul(Tensor({}) * Tensor({}))", left.0, right_id.0)
            }
            OperatorHandle::Scalar(scalar) => format!("Mul(Tensor({}) * Scalar{})", left.0, scalar),
        },
        TensorOperation::Sub { left, right } => match right {
            OperatorHandle::Tensor(right_id) => {
                format!("Sub(Tensor({}) - Tensor({}))", left.0, right_id.0)
            }
            OperatorHandle::Scalar(scalar) => format!("Sub(Tensor({}) - Scalar{})", left.0, scalar),
        },
        TensorOperation::Div { left, right } => match right {
            OperatorHandle::Tensor(right_id) => {
                format!("Div(Tensor({}) / Tensor({}))", left.0, right_id.0)
            }
            OperatorHandle::Scalar(scalar) => format!("Div(Tensor({}) / Scalar{})", left.0, scalar),
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
        TensorOperation::Abs {
            input,
            forward_positive,
        } => {
            format!(
                "Abs(Tensor({}), forward_positive: {})",
                input.0, forward_positive
            )
        }
        TensorOperation::None => "None".to_string(),
    }
}
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
        let input_tensor = get_tensor(*input).unwrap();
        let input_shape = input_tensor.shape();
        let batch_size = input_shape[0];

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

        // Add bias
        result.unwrap() + self.bias
    }
}
fn main() {
    let t1 = Tensor::with_shape_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4], false);
    let lin_l = LinearLayer::new(4, 4);
    let wanted = Tensor::with_shape_f32(vec![4.0, 6.0, 2.0, 8.0], vec![4], false);
    let params = vec![lin_l.weights, lin_l.bias];
    for i in 0..20000 {
        let res = lin_l.forward(&t1);
        let loss = res.mse(&wanted);
        TENSOR_CONTEXT.with_borrow_mut(|ctx| {
            for param in &params {
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
        loss.backward();
        TENSOR_CONTEXT.with_borrow_mut(|ctx| {
            for param in &params {
                if let Some(tensor) = ctx.get_mut_tensor(*param) {
                    match tensor {
                        Tensor::F32 { id, data, grad, .. } => {
                            if let Some(grad) = grad {
                                let grad_data = grad.data_f32();
                                let data = data.mut_data_f32();

                                debug_assert!(grad_data.len() == data.len());
                                for i in 0..data.len() {
                                    data[i] -= grad_data[i] * 0.001;
                                }
                            }
                        }
                        _ => todo!(),
                    }
                }
            }
        });
        if (i + 1) % 100 == 0 {
            println!("Loss: {:?}", get_tensor(loss).unwrap().data_f32());
        }
    }

    for param in params {
        let tensor = get_tensor(param).unwrap();
        println!("Param: {:?}, Value: {:?}", param, tensor.data_f32());
    }
    println!(
        "Result: {:?}",
        get_tensor(lin_l.forward(&t1)).unwrap().data_f32()
    );
}
