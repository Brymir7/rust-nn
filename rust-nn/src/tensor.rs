#![allow(dead_code)]
use std::fmt::Debug;
use std::process::id;
use std::{
    cell::{RefCell, RefMut},
    collections::VecDeque,
    ops::{Add, Div, Mul, Sub},
};

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
pub struct TensorHandle(pub usize);
// Used to reuse tensors (especially intermediary tensors) in consecutive loops
pub struct OperationCache {
    pub op_result_pointers: Vec<TensorHandle>,
    pub current_op_index: usize,
    pub start_op_index: usize,
}
impl OperationCache {
    fn new() -> Self {
        Self {
            op_result_pointers: Vec::new(),
            current_op_index: 0,
            start_op_index: 0,
        }
    }
    pub fn next_iteration(&mut self) {
        self.current_op_index = self.start_op_index;
    }
}
pub struct TensorContext {
    pub all_tensors: Vec<Tensor>,
    pub tensor_cache: OperationCache,
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
    pub fn get_mut_tensor(&mut self, id: TensorHandle) -> Option<&mut Tensor> {
        self.all_tensors.get_mut(id.0)
    }
}

thread_local! {
    pub static TENSOR_CONTEXT: RefCell<TensorContext> = RefCell::new(TensorContext {
        all_tensors: Vec::new(),
        tensor_cache: OperationCache::new(),
    });
}

fn register_tensor(tensor: Tensor) -> TensorHandle {
    TensorHandle(TENSOR_CONTEXT.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        ctx.register_tensor(tensor)
    }))
}

pub fn get_tensor(id: TensorHandle) -> Option<Tensor> {
    TENSOR_CONTEXT.with(|ctx| {
        let ctx = ctx.borrow();
        ctx.get_tensor(id).cloned()
    })
}
fn get_tensor_with_ctx(ctx: &RefMut<'_, TensorContext>, id: TensorHandle) -> Option<Tensor> {
    ctx.get_tensor(id).cloned()
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
fn with_mut_tensor_ctx<F, R>(
    ctx: &mut RefMut<'_, TensorContext>,
    id: TensorHandle,
    f: F,
) -> Option<R>
where
    F: FnOnce(&mut Tensor) -> R,
{
    let tensor = ctx.get_mut_tensor(id)?;
    Some(f(tensor))
}

fn get_next_tensor_id() -> usize {
    TENSOR_CONTEXT.with(|ctx| {
        let ctx = ctx.borrow();
        ctx.get_next_id()
    })
}
fn get_next_tensor_id_with_ctx(ctx: &RefMut<'_, TensorContext>) -> usize {
    ctx.get_next_id()
}

#[derive(Clone, Debug)]
pub enum OperatorHandle {
    Tensor(TensorHandle),
    Scalar(f32),
}
#[derive(Clone, Debug)]
pub enum TensorOperation {
    Add {
        left: TensorHandle,
        right: OperatorHandle,
    },
    Mul {
        left: TensorHandle,
        right: OperatorHandle,
    },
    Sub {
        left: TensorHandle,
        right: OperatorHandle,
    },
    Div {
        left: TensorHandle,
        right: OperatorHandle,
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
pub enum TensorData {
    F32 { data: Vec<f32>, shape: Vec<usize> },
    F64 { data: Vec<f64>, shape: Vec<usize> },
}
impl TensorData {
    pub fn shape(&self) -> &Vec<usize> {
        match self {
            TensorData::F32 { shape, .. } => shape,
            TensorData::F64 { shape, .. } => shape,
        }
    }

    pub fn data_f32(&self) -> &Vec<f32> {
        match self {
            TensorData::F32 { data, .. } => data,
            _ => panic!("Not an f32 tensor"),
        }
    }
    pub fn mut_data_f32(&mut self) -> &mut Vec<f32> {
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

    pub fn to_f32_vec(&self) -> Vec<f32> {
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

    pub fn size(&self) -> usize {
        match self {
            TensorData::F32 { data, .. } => data.len(),
            TensorData::F64 { data, .. } => data.len(),
        }
    }
}

#[derive(Clone)]
pub enum Tensor {
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
    pub fn new_f32(data: Vec<f32>, shape: Option<Vec<usize>>, req_grad: bool) -> TensorHandle {
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

        TENSOR_CONTEXT.with(|ctx| {
            let mut ctx_ref = ctx.borrow_mut();
            let tensor_data = TensorData::F32 { data, shape };
            if ctx_ref.tensor_cache.current_op_index < ctx_ref.tensor_cache.op_result_pointers.len()
            {
                let cache = &mut ctx_ref.tensor_cache;
                let handle = cache.op_result_pointers[cache.current_op_index];
                cache.current_op_index += 1;
                with_mut_tensor_ctx(&mut ctx_ref, handle, |tensor| match tensor {
                    Tensor::F32 {
                        ref mut data,
                        ref mut graph,
                        ref mut grad,
                        ref mut requires_grad,
                        ..
                    } => {
                        *data = tensor_data;
                        *graph = TensorOperation::None;
                        *grad = None;
                        *requires_grad = req_grad;
                    }
                    _ => panic!("Expected F32 tensor"),
                });

                handle
            } else {
                let tensor = Tensor::F32 {
                    id: TensorHandle(ctx_ref.get_next_id()),
                    data: tensor_data,
                    graph: TensorOperation::None,
                    grad: None,
                    requires_grad: req_grad,
                };

                let handle = TensorHandle(ctx_ref.register_tensor(tensor));
                let cache = &mut ctx_ref.tensor_cache;
                cache.op_result_pointers.push(handle);
                cache.current_op_index += 1;
                handle
            }
        })
    }

    fn new_f64(data: Vec<f64>, shape: Option<Vec<usize>>, _requires_grad: bool) -> TensorHandle {
        todo!("Implement f64 tensors")
    }

    fn from_op(data: Vec<f32>, shape: Vec<usize>, op: TensorOperation) -> TensorHandle {
        TENSOR_CONTEXT.with(|ctx| {
            let mut ctx_ref = ctx.borrow_mut();
            let tensor_data = TensorData::F32 { data, shape };
            if ctx_ref.tensor_cache.current_op_index < ctx_ref.tensor_cache.op_result_pointers.len()
            {
                let cache = &mut ctx_ref.tensor_cache;
                let handle = cache.op_result_pointers[cache.current_op_index];
                cache.current_op_index += 1;
                with_mut_tensor_ctx(&mut ctx_ref, handle, |tensor| match tensor {
                    Tensor::F32 {
                        ref mut data,
                        ref mut graph,
                        ref mut grad,
                        ..
                    } => {
                        *data = tensor_data;
                        *graph = op;
                        *grad = None;
                    }
                    _ => panic!("Expected F32 tensor"),
                });

                handle
            } else {
                let tensor =
                    Self::create_or_update_tensor(Some(&ctx_ref), None, tensor_data, op, true);
                let handle = TensorHandle(ctx_ref.register_tensor(tensor));
                let cache = &mut ctx_ref.tensor_cache;
                cache.op_result_pointers.push(handle);
                cache.current_op_index += 1;
                handle
            }
        })
    }
    fn create_or_update_tensor(
        ctx: Option<&RefMut<'_, TensorContext>>,
        id_handle: Option<TensorHandle>,
        tensor_data: TensorData,
        op: TensorOperation,
        requires_grad: bool,
    ) -> Tensor {
        match id_handle {
            Some(handle) => match ctx {
                Some(ctx) => {
                    let mut existing = get_tensor_with_ctx(ctx, handle).unwrap();
                    match existing {
                        Tensor::F32 {
                            id: _,
                            ref mut data,
                            ref mut graph,
                            ref mut grad,
                            requires_grad: _,
                        } => {
                            *data = tensor_data;
                            *graph = op;
                            *grad = None;
                            existing
                        }
                        _ => panic!("Expected F32 tensor"),
                    }
                }
                None => {
                    let mut existing = get_tensor(handle).unwrap();
                    match existing {
                        Tensor::F32 {
                            id: _,
                            ref mut data,
                            ref mut graph,
                            ref mut grad,
                            requires_grad: _,
                        } => {
                            *data = tensor_data;
                            *graph = op;
                            *grad = None;
                            existing
                        }
                        _ => panic!("Expected F32 tensor"),
                    }
                }
            },

            None => {
                let next_id = match ctx {
                    Some(ctx) => get_next_tensor_id_with_ctx(ctx),
                    None => get_next_tensor_id(),
                };
                return Tensor::F32 {
                    id: TensorHandle(next_id),
                    data: tensor_data,
                    graph: op,
                    grad: None,
                    requires_grad,
                };
            }
        }
    }
    pub fn with_shape_f32(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> TensorHandle {
        Self::new_f32(data, Some(shape), requires_grad)
    }
    fn with_shape_f64(data: Vec<f64>, shape: Vec<usize>, requires_grad: bool) -> TensorHandle {
        Self::new_f64(data, Some(shape), requires_grad)
    }
    pub fn from_vec_f32(data: Vec<f32>, requires_grad: bool) -> TensorHandle {
        Self::new_f32(data, None, requires_grad)
    }
    pub fn random_f32(shape: Vec<usize>, requires_grad: bool) -> TensorHandle {
        let data = (0..shape.iter().product())
            .map(|_| rand::random::<f32>())
            .collect();
        Self::new_f32(data, Some(shape), requires_grad)
    }
    pub fn backward(&self) {
        let mut queue: VecDeque<(TensorOperation, TensorData)> = VecDeque::new();
        match self {
            Tensor::F32 { graph, data, .. } => {
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
                    with_mut_tensor(left, |tensor| {
                        if !tensor.requires_grad() {
                            return;
                        }

                        match tensor {
                            Tensor::F32 { grad, data, .. } => {
                                let shape = data.shape().clone();
                                let mut new_grad = TensorData::F32 {
                                    data: prev_grad.data_f32().clone(),
                                    shape: shape.clone(),
                                };

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

                    match right {
                        OperatorHandle::Tensor(right) => {
                            with_mut_tensor(right, |tensor| {
                                if !tensor.requires_grad() {
                                    return;
                                }

                                match tensor {
                                    Tensor::F32 { grad, data, .. } => {
                                        let shape = data.shape().clone();
                                        let mut new_grad = TensorData::F32 {
                                            data: prev_grad.data_f32().clone(),
                                            shape: shape.clone(),
                                        };

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
                TensorOperation::Sub { left, right } => {
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
                            vec![scalar_val]
                        }
                    };

                    // Gradient for the numerator (left operand): dz/dx = 1/y
                    with_mut_tensor(left, |tensor| {
                        if !tensor.requires_grad() {
                            return;
                        }
                        match tensor {
                            Tensor::F32 { grad, .. } => {
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
                TensorOperation::Concat { inputs, dim: _ } => {
                    let mut offset = 0;
                    let mut input_grads = Vec::new();
                    for input in &inputs {
                        let tensor = get_tensor(*input).unwrap();
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
    pub fn grad_f32(&self) -> f32 {
        match self {
            Tensor::F32 { grad, .. } => match grad {
                Some(grad) => grad.data_f32()[0],
                None => 0.0,
            },
            _ => todo!(),
        }
    }
    pub fn requires_grad(&self) -> bool {
        match self {
            Tensor::F32 { requires_grad, .. } => *requires_grad,
            _ => todo!(),
        }
    }
    pub fn graph(&self) -> TensorOperation {
        match self {
            Tensor::F32 { graph, .. } => graph.clone(),
            _ => todo!(),
        }
    }
    pub fn data_f32(&self) -> Vec<f32> {
        match self {
            Tensor::F32 { data, .. } => data.data_f32().clone(),
            _ => todo!(),
        }
    }
    pub fn shape(&self) -> Vec<usize> {
        match self {
            Tensor::F32 { data, .. } => data.shape().clone(),
            _ => todo!(),
        }
    }
    pub fn abs(&self) -> TensorHandle {
        match self {
            Tensor::F32 { id, data, .. } => {
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
    pub fn sum(&self) -> TensorHandle {
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

    pub fn concat(&self, other: &Tensor, dim: Option<usize>) -> TensorHandle {
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
                            let inner_dim = shape[dim + 1..].iter().product();
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
                            panic!("Cannot {} tensors of different shapes ({:?}D vs {:?}D)", $op_name, shape1.len(), shape2.len())
                        }
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
                            id: _,
                            data: _,
                            ..
                        },
                        Tensor::F64 {
                            id: _,
                            data: _,
                            ..
                        },
                    ) => {
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
                    Tensor::F64 {   .. } => {
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
                    Tensor::F64 {   .. } => {
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

        impl $trait<&Tensor> for f32 {
            type Output = TensorHandle;

            fn $method(self, tensor: &Tensor) -> TensorHandle {
                match tensor {
                    Tensor::F32 { id, data, .. } => {
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
                    Tensor::F64 {  .. } => {
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
                    Tensor::F64 {  .. } => {
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
    pub fn concat(&self, other: &TensorHandle, dim: Option<usize>) -> TensorHandle {
        let tensor1 = get_tensor(self.clone()).unwrap();
        let tensor2 = get_tensor(other.clone()).unwrap();
        tensor1.concat(&tensor2, dim)
    }
    pub fn sum(&self) -> TensorHandle {
        let tensor = get_tensor(self.clone()).unwrap();
        tensor.sum()
    }
    pub fn mse(&self, target: &TensorHandle) -> TensorHandle {
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
    // todo reimplement with a reshape
    pub fn flatten(&self) -> TensorHandle {
        let tensor = get_tensor(self.clone()).unwrap();
        let shape = tensor.shape();
        let size = shape.iter().product::<usize>();
        let new_shape = vec![size];
        let data = tensor.data_f32().clone();
        Tensor::from_op(data, new_shape, TensorOperation::None)
    }
    pub fn abs(&self) -> TensorHandle {
        let tensor = get_tensor(self.clone()).unwrap();
        tensor.abs()
    }
    pub fn backward(&self) {
        let tensor = get_tensor(self.clone()).unwrap();
        tensor.backward();
    }
}
impl_tensordata_op!(Add, add, +, "add");
impl_tensordata_op!(Sub, sub, -, "subtract");
impl_tensordata_op!(Mul, mul, *, "multiply");
impl_tensordata_op!(Div, div, /, "divide");

impl_tensor_op!(Add, add, +, "add", Add);
impl_tensor_op!(Sub, sub, -, "subtract", Sub);
impl_tensor_op!(Mul, mul, *, "multiply", Mul);
impl_tensor_op!(Div, div, /, "divide", Div);

impl_tensorhandle_op!(Add, add, +, "add", Add);
impl_tensorhandle_op!(Sub, sub, -, "subtract", Sub);
impl_tensorhandle_op!(Mul, mul, *, "multiply", Mul);
impl_tensorhandle_op!(Div, div, /, "divide", Div);
