#![allow(dead_code)]
use core::panic;
use ndarray::{Array, ArrayD, Axis, IxDyn};
use std::fmt::{self, Debug};
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

#[derive(Clone, Debug, Copy, Hash, PartialEq, Eq)]
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

    pub fn get_tensor(&self, id: TensorHandle) -> Option<&Tensor> {
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
    Max {
        input: TensorHandle,
        threshold: f32,
        mask: Vec<bool>, //  which inputs were greater than threshold (for backprop)
    },
    Exp {
        input: TensorHandle,
    },
    Log {
        input: TensorHandle,
    },
    // we use this op because we use ndarray for mat mul so it doesnt use our binary ops to autograd
    MatMul {
        input: TensorHandle,
        weights: TensorHandle,
    },
    None,
}

// New TensorData implementation using ndarray
#[derive(Clone, Debug)]
pub enum TensorData {
    F32 { data: ArrayD<f32> },
    F64 { data: ArrayD<f64> },
}

impl TensorData {
    pub fn shape(&self) -> Vec<usize> {
        match self {
            TensorData::F32 { data } => data.shape().to_vec(),
            TensorData::F64 { data } => data.shape().to_vec(),
        }
    }

    pub fn data_f32(&self) -> &ArrayD<f32> {
        match self {
            TensorData::F32 { data } => data,
            _ => panic!("Not an f32 tensor"),
        }
    }

    pub fn mut_data_f32(&mut self) -> &mut ArrayD<f32> {
        match self {
            TensorData::F32 { data } => data,
            _ => panic!("Not an f32 tensor"),
        }
    }

    fn data_f64(&self) -> &ArrayD<f64> {
        match self {
            TensorData::F64 { data } => data,
            _ => panic!("Not an f64 tensor"),
        }
    }

    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            TensorData::F32 { data } => data.iter().cloned().collect(),
            TensorData::F64 { data } => data.iter().map(|&x| x as f32).collect(),
        }
    }

    fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            TensorData::F32 { data } => data.iter().map(|&x| x as f64).collect(),
            TensorData::F64 { data } => data.iter().cloned().collect(),
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
            TensorData::F32 { data } => data.len(),
            TensorData::F64 { data } => data.len(),
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
                requires_grad,
                ..
            } => {
                write!(f, "Tensor::F32 {:?}{{\n", id)?;
                write!(f, "  data: {:?}\n", data.data_f32())?;
                write!(f, "  graph: {:?}\n", graph)?;
                write!(f, "  grad: {:?}\n", grad)?;
                write!(f, "  requires_grad: {:?}\n}}", requires_grad)
            }
            _ => todo!(),
        }
    }
}

impl Tensor {
    pub fn new_f32(data: Vec<f32>, shape: Option<Vec<usize>>, req_grad: bool) -> TensorHandle {
        let array_data = match shape {
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
                Array::from_shape_vec(IxDyn(&shape), data).unwrap()
            }
            None => Array::from_shape_vec(IxDyn(&[data.len()]), data).unwrap(),
        };
        // check storage to avoid reallocating every loop
        TENSOR_CONTEXT.with(|ctx| {
            let mut ctx_ref = ctx.borrow_mut();
            let tensor_data = TensorData::F32 { data: array_data };

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

    pub fn from_op(data: ArrayD<f32>, op: TensorOperation) -> TensorHandle {
        TENSOR_CONTEXT.with(|ctx| {
            let mut ctx_ref = ctx.borrow_mut();
            let tensor_data = TensorData::F32 { data };

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
                        *graph = op;
                        *grad = None;
                        *requires_grad = true;
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
                            requires_grad: ref mut r,
                        } => {
                            *data = tensor_data;
                            *graph = op;
                            *grad = None;
                            *r = requires_grad;
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
                            requires_grad: ref mut r,
                        } => {
                            *data = tensor_data;
                            *graph = op;
                            *grad = None;
                            *r = requires_grad;
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
        let n_in = shape[0];
        let n_out = if shape.len() > 1 { shape[1] } else { 1 };

        let scale = (6.0 / (n_in + n_out) as f32).sqrt();

        let data = (0..shape.iter().product())
            .map(|_| rand::random::<f32>() * 2.0 * scale - scale)
            .collect();

        Self::new_f32(data, Some(shape), requires_grad)
    }

    pub fn backward(&self) {
        let mut queue: VecDeque<(TensorOperation, TensorData)> = VecDeque::new();

        match self {
            Tensor::F32 { graph, data, .. } => {
                let shape = data.shape();
                let ones_array = Array::ones(IxDyn(&shape));
                queue.push_back((graph.clone(), TensorData::F32 { data: ones_array }));
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
                                let mut new_grad = prev_grad.clone();

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
                                        let mut new_grad = prev_grad.clone();

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
                                let mut new_grad = prev_grad.clone();
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
                                let scalar_grad_value = *prev_grad.data_f32().first().unwrap();
                                let broadcast_grad =
                                    Array::from_elem(IxDyn(&input_shape), scalar_grad_value);
                                let new_grad = TensorData::F32 {
                                    data: broadcast_grad,
                                };

                                match grad {
                                    Some(grad) => {
                                        let combined_grad = new_grad + grad.clone();
                                        *grad = combined_grad.clone();
                                        if tensor.requires_grad() {
                                            queue.push_back((tensor.graph(), combined_grad));
                                        }
                                    }
                                    None => {
                                        *grad = Some(new_grad.clone());
                                        if tensor.requires_grad() {
                                            queue.push_back((tensor.graph(), new_grad));
                                        }
                                    }
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
                        OperatorHandle::Scalar(scalar_val) => {
                            Array::from_elem(IxDyn(&[1]), scalar_val)
                        }
                    };
                    with_mut_tensor(left, |tensor| {
                        if !tensor.requires_grad() {
                            return;
                        }
                        match tensor {
                            Tensor::F32 { data, grad, .. } => {
                                let right_tensor_data = TensorData::F32 { data: right_data };
                                let mut new_grad = prev_grad.clone() * right_tensor_data;
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
                            Array::from_elem(IxDyn(&[1]), scalar_val)
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
                                        let divisor = TensorData::F32 {
                                            data: right_data.clone(),
                                        };
                                        prev_grad.clone() / divisor
                                    }
                                    OperatorHandle::Scalar(scalar_val) => {
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
                                    let neg_left = -left_data;
                                    let right_squared = right_data * right_data;
                                    let mut denom_grad = &neg_left / &right_squared;

                                    denom_grad *= prev_grad.data_f32();

                                    let new_grad = TensorData::F32 { data: denom_grad };

                                    match grad {
                                        Some(existing_grad) => {
                                            let combined = new_grad + existing_grad.clone();
                                            *grad = Some(combined.clone());
                                            if tensor.requires_grad() {
                                                queue.push_back((tensor.graph(), combined));
                                            }
                                        }
                                        None => {
                                            *grad = Some(new_grad.clone());
                                            if tensor.requires_grad() {
                                                queue.push_back((tensor.graph(), new_grad));
                                            }
                                        }
                                    }
                                }
                                _ => todo!(),
                            }
                        });
                    }
                }
                TensorOperation::MatMul { input, weights } => {
                    let weights_tensor = get_tensor(weights).unwrap();
                    // handle input tensor grad
                    with_mut_tensor(input, |tensor| {
                        if !tensor.requires_grad() {
                            return;
                        }
                        match tensor {
                            Tensor::F32 { grad, .. } => {
                                let weights_data = weights_tensor.data_f32();
                                let prev_grad_data = prev_grad.data_f32();

                                // For input gradient: dL/dY · W^T

                                let result = match (prev_grad_data.ndim(), weights_data.ndim()) {
                                    (2, 2) => {
                                        let p = prev_grad_data
                                            .clone()
                                            .into_dimensionality::<ndarray::Ix2>()
                                            .unwrap();
                                        let w = weights_data
                                            .clone()
                                            .into_dimensionality::<ndarray::Ix2>()
                                            .unwrap();
                                        let w = w.t();

                                        p.dot(&w).into_dyn()
                                    }
                                    (1, 2) => {
                                        // If prev_grad is vector and weights is matrix
                                        let p = prev_grad_data
                                            .clone()
                                            .into_dimensionality::<ndarray::Ix1>()
                                            .unwrap();
                                        let w = weights_data
                                            .clone()
                                            .into_dimensionality::<ndarray::Ix2>()
                                            .unwrap();
                                        let w = w.t();

                                        p.dot(&w).into_dyn()
                                    }
                                    _ => {
                                        panic!(
                                            "Unsupported tensor dimensions for matmul: {} and {}",
                                            prev_grad_data.ndim(),
                                            weights_data.ndim()
                                        );
                                    }
                                };

                                let mut new_grad = TensorData::F32 { data: result };

                                match grad {
                                    Some(existing_grad) => {
                                        println!(
                                            "Existing grad shape: {:?} tensor {:?}",
                                            existing_grad.shape(),
                                            input
                                        );
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
                    // handle weights tensor grad
                    let input_tensor = get_tensor(input).unwrap();
                    with_mut_tensor(weights, |tensor| {
                        if !tensor.requires_grad() {
                            return;
                        }
                        match tensor {
                            Tensor::F32 { grad, .. } => {
                                let input_data = input_tensor.data_f32();
                                let prev_grad_data = prev_grad.data_f32();

                                // X^T * dL/dY
                                let result = match (input_data.ndim(), prev_grad_data.ndim()) {
                                    (2, 2) => {
                                        // Normal matrix multiplication case
                                        let x_t = input_data
                                            .t()
                                            .clone()
                                            .into_dimensionality::<ndarray::Ix2>()
                                            .unwrap();
                                        let dy = prev_grad_data
                                            .clone()
                                            .into_dimensionality::<ndarray::Ix2>()
                                            .unwrap();
                                        x_t.dot(&dy).into_dyn()
                                    }
                                    (1, 1) => {
                                        let x = input_data
                                            .clone()
                                            .into_dimensionality::<ndarray::Ix1>()
                                            .unwrap();
                                        let dy = prev_grad_data
                                            .clone()
                                            .into_dimensionality::<ndarray::Ix1>()
                                            .unwrap();

                                        let x_col = x.insert_axis(ndarray::Axis(1));
                                        let dy_row = dy.insert_axis(ndarray::Axis(0));

                                        x_col.dot(&dy_row).into_dyn()
                                    }
                                    (1, 2) => {
                                        let x = input_data
                                            .clone()
                                            .into_dimensionality::<ndarray::Ix1>()
                                            .unwrap();
                                        let dy = prev_grad_data
                                            .clone()
                                            .into_dimensionality::<ndarray::Ix2>()
                                            .unwrap();

                                        let x_col = x.insert_axis(ndarray::Axis(1));

                                        x_col.dot(&dy).into_dyn()
                                    }
                                    (2, 1) => {
                                        let x_t = input_data
                                            .t()
                                            .clone()
                                            .into_dimensionality::<ndarray::Ix2>()
                                            .unwrap();
                                        let dy = prev_grad_data
                                            .clone()
                                            .into_dimensionality::<ndarray::Ix1>()
                                            .unwrap();

                                        let dy_row = dy.insert_axis(ndarray::Axis(0));

                                        x_t.dot(&dy_row).into_dyn()
                                    }
                                    _ => {
                                        panic!(
                                            "Unsupported tensor dimensions for weight gradient: {} and {}",
                                            input_data.ndim(),
                                            prev_grad_data.ndim()
                                        );
                                    }
                                };

                                let mut new_grad = TensorData::F32 { data: result };
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
                TensorOperation::None => {}
                TensorOperation::Concat { inputs, dim } => {
                    let dim = dim.unwrap_or(0);
                    let mut offset = 0;

                    for input_handle in &inputs {
                        with_mut_tensor(*input_handle, |tensor| {
                            if !tensor.requires_grad() {
                                return;
                            }

                            match tensor {
                                Tensor::F32 { grad, data, .. } => {
                                    let input_shape = data.shape();
                                    let mut indices = vec![];

                                    for i in 0..prev_grad.data_f32().ndim() {
                                        if i == dim {
                                            let slice_size = input_shape[i];
                                            indices.push((offset, offset + slice_size));
                                            offset += slice_size;
                                        } else {
                                            indices.push((0, prev_grad.data_f32().shape()[i]));
                                        }
                                    }

                                    let input_grad = prev_grad.data_f32().slice(ndarray::s![..]);
                                    let new_grad = TensorData::F32 {
                                        data: input_grad.to_owned().into_dyn(),
                                    };

                                    match grad {
                                        Some(existing_grad) => {
                                            let combined = new_grad + existing_grad.clone();
                                            *grad = Some(combined.clone());
                                            queue.push_back((tensor.graph(), combined));
                                        }
                                        None => {
                                            *grad = Some(new_grad.clone());
                                            queue.push_back((tensor.graph(), new_grad));
                                        }
                                    }
                                }
                                _ => todo!(),
                            }
                        });
                    }
                }
                TensorOperation::Max { input, mask, .. } => {
                    with_mut_tensor(input, |tensor| {
                        if !tensor.requires_grad() {
                            return;
                        }
                        match tensor {
                            Tensor::F32 { grad, data, .. } => {
                                let input_shape = data.shape();
                                let mut filtered_grad_array = Array::zeros(IxDyn(&input_shape));

                                // Apply mask to gradient - only pass gradient where input > threshold
                                for (i, &pass) in mask.iter().enumerate() {
                                    if pass {
                                        let idx = ndarray::indices(input_shape.clone());
                                        let mut linear_count = 0;
                                        for multi_idx in idx {
                                            if linear_count == i {
                                                filtered_grad_array[&multi_idx] =
                                                    prev_grad.data_f32().as_slice().unwrap()[i];
                                                break;
                                            }
                                            linear_count += 1;
                                        }
                                    }
                                }

                                let new_grad = TensorData::F32 {
                                    data: filtered_grad_array,
                                };

                                match grad {
                                    Some(existing_grad) => {
                                        let combined = new_grad + existing_grad.clone();
                                        *grad = Some(combined.clone());
                                        queue.push_back((tensor.graph(), combined));
                                    }
                                    None => {
                                        *grad = Some(new_grad.clone());
                                        queue.push_back((tensor.graph(), new_grad));
                                    }
                                }
                            }
                            _ => todo!(),
                        }
                    });
                }
                TensorOperation::Exp { input } => {
                    with_mut_tensor(input, |tensor| {
                        if !tensor.requires_grad() {
                            return;
                        }
                        match tensor {
                            Tensor::F32 { grad, data, .. } => {
                                let input_data = data.data_f32();
                                let exp_data = input_data.mapv(|x| x.exp());
                                let new_grad = TensorData::F32 {
                                    data: prev_grad.data_f32() * exp_data,
                                };

                                match grad {
                                    Some(existing_grad) => {
                                        let combined = new_grad + existing_grad.clone();
                                        *grad = Some(combined.clone());
                                        queue.push_back((tensor.graph(), combined));
                                    }
                                    None => {
                                        *grad = Some(new_grad.clone());
                                        queue.push_back((tensor.graph(), new_grad));
                                    }
                                }
                            }
                            _ => todo!(),
                        }
                    });
                }
                TensorOperation::Log { input } => {
                    with_mut_tensor(input, |tensor| {
                        if !tensor.requires_grad() {
                            return;
                        }
                        match tensor {
                            Tensor::F32 { grad, data, .. } => {
                                let input_data = data.data_f32();
                                let new_grad = TensorData::F32 {
                                    data: prev_grad.data_f32() / input_data,
                                };
                                match grad {
                                    Some(existing_grad) => {
                                        let combined = new_grad + existing_grad.clone();
                                        *grad = Some(combined.clone());
                                        queue.push_back((tensor.graph(), combined));
                                    }
                                    None => {
                                        *grad = Some(new_grad.clone());
                                        queue.push_back((tensor.graph(), new_grad));
                                    }
                                }
                            }
                            _ => todo!(),
                        }
                    });
                }
            }
        }
    }

    pub fn grad_f32(&self) -> f32 {
        match self {
            Tensor::F32 { grad, .. } => match grad {
                Some(grad) => *grad.data_f32().first().unwrap_or(&0.0),
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

    pub fn data_f32(&self) -> &ArrayD<f32> {
        match self {
            Tensor::F32 { data, .. } => &data.data_f32(),
            _ => todo!(),
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match self {
            Tensor::F32 { data, .. } => data.shape(),
            _ => todo!(),
        }
    }
    pub fn exp(&self) -> TensorHandle {
        match self {
            Tensor::F32 { id, data, .. } => {
                let new_data = data.data_f32().mapv(|x| x.exp());
                Tensor::from_op(new_data, TensorOperation::Exp { input: id.clone() })
            }
            _ => todo!(),
        }
    }

    pub fn log(&self) -> TensorHandle {
        match self {
            Tensor::F32 { id, data, .. } => {
                let new_data = data.data_f32().mapv(|x| x.max(1e-10).ln());
                Tensor::from_op(new_data, TensorOperation::Log { input: id.clone() })
            }
            _ => todo!(),
        }
    }
    pub fn abs(&self) -> TensorHandle {
        match self {
            Tensor::F32 { id, data, .. } => {
                let forward_positive = *data.data_f32().first().unwrap_or(&0.0) >= 0.0;
                let new_data = data.data_f32().mapv(|x| x.abs());

                Tensor::from_op(
                    new_data,
                    TensorOperation::Abs {
                        input: id.clone(),
                        forward_positive,
                    },
                )
            }
            _ => todo!(),
        }
    }
    // todo add along dim
    pub fn sum(&self) -> TensorHandle {
        match self {
            Tensor::F32 { id, data, .. } => {
                let sum = data.data_f32().sum();
                let new_data = Array::from_elem(IxDyn(&[1]), sum);

                Tensor::from_op(new_data, TensorOperation::Sum { input: id.clone() })
            }
            _ => todo!(),
        }
    }

    pub fn max(&self, threshold: f32) -> TensorHandle {
        match self {
            Tensor::F32 { id, data, .. } => {
                let data_f32 = data.data_f32();
                let mask: Vec<bool> = data_f32.iter().map(|&x| x > threshold).collect();
                let new_data = data_f32.mapv(|x| if x > threshold { x } else { threshold });

                Tensor::from_op(
                    new_data,
                    TensorOperation::Max {
                        input: id.clone(),
                        mask,
                        threshold,
                    },
                )
            }
            _ => todo!(),
        }
    }

    pub fn concat(&self, other: &Tensor, dim: Option<usize>) -> TensorHandle {
        match (self, other) {
            (
                Tensor::F32 { id, data, .. },
                Tensor::F32 {
                    id: other_id,
                    data: other_data,
                    ..
                },
            ) => {
                let axis = dim.unwrap_or(0);

                let left_array = data.data_f32();
                let right_array = other_data.data_f32();
                let result = ndarray::stack(Axis(axis), &[left_array.view(), right_array.view()])
                    .expect("Arrays could not be concatenated");

                Tensor::from_op(
                    result,
                    TensorOperation::Concat {
                        inputs: vec![id.clone(), other_id.clone()],
                        dim,
                    },
                )
            }
            _ => panic!("Cannot concatenate tensors of different types"),
        }
    }
}

impl TensorData {
    fn new_f32(data: Vec<f32>, shape: Option<Vec<usize>>) -> Self {
        let array_data = match shape {
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
                Array::from_shape_vec(IxDyn(&shape), data).unwrap()
            }
            None => Array::from_shape_vec(IxDyn(&[data.len()]), data).unwrap(),
        };

        TensorData::F32 { data: array_data }
    }

    fn new_f64(data: Vec<f64>, shape: Option<Vec<usize>>) -> Self {
        let array_data = match shape {
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
                Array::from_shape_vec(IxDyn(&shape), data).unwrap()
            }
            None => Array::from_shape_vec(IxDyn(&[data.len()]), data).unwrap(),
        };

        TensorData::F64 { data: array_data }
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
fn can_broadcast(shape1: &[usize], shape2: &[usize]) -> bool {
    let mut i1 = shape1.len();
    let mut i2 = shape2.len();

    while i1 > 0 || i2 > 0 {
        let dim1 = if i1 > 0 { shape1[i1 - 1] } else { 1 };
        let dim2 = if i2 > 0 { shape2[i2 - 1] } else { 1 };

        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            return false;
        }

        if i1 > 0 {
            i1 -= 1;
        }
        if i2 > 0 {
            i2 -= 1;
        }
    }

    true
}
macro_rules! impl_tensordata_op {
    ($trait:ident, $method:ident, $op:tt, $op_name:expr) => {
        impl $trait for &TensorData {
            type Output = TensorData;

            fn $method(self, other: &TensorData) -> TensorData {
                match (self, other) {
                    (
                        TensorData::F32 { data: data1 },
                        TensorData::F32 { data: data2 },
                    ) => {
                        if data1.shape() == data2.shape() {
                            // Shapes match exactly - perform element-wise operation
                            let result = std::panic::catch_unwind(|| {
                                data1 $op data2
                            });

                            match result {
                                Ok(data) => TensorData::F32 { data },
                                Err(_) => {
                                    panic!("Operation '{}' failed with matching shapes:\n  - Shape: {:?}\n  - Left values: {:?}\n  - Right values: {:?}",
                                           $op_name, data1.shape(),
                                           if data1.len() <= 10 { format!("{:?}", data1) } else { format!("[{} elements]", data1.len()) },
                                           if data2.len() <= 10 { format!("{:?}", data2) } else { format!("[{} elements]", data2.len()) })
                                }
                            }
                        } else if data1.len() == 1 {
                            // Self is a scalar - broadcast to other's shape
                            let scalar_value = data1.first().unwrap();
                            let result = std::panic::catch_unwind(|| {
                                *scalar_value $op data2
                            });

                            match result {
                                Ok(data) => TensorData::F32 { data },
                                Err(_) => {
                                    panic!("Operation '{}' failed with scalar broadcast:\n  - Scalar: {:?}\n  - Shape: {:?}\n  - Right values: {:?}",
                                           $op_name, scalar_value, data2.shape(),
                                           if data2.len() <= 10 { format!("{:?}", data2) } else { format!("[{} elements]", data2.len()) })
                                }
                            }
                        } else if data2.len() == 1 {
                            // Other is a scalar - broadcast to self's shape
                            let scalar_value = data2.first().unwrap();
                            let result = std::panic::catch_unwind(|| {
                                data1 $op *scalar_value
                            });

                            match result {
                                Ok(data) => TensorData::F32 { data },
                                Err(_) => {
                                    panic!("Operation '{}' failed with scalar broadcast:\n  - Shape: {:?}\n  - Scalar: {:?}\n  - Left values: {:?}",
                                           $op_name, data1.shape(), scalar_value,
                                           if data1.len() <= 10 { format!("{:?}", data1) } else { format!("[{} elements]", data1.len()) })
                                }
                            }
                        } else {
                            let shape1 = data1.shape();
                            let shape2 = data2.shape();
                            if can_broadcast(shape1, shape2) {
                                let result = std::panic::catch_unwind(|| {
                                    data1 $op data2
                                });

                                match result {
                                    Ok(data) => TensorData::F32 { data },
                                    Err(_) => {
                                        panic!("Operation '{}' failed during broadcasting:\n  - Left shape: {:?}\n  - Right shape: {:?}\n  - Left values: {:?}\n  - Right values: {:?}",
                                               $op_name, shape1, shape2,
                                               if data1.len() <= 10 { format!("{:?}", data1) } else { format!("[{} elements]", data1.len()) },
                                               if data2.len() <= 10 { format!("{:?}", data2) } else { format!("[{} elements]", data2.len()) })
                                    }
                                }
                            } else {
                                panic!("Cannot {} tensors of different shapes, shape 1: {:?}, shape 2: {:?}",
                                       $op_name, shape1, shape2)
                            }
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
                    TensorData::F32 { data } => {
                        TensorData::F32 { data: data.clone() $op scalar }
                    }
                    TensorData::F64 { data } => {
                        TensorData::F64 { data: data.clone() $op scalar as f64 }
                    }
                }
            }
        }

        impl $trait<f64> for &TensorData {
            type Output = TensorData;

            fn $method(self, scalar: f64) -> TensorData {
                match self {
                    TensorData::F32 { data } => {
                        TensorData::F32 { data: data.clone() $op scalar as f32 }
                    }
                    TensorData::F64 { data } => {
                        TensorData::F64 { data: data.clone() $op scalar }
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
                    TensorData::F32 { data } => {
                        TensorData::F32 { data: self $op data.clone() }
                    }
                    TensorData::F64 { data } => {
                        TensorData::F64 { data: self as f64 $op data.clone() }
                    }
                }
            }
        }

        impl $trait<&TensorData> for f64 {
            type Output = TensorData;

            fn $method(self, tensor_data: &TensorData) -> TensorData {
                match tensor_data {
                    TensorData::F32 { data } => {
                        TensorData::F32 { data: self as f32 $op data.clone() }
                    }
                    TensorData::F64 { data } => {
                        TensorData::F64 { data: self $op data.clone() }
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
                            TensorOperation::$op_enum {
                                left: id1.clone(),
                                right: OperatorHandle::Tensor(id2.clone()),
                            }
                        )
                    }
                    _ => panic!(concat!("Cannot ", $op_name, " tensors of different types")),
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
                        let new_data = data $op scalar;
                        Tensor::from_op(
                            new_data.data_f32().clone(),
                            TensorOperation::$op_enum {
                                left: id.clone(),
                                right: OperatorHandle::Scalar(scalar),
                            }
                        )
                    }
                    Tensor::F64 { .. } => todo!(),
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
                            TensorOperation::$op_enum {
                                left: id.clone(),
                                right: OperatorHandle::Scalar(scalar as f32),
                            }
                        )
                    }
                    Tensor::F64 { .. } => todo!(),
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
                            TensorOperation::$op_enum {
                                left: id.clone(),
                                right: OperatorHandle::Scalar(self),
                            }
                        )
                    }
                    Tensor::F64 { .. } => todo!(),
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
                            TensorOperation::$op_enum {
                                left: id.clone(),
                                right: OperatorHandle::Scalar(self as f32),
                            }
                        )
                    }
                    Tensor::F64 { .. } => todo!(),
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
    pub fn exp(&self) -> TensorHandle {
        let tensor = get_tensor(self.clone()).unwrap();
        tensor.exp()
    }

    pub fn log(&self) -> TensorHandle {
        let tensor = get_tensor(self.clone()).unwrap();
        tensor.log()
    }
    pub fn sum(&self) -> TensorHandle {
        let tensor = get_tensor(self.clone()).unwrap();
        tensor.sum()
    }
    pub fn softmax(&self) -> TensorHandle {
        let tensor = get_tensor(self.clone()).unwrap();
        let input_data = tensor.data_f32();
        let max_val = input_data.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let shifted = self - max_val;
        let exp_values = shifted.exp();
        let sum_exp = exp_values.sum();
        exp_values / sum_exp
    }
    pub fn cross_entropy(&self, target: &TensorHandle) -> TensorHandle {
        let probabilities = self.softmax();
        let log_probs = probabilities.log();
        let product = *target * log_probs;
        let sum_product = product.sum();
        sum_product * -1.0
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

    pub fn flatten(&self) -> TensorHandle {
        let tensor = get_tensor(self.clone()).unwrap();

        match tensor {
            Tensor::F32 {
                data: ref tensor_data,
                ..
            } => {
                let size = tensor_data.size();
                let flattened = tensor_data
                    .data_f32()
                    .clone()
                    .into_shape(IxDyn(&[size]))
                    .unwrap();
                Tensor::from_op(flattened, TensorOperation::None)
            }
            _ => todo!(),
        }
    }

    pub fn abs(&self) -> TensorHandle {
        let tensor = get_tensor(self.clone()).unwrap();
        tensor.abs()
    }

    pub fn relu(&self) -> TensorHandle {
        let tensor = get_tensor(self.clone()).unwrap();
        tensor.max(0.0)
    }

    pub fn backward(&self) {
        let tensor = get_tensor(self.clone()).unwrap();
        tensor.backward();
    }
}
impl fmt::Display for TensorHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match get_tensor(*self) {
            Some(tensor) => write!(f, "{}", format_tensor(&tensor)),
            None => write!(f, "Tensor({:?}): <not found>", self),
        }
    }
}
pub fn format_tensor(tensor: &Tensor) -> String {
    match tensor {
        Tensor::F32 {
            id,
            data,
            grad,
            requires_grad,
            ..
        } => {
            let grad_str = match grad {
                Some(g) => format!("\ngrad: {}", tensor_data_to_string(g)),
                None => "".to_string(),
            };

            format!(
                "Tensor({:?}) [requires_grad={}]:\n{}{}",
                id,
                requires_grad,
                tensor_data_to_string(data),
                grad_str
            )
        }
        Tensor::F64 { id, data, grad, .. } => {
            let grad_str = match grad {
                Some(g) => format!("\ngrad: {}", tensor_data_to_string(g)),
                None => "".to_string(),
            };

            format!(
                "Tensor({:?}):\n{}{}",
                id,
                tensor_data_to_string(data),
                grad_str
            )
        }
    }
}
fn tensor_data_to_string(data: &TensorData) -> String {
    match data {
        TensorData::F32 { data } => {
            let shape = data.shape();
            let total_elements = data.len();

            if total_elements <= 100 {
                if shape.len() == 1 {
                    format!(
                        "shape: {:?}, data: {:?}",
                        shape,
                        data.as_slice().unwrap_or(&[])
                    )
                } else if shape.len() == 2 {
                    let mut result = format!("shape: {:?}\n", shape);
                    for i in 0..shape[0] {
                        let row: Vec<_> = (0..shape[1])
                            .map(|j| format!("{:.6}", data[[i, j]]))
                            .collect();
                        result.push_str(&format!("  [{}]\n", row.join(", ")));
                    }
                    result
                } else {
                    format!("shape: {:?}, data: {:?}", shape, data)
                }
            } else {
                // For large tensors, show shape and sample values
                format!(
                    "shape: {:?}, size: {}, sample: [{:.6}, {:.6}, ... {:.6}, {:.6}]",
                    shape,
                    total_elements,
                    data.as_slice().unwrap_or(&[])[0],
                    data.as_slice().unwrap_or(&[])[1],
                    data.as_slice().unwrap_or(&[])[total_elements - 2],
                    data.as_slice().unwrap_or(&[])[total_elements - 1]
                )
            }
        }
        TensorData::F64 { data } => {
            let shape = data.shape();
            let total_elements = data.len();

            if total_elements <= 100 {
                if shape.len() == 1 {
                    format!(
                        "shape: {:?}, data: {:?}",
                        shape,
                        data.as_slice().unwrap_or(&[])
                    )
                } else if shape.len() == 2 {
                    let mut result = format!("shape: {:?}\n", shape);
                    for i in 0..shape[0] {
                        let row: Vec<_> = (0..shape[1])
                            .map(|j| format!("{:.6}", data[[i, j]]))
                            .collect();
                        result.push_str(&format!("  [{}]\n", row.join(", ")));
                    }
                    result
                } else {
                    format!("shape: {:?}, data: {:?}", shape, data)
                }
            } else {
                format!(
                    "shape: {:?}, size: {}, sample: [{:.6}, {:.6}, ... {:.6}, {:.6}]",
                    shape,
                    total_elements,
                    data.as_slice().unwrap_or(&[])[0],
                    data.as_slice().unwrap_or(&[])[1],
                    data.as_slice().unwrap_or(&[])[total_elements - 2],
                    data.as_slice().unwrap_or(&[])[total_elements - 1]
                )
            }
        }
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
