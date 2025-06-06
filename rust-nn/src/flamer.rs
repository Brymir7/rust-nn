use crate::optimizer::Optimizer;
use crate::tensor::{get_tensor, Tensor, TensorHandle, TENSOR_CONTEXT};
use std::marker::PhantomData;

pub struct Flamer<'a> {
    _phantom: PhantomData<&'a ()>,
    start_index: usize,
}

impl<'a> Flamer<'a> {
    pub fn new() -> Self {
        let start_index =
            TENSOR_CONTEXT.with_borrow(|ctx| ctx.tensor_cache.op_result_pointers.len());

        TENSOR_CONTEXT.with_borrow_mut(|ctx| {
            ctx.tensor_cache.start_op_index = start_index;
        });

        Self {
            _phantom: PhantomData,
            start_index,
        }
    }

    pub fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        f()
    }

    pub fn current_index(&self) -> usize {
        TENSOR_CONTEXT.with_borrow(|ctx| ctx.tensor_cache.current_op_index)
    }

    pub fn training_step<F, R>(
        &self,
        optimizer: &mut impl Optimizer,
        params: &[TensorHandle],
        batch_fn: F,
    ) -> R
    where
        F: FnOnce() -> R,
    {
        optimizer.zero_grad(params);
        optimizer.prepare_next_iteration();

        let result = batch_fn();

        optimizer.step(params);

        result
    }
}

impl<'a> Drop for Flamer<'a> {
    fn drop(&mut self) {
        TENSOR_CONTEXT.with_borrow_mut(|ctx| {
            ctx.tensor_cache.current_op_index = self.start_index;
        });
    }
}

pub fn with_training_step<F, R>(
    optimizer: &mut impl Optimizer,
    params: &[TensorHandle],
    batch_fn: F,
) -> R
where
    F: FnOnce() -> R,
{
    let flamer = Flamer::new();
    flamer.training_step(optimizer, params, batch_fn)
}

pub fn predict<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let flamer = Flamer::new();
    flamer.execute(f)
}

pub fn get_predicted_class(output: TensorHandle) -> usize {
    let output_data = get_tensor(output).unwrap();
    let output_data = output_data.data_f32();
    let mut max_idx = 0;
    let mut max_val = output_data[0];

    for j in 1..output_data.len() {
        if output_data[j] > max_val {
            max_val = output_data[j];
            max_idx = j;
        }
    }

    max_idx
}
