pub mod utils {

    use crate::tensor::{get_tensor, OperatorHandle, Tensor, TensorOperation};

    pub fn print_computation_graph(tensor: &Tensor) {
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

    pub fn print_tensor_data(tensor: &Tensor) {
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

    pub fn format_operation(op: &TensorOperation) -> String {
        match op {
            TensorOperation::Add { left, right } => match right {
                OperatorHandle::Tensor(right_id) => {
                    format!("Add(Tensor({}) + Tensor({}))", left.0, right_id.0)
                }
                OperatorHandle::Scalar(scalar) => {
                    format!("Add(Tensor({}) + Scalar{})", left.0, scalar)
                }
            },
            TensorOperation::Mul { left, right } => match right {
                OperatorHandle::Tensor(right_id) => {
                    format!("Mul(Tensor({}) * Tensor({}))", left.0, right_id.0)
                }
                OperatorHandle::Scalar(scalar) => {
                    format!("Mul(Tensor({}) * Scalar{})", left.0, scalar)
                }
            },
            TensorOperation::Sub { left, right } => match right {
                OperatorHandle::Tensor(right_id) => {
                    format!("Sub(Tensor({}) - Tensor({}))", left.0, right_id.0)
                }
                OperatorHandle::Scalar(scalar) => {
                    format!("Sub(Tensor({}) - Scalar{})", left.0, scalar)
                }
            },
            TensorOperation::Div { left, right } => match right {
                OperatorHandle::Tensor(right_id) => {
                    format!("Div(Tensor({}) / Tensor({}))", left.0, right_id.0)
                }
                OperatorHandle::Scalar(scalar) => {
                    format!("Div(Tensor({}) / Scalar{})", left.0, scalar)
                }
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

    /// Gets the computation graph for a tensor
    pub fn get_computation_graph(curr_tensor: &Tensor, graph: &mut Vec<TensorOperation>) {
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
                                                if let OperatorHandle::Tensor(right_handle) = right
                                                {
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
}
