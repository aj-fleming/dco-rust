use num::{Num, One, Zero};
use std::cell::RefCell;
use std::ops::{Add, Mul};

#[cfg(test)]
mod tests;

// use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

/// Marks a scalar number as differentiable.
pub trait DifferentiableScalar {}

impl DifferentiableScalar for f32 {}
impl DifferentiableScalar for f64 {}

#[derive(Clone, Copy)]
#[repr(C)] // (necessary for re-interpreting the struct as something else.)
/// Definition of forward-mode types for automatic differentiation.
///
/// The "innermost" `T` should implement `DifferentiableScalar`.
pub struct Tangent<T> {
    v: T,
    dv: T,
}

#[derive(Clone, Copy)]
pub struct Adjoint<'tape, T, TAPE>
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
    TAPE: AdjointTape<T> + ?Sized,
{
    v: T,
    id: usize,
    tape: Option<&'tape TAPE>,
}

pub trait AdjointTape<T>
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
{
    /// Get the number of independent variables on this tape.
    fn num_independents(&self) -> usize;
    /// Get the number of dependent variables on this tape.
    fn num_dependents(&self) -> usize;
    /// Get a new independent adjoint variable on this tape
    /// with its value intialized to `v`
    fn new_independent_variable<'tape>(&'tape mut self, v: T) -> Adjoint<'tape, T, Self>;

    /// Get the size of the adjoint vector necessary to fully interpret the tape.
    /// That is, how many elements of `T` must be allocated to to interpret the tape?
    fn adjoint_vector_size(&self) -> usize;
    /// Get an empty adjoint vector (intialized to `T::zero`)
    fn empty_seed(&self) -> Vec<T>;

    /// Get the size of the vector of dependencies.
    fn num_dependencies(&self) -> usize;
    /// Get the size of the vector of derivative information.
    fn num_pullbacks(&self) -> usize;
    /// Get the number of adjoint variables allocated during recording.
    fn num_adjoints(&self) -> usize;

    /// Record a derivative `vbar` with respect to `v` to the tape.
    fn record_derivative<'a>(&self, v: &Adjoint<'a, T, Self>, vbar: T);
    /// Record `result`, which has been produced by an operation with `nargs` arguments, onto the tape.
    /// Updates the `id` of the result.
    fn record_result<'a>(&self, result: &mut Adjoint<'a, T, Self>, nargs: usize);
    /// Reset the tape.
    fn scrub(&mut self);

    /// Get the position where `adjoint_id` is stored in the vector of adjoints
    fn adjoint_vector_idx(&self, adjoint_id: usize) -> usize;
    /// Get the pullback value at `tape_position`
    fn taped_pullback_value_at(&self, tape_position: usize) -> T;
    /// Get the dependency tape information at `tape_position`
    fn taped_dependency_at(&self, tape_position: usize) -> usize;

    fn interpret(&self, seed: &mut Vec<T>) {
        let mut dependency_tape_pos: usize = self.num_dependencies() - 1;
        let mut pullback_tape_pos: usize = self.num_pullbacks() - 1;
        // the first entries on the tape are the labels for the independent vars
        // so we don't need to keep reading backwards to their dependencies
        // since there aren't any
        while dependency_tape_pos > self.num_independents() - 1 {
            // portions of the dependency tape look like this:
            // ... ID1 ID2 ID3 NARGS ADJID ...
            // so we read backwards, decrementing by 1 after acquiring piece of information
            let result_id: usize = self.taped_dependency_at(dependency_tape_pos);
            let result_adj_idx: usize = self.adjoint_vector_idx(result_id);
            dependency_tape_pos -= 1;
            let num_pullbacks: usize = self.taped_dependency_at(dependency_tape_pos);
            dependency_tape_pos -= 1;
            let result_pullback: T = seed[result_adj_idx];
            seed[result_adj_idx] = T::zero();
            for _ in 0..num_pullbacks {
                let pb_id: usize = self.taped_dependency_at(dependency_tape_pos);
                let pullback_idx: usize = self.adjoint_vector_idx(pb_id);
                dependency_tape_pos -= 1;
                let arg_pullback: T = self.taped_pullback_value_at(pullback_tape_pos);
                pullback_tape_pos -= 1;
                seed[pullback_idx] = seed[pullback_idx] + result_pullback * arg_pullback;
            }
        }
    }
}

pub struct MinimalTape<T> {
    n_inputs: usize,
    n_outputs: usize,
    n_adjoints: RefCell<usize>,

    adj_deps: RefCell<Vec<usize>>,
    pullbacks: RefCell<Vec<T>>,
}

impl<T> MinimalTape<T>
where
    T: Copy + Zero,
{
    fn new() -> MinimalTape<T> {
        MinimalTape {
            n_inputs: 0,
            n_outputs: 0,
            n_adjoints: RefCell::new(0),
            adj_deps: RefCell::new(vec![usize::zero(); 0]),
            pullbacks: RefCell::new(Vec::new()),
        }
    }
}

impl<T> AdjointTape<T> for MinimalTape<T>
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
{
    #[inline]
    fn num_independents(&self) -> usize {
        self.n_inputs
    }

    #[inline]
    fn num_dependents(&self) -> usize {
        self.n_outputs
    }

    fn new_independent_variable<'tape>(&'tape mut self, val: T) -> Adjoint<'tape, T, Self> {
        let new_id: usize = { *(self.n_adjoints.borrow()) };
        self.n_adjoints.replace_with(|&mut n| n + 1);
        Adjoint {
            v: val,
            id: new_id,
            tape: Some(self),
        }
    }

    #[inline]
    fn adjoint_vector_size(&self) -> usize {
        self.num_adjoints()
    }

    #[inline]
    fn empty_seed(&self) -> Vec<T> {
        vec![T::zero(); self.adjoint_vector_size()]
    }

    #[inline]
    fn num_dependencies(&self) -> usize {
        self.adj_deps.borrow().len()
    }

    #[inline]
    fn num_pullbacks(&self) -> usize {
        self.pullbacks.borrow().len()
    }

    #[inline]
    fn num_adjoints(&self) -> usize {
        *(self.n_adjoints.borrow())
    }

    fn record_derivative<'a>(&self, v: &Adjoint<'a, T, MinimalTape<T>>, vbar: T) {
        self.adj_deps.borrow_mut().push(v.id);
        self.pullbacks.borrow_mut().push(vbar);
    }

    fn record_result<'a>(&self, v: &mut Adjoint<'a, T, MinimalTape<T>>, nargs: usize) {
        v.id = *(self.n_adjoints.borrow());
        self.n_adjoints.replace_with(|&mut n| n + 1);
        let mut adj_deps = self.adj_deps.borrow_mut();
        adj_deps.push(nargs);
        adj_deps.push(v.id);
    }

    fn scrub(&mut self) {
        self.n_adjoints.replace(0);
        self.n_inputs = 0;
        self.n_outputs = 0;
        self.adj_deps.borrow_mut().clear();
        self.pullbacks.borrow_mut().clear();
    }

    #[inline]
    fn adjoint_vector_idx(&self, adjoint_id: usize) -> usize {
        adjoint_id
    }

    #[inline]
    fn taped_pullback_value_at(&self, tape_position: usize) -> T {
        self.pullbacks.borrow()[tape_position]
    }

    #[inline]
    fn taped_dependency_at(&self, tape_position: usize) -> usize {
        self.adj_deps.borrow()[tape_position]
    }
}

//
// VALUE AND DERIVATIVE METHODS
//

trait PassiveValue {
    type PassiveValueType;

    /// Unwrap every layer of `Tangent<...>` around some value `v`.
    fn passive_value(self) -> Self::PassiveValueType;
}

//
// PassiveValue might be better specialized on f32, f64... maybe also on smaller floats?
//

impl<T: DifferentiableScalar> PassiveValue for T {
    type PassiveValueType = Self;

    #[inline]
    fn passive_value(self) -> Self::PassiveValueType {
        self
    }
}

impl<T: PassiveValue> PassiveValue for Tangent<T> {
    type PassiveValueType = T::PassiveValueType;

    #[inline]
    fn passive_value(self) -> Self::PassiveValueType {
        self.v.passive_value()
    }
}

impl<'a, T, TAPE> PassiveValue for Adjoint<'a, T, TAPE>
where
    T: PassiveValue + Copy + Num,
    TAPE: AdjointTape<T>,
{
    type PassiveValueType = T::PassiveValueType;

    fn passive_value(self) -> Self::PassiveValueType {
        self.v.passive_value()
    }
}
trait HighestOrderDerivative {
    type HighestOrderDerivativeType;

    fn highest_order_derivative(self) -> Self::HighestOrderDerivativeType;
}

impl<T: DifferentiableScalar> HighestOrderDerivative for T {
    type HighestOrderDerivativeType = Self;

    #[inline]
    fn highest_order_derivative(self) -> Self::HighestOrderDerivativeType {
        self
    }
}

impl<T: HighestOrderDerivative> HighestOrderDerivative for Tangent<T> {
    type HighestOrderDerivativeType = T::HighestOrderDerivativeType;

    #[inline]
    fn highest_order_derivative(self) -> Self::HighestOrderDerivativeType {
        self.dv.highest_order_derivative()
    }
}

//
// OPERATIONS ON TANGENTS
//

mod fwd_ops;
pub use self::fwd_ops::*;

impl<T: Zero> Tangent<T> {
    /// Take some passive value `arg` and turn it into a Tangent with zero derivative (constant)
    #[inline]
    pub fn make_constant(arg: impl Into<T>) -> Tangent<T> {
        Self {
            v: arg.into(),
            dv: T::zero(),
        }
    }
}

impl<T: One> Tangent<T> {
    /// Take some passive value `arg` and turn it into a Tangent with unity derivative (active)
    #[inline]
    pub fn make_active(arg: impl Into<T>) -> Tangent<T> {
        Self {
            v: arg.into(),
            dv: T::one(),
        }
    }
}

//
// OPERATIONS ON ADJOINTS
//

mod rev_ops;
pub use self::rev_ops::*;
