use num::traits::real::Real;
use num::{Num, NumCast, One, ToPrimitive, Zero};
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

// use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

/// Marks a scalar number as differentiable.
pub trait DifferentiableScalar {}

impl DifferentiableScalar for f32 {}
impl DifferentiableScalar for f64 {}

#[derive(Clone, Copy)]
#[repr(C)] // (necessary for re-interpreting the struct as something else.)
/// Definition of forward-mode types for automatic differentiation.
///
///
/// The "innermost" `T` should implement `DifferentiableScalar`.
pub struct Tangent<T> {
    v: T,
    dv: T,
}

pub struct Adjoint<'tape, T, TAPE>
where
    T: Copy + Num,
    TAPE: AdjointTape<T> + ?Sized,
{
    v: T,
    id: usize,
    tape: &'tape TAPE,
}

impl<'tape, T, TAPE> Adjoint<'tape, T, TAPE>
where
    T: Copy + Num,
    TAPE: AdjointTape<T>,
{
}

pub trait AdjointTape<T>
where
    T: Copy + Num,
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
    T: Copy + Num,
{
    fn new() -> MinimalTape<T> {
        MinimalTape {
            n_inputs: 0,
            n_outputs: 0,
            n_adjoints: RefCell::new(0),
            adj_deps: RefCell::new(vec![usize::zero(); 0]),
            pullbacks: RefCell::new(vec![T::zero(); 0]),
        }
    }
}

impl<T> AdjointTape<T> for MinimalTape<T>
where
    T: Copy + Num,
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
            tape: self,
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

trait DualNumberEq {
    fn dual_eq(&self, other: &Self) -> bool;
}

impl<T: PartialEq> DualNumberEq for Tangent<T> {
    #[inline]
    fn dual_eq(&self, other: &Self) -> bool {
        self.v == other.v && self.dv == other.dv
    }
}

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
// EQUALITY AND ORDERING
//

impl<T: PartialEq> PartialEq<Tangent<T>> for Tangent<T> {
    #[inline]
    /// We discard the derivative information when checking equality.
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&self.v, &other.v)
    }
}

impl<T: PartialOrd> PartialOrd<Tangent<T>> for Tangent<T> {
    #[inline]
    // We discard the derivative information when comparing.
    fn partial_cmp(&self, other: &Tangent<T>) -> Option<std::cmp::Ordering> {
        PartialOrd::partial_cmp(&self.v, &other.v)
    }
}

//
// ZERO AND ONE
//

impl<T: Zero> Zero for Tangent<T> {
    #[inline]
    fn zero() -> Self {
        Self {
            v: T::zero(),
            dv: T::zero(),
        }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.v.is_zero()
    }
}

impl<T: Copy + One + Zero + PartialEq> One for Tangent<T> {
    #[inline]
    fn one() -> Self {
        Self {
            v: T::one(),
            dv: T::zero(),
        }
    }

    #[inline]
    fn is_one(&self) -> bool
    where
        Self: PartialEq,
    {
        self.v.is_one()
    }
}

//
// NUMOPS AND NUM
//

impl<T> Add for Tangent<T>
where
    T: Add<T, Output = T>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            v: self.v + rhs.v,
            dv: self.dv + rhs.dv,
        }
    }
}

impl<T> Sub for Tangent<T>
where
    T: Sub<T, Output = T>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            v: self.v - rhs.v,
            dv: self.dv - rhs.dv,
        }
    }
}

impl<T> Mul for Tangent<T>
where
    T: Copy + Mul<T, Output = T> + Add<T, Output = T>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            v: self.v * rhs.v,
            dv: self.dv * rhs.v + rhs.dv * self.v,
        }
    }
}

impl<T> Neg for Tangent<T>
where
    T: Neg<Output = T>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            v: self.v.neg(),
            dv: self.dv.neg(),
        }
    }
}

impl<T> Div for Tangent<T>
where
    T: Copy + Div<T, Output = T> + Mul<T, Output = T> + Sub<T, Output = T>,
{
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self {
            v: self.v / rhs.v,
            dv: (self.dv * rhs.v - rhs.dv * self.v) / (rhs.v * rhs.v),
        }
    }
}

impl<T> Rem for Tangent<T> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl<T> Num for Tangent<T>
where
    T: Copy + Num,
{
    type FromStrRadixErr = T::FromStrRadixErr;

    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(src, radix).map(Tangent::make_constant)
    }
}

//
// NUMCAST AND REAL
//

impl<T: NumCast + Zero> NumCast for Tangent<T> {
    fn from<V: ToPrimitive>(n: V) -> Option<Self> {
        T::from(n).map(Self::make_constant)
    }
}

impl<T: ToPrimitive> ToPrimitive for Tangent<T> {
    fn to_isize(&self) -> Option<isize> {
        self.v.to_isize()
    }

    fn to_i8(&self) -> Option<i8> {
        self.v.to_i8()
    }

    fn to_i16(&self) -> Option<i16> {
        self.v.to_i16()
    }

    fn to_i32(&self) -> Option<i32> {
        self.v.to_i32()
    }

    fn to_i64(&self) -> Option<i64> {
        self.v.to_i64()
    }

    fn to_usize(&self) -> Option<usize> {
        self.v.to_usize()
    }

    fn to_u8(&self) -> Option<u8> {
        self.v.to_u8()
    }

    fn to_u16(&self) -> Option<u16> {
        self.v.to_u16()
    }

    fn to_u32(&self) -> Option<u32> {
        self.v.to_u32()
    }

    fn to_u64(&self) -> Option<u64> {
        self.v.to_u64()
    }

    fn to_f32(&self) -> Option<f32> {
        self.v.to_f32()
    }

    fn to_f64(&self) -> Option<f64> {
        self.v.to_f64()
    }
}

impl<T> Real for Tangent<T>
where
    T: Real,
{
    /// Get the minimum value of a Tangent.
    /// Does {min, min} make sense here?
    #[inline]
    fn min_value() -> Self {
        Self {
            v: T::min_value(),
            dv: T::min_value(),
        }
    }

    #[inline]
    fn min_positive_value() -> Self {
        Self {
            v: T::min_positive_value(),
            dv: T::min_positive_value(),
        }
    }

    #[inline]
    fn epsilon() -> Self {
        Self {
            v: T::epsilon(),
            dv: T::epsilon(),
        }
    }

    #[inline]
    fn max_value() -> Self {
        Self {
            v: T::max_value(),
            dv: T::max_value(),
        }
    }

    /// We do not round the derivative.
    #[inline]
    fn floor(self) -> Self {
        Self {
            v: self.v.floor(),
            dv: self.dv,
        }
    }

    /// We do not round the derivative.
    fn ceil(self) -> Self {
        Self {
            v: self.v.ceil(),
            dv: self.dv,
        }
    }

    /// We do not round the derivative.
    fn round(self) -> Self {
        Self {
            v: self.v.round(),
            dv: self.dv,
        }
    }

    /// We do not truncate the derivative.
    fn trunc(self) -> Self {
        Self {
            v: self.v.trunc(),
            dv: self.dv,
        }
    }

    fn fract(self) -> Self {
        Self {
            v: self.v.fract(),
            dv: self.dv,
        }
    }

    fn abs(self) -> Self {
        Self {
            v: self.v.abs(),
            dv: if self.v >= T::zero() {
                self.dv
            } else {
                -self.dv
            },
        }
    }

    fn signum(self) -> Self {
        Tangent::make_constant(self.v.signum())
    }

    fn is_sign_positive(self) -> bool {
        self.v.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.v.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Self {
            // f = a*x + b
            v: self.v.mul_add(a.v, b.v),
            // f' = a' * x + (a * x' + b')
            dv: self.dv.mul_add(self.v, self.dv.mul_add(a.v, b.dv)),
        }
    }

    fn recip(self) -> Self {
        Self {
            v: self.v.recip(),
            dv: -self.dv.recip() * self.dv.recip(),
        }
    }

    /// Panics if T is not constructible from i32
    fn powi(self, n: i32) -> Self {
        Self {
            v: self.v.powi(n),
            dv: self.dv * T::from(n).unwrap() * (self.v.powi(n - 1)),
        }
    }

    fn powf(self, n: Self) -> Self {
        todo!()
    }

    fn sqrt(self) -> Self {
        let sqrtvalue = self.v.sqrt();
        Self {
            v: sqrtvalue,
            dv: {
                // we need to avoid division by "close to zero" if possible
                // dsqrt = 1/(2sqrt(x))dx
                let denom = sqrtvalue * T::from(2).unwrap();
                if denom == T::zero() && self.dv == T::zero() {
                    T::zero()
                } else {
                    self.dv / denom
                }
            },
        }
    }

    fn exp(self) -> Self {
        Self {
            v: self.v.exp(),
            dv: self.dv * self.v.exp(),
        }
    }

    fn exp2(self) -> Self {
        let exp2value = self.v.exp2();
        Self {
            v: exp2value,
            dv: self.dv * T::from(2).unwrap().ln() * exp2value,
        }
    }

    fn ln(self) -> Self {
        Self {
            v: self.v.ln(),
            dv: self.dv * self.v.recip(),
        }
    }

    // Here we use log rules to write log_a(arg) = ln(arg) / ln(a)
    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    fn log2(self) -> Self {
        Self {
            v: self.v.log2(),
            dv: self.dv * T::from(2).unwrap().ln().recip() * self.v.recip(),
        }
    }

    fn log10(self) -> Self {
        Self {
            v: self.v.log10(),
            dv: self.dv * T::from(10).unwrap().ln().recip() * self.v.recip(),
        }
    }

    fn to_degrees(self) -> Self {
        Self {
            v: self.v.to_degrees(),
            dv: self.dv.to_degrees(),
        }
    }

    fn to_radians(self) -> Self {
        Self {
            v: self.v.to_radians(),
            dv: self.dv.to_radians(),
        }
    }

    fn max(self, other: Self) -> Self {
        if self.v < other.v {
            other
        } else {
            self
        }
    }

    fn min(self, other: Self) -> Self {
        if self.v > other.v {
            other
        } else {
            self
        }
    }

    fn abs_sub(self, other: Self) -> Self {
        if self > other {
            Self {
                v: self.v.abs_sub(other.v),
                dv: self.dv - other.dv,
            }
        } else {
            Self::zero()
        }
    }

    fn cbrt(self) -> Self {
        let cbrtvalue = self.v.cbrt();
        Self {
            v: cbrtvalue,
            dv: {
                let denom = cbrtvalue.powi(2) * T::from(3).unwrap();
                if denom == T::zero() && self.dv == T::zero() {
                    T::zero()
                } else {
                    T::from(2).unwrap() * self.dv / denom
                }
            },
        }
    }

    fn hypot(self, other: Self) -> Self {
        let hypotvalue = self.v.hypot(other.v);
        Self {
            v: hypotvalue,
            dv: (T::from(2).unwrap())
                * (self.dv * self.v + other.dv * other.v)
                * hypotvalue.recip(),
        }
    }

    #[inline]
    fn sin(self) -> Self {
        Self {
            v: self.v.sin(),
            dv: self.dv * self.v.cos(),
        }
    }

    #[inline]
    fn cos(self) -> Self {
        Self {
            v: self.v.cos(),
            dv: -self.dv * self.v.sin(),
        }
    }

    #[inline]
    fn tan(self) -> Self {
        Self {
            v: self.v.tan(),
            dv: self.dv / (self.v.cos() * self.v.cos()),
        }
    }

    fn asin(self) -> Self {
        todo!()
    }

    fn acos(self) -> Self {
        todo!()
    }

    fn atan(self) -> Self {
        todo!()
    }

    fn atan2(self, other: Self) -> Self {
        todo!()
    }

    fn sin_cos(self) -> (Self, Self) {
        let (sinvalue, cosvalue) = self.v.sin_cos();
        (
            Self {
                v: sinvalue,
                dv: self.dv * cosvalue,
            },
            Self {
                v: cosvalue,
                dv: -self.dv * sinvalue,
            },
        )
    }

    fn exp_m1(self) -> Self {
        todo!()
    }

    #[inline]
    fn ln_1p(self) -> Self {
        Self {
            v: self.v.ln_1p(),
            dv: self.dv * (T::one() + self.v).recip(),
        }
    }

    fn sinh(self) -> Self {
        todo!()
    }

    fn cosh(self) -> Self {
        todo!()
    }

    fn tanh(self) -> Self {
        todo!()
    }

    fn asinh(self) -> Self {
        todo!()
    }

    fn acosh(self) -> Self {
        todo!()
    }

    fn atanh(self) -> Self {
        todo!()
    }
}

impl<T> Tangent<T>
where
    T: Real,
{
    fn powf(self, n: T) -> Self {
        Self {
            v: self.v.powf(n.into()),
            dv: self.dv * n * self.v.powf(n - T::one()),
        }
    }
}

#[cfg(test)]
mod tests;
