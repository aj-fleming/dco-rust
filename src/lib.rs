use num::traits::{real::Real, MulAdd};
use num::{Float, Num, NumCast, One, ToPrimitive, Zero};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};


// use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

#[derive(Clone, Copy)]
#[repr(C)] // (necessary for re-interpreting the struct as something else.)
/// Definition of forward-mode types for automatic differentiation.
///
///
/// The "innermost" `T` should implement `num::Real` or `num::Float`.
pub struct ForwardDiffDual<T, U> {
    v: T,
    dv: U,
}
pub type Tangent<T> = ForwardDiffDual<T, T>;

pub struct ReverseDiffDual<'tape, T, U, TAPE> {
    v: T,
    vbar: U,
    tape: &'tape mut TAPE,
}

pub struct Tape<T> {
    tape: Vec<T>,
    n_in: i32,
    n_out: i32,
    position: i32,
}

impl<T> Tape<T> {
    fn interpret() {}

    fn push_derivative(&mut self, dv: T) {
        self.tape.push(dv);
    }
}

pub type Adjoint<'tape, T> = ReverseDiffDual<'tape, T, T, Tape<T>>;

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

impl<T: Float> PassiveValue for T {
    type PassiveValueType = Self;

    #[inline]
    fn passive_value(self) -> Self::PassiveValueType {
        self
    }
}

impl<T: PassiveValue, U> PassiveValue for ForwardDiffDual<T, U> {
    type PassiveValueType = T::PassiveValueType;

    #[inline]
    fn passive_value(self) -> Self::PassiveValueType {
        self.v.passive_value()
    }
}

trait DualNumberEq {
    fn dual_eq(&self, other: &Self) -> bool;
}

impl<T: PartialEq, U: PartialEq> DualNumberEq for ForwardDiffDual<T, U> {
    #[inline]
    fn dual_eq(&self, other: &Self) -> bool {
        self.v == other.v && self.dv == other.dv
    }
}

impl<'a, T: PartialEq, U: PartialEq, TAPE> DualNumberEq for ReverseDiffDual<'a, T, U, TAPE> {
    #[inline]
    fn dual_eq(&self, other: &Self) -> bool {
        self.v == other.v && self.vbar == other.vbar
    }
}

impl<T, U: Zero> ForwardDiffDual<T, U> {
    /// Take some passive value `arg` and turn it into a Tangent with zero derivative (constant)
    #[inline]
    pub fn make_constant(arg: impl Into<T>) -> ForwardDiffDual<T, U> {
        Self {
            v: arg.into(),
            dv: U::zero(),
        }
    }
}

impl<T, U: One> ForwardDiffDual<T, U> {
    /// Take some passive value `arg` and turn it into a Tangent with unity derivative (active)
    #[inline]
    pub fn make_active(arg: impl Into<T>) -> ForwardDiffDual<T, U> {
        Self {
            v: arg.into(),
            dv: U::one(),
        }
    }
}

//
// EQUALITY AND ORDERING
//

impl<T: PartialEq, U> PartialEq<ForwardDiffDual<T, U>> for ForwardDiffDual<T, U> {
    #[inline]
    /// We discard the derivative information when checking equality.
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&self.v, &other.v)
    }
}

impl<T: PartialOrd, U> PartialOrd<ForwardDiffDual<T, U>> for ForwardDiffDual<T, U> {
    #[inline]
    // We discard the derivative information when comparing.
    fn partial_cmp(&self, other: &ForwardDiffDual<T, U>) -> Option<std::cmp::Ordering> {
        PartialOrd::partial_cmp(&self.v, &other.v)
    }
}

//
// ZERO AND ONE
//

impl<T: Zero, U: Zero> Zero for ForwardDiffDual<T, U> {
    #[inline]
    fn zero() -> Self {
        Self {
            v: T::zero(),
            dv: U::zero(),
        }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.v.is_zero()
    }
}

impl<T: Copy + One + PartialEq, U: Zero + Mul<T, Output = U>> One for ForwardDiffDual<T, U> {
    #[inline]
    fn one() -> Self {
        Self {
            v: T::one(),
            dv: U::zero(),
        }
    }

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

impl<T, U> Add for ForwardDiffDual<T, U>
where
    T: Add<T, Output = T>,
    U: Add<U, Output = U>,
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

impl<T, U> Sub for ForwardDiffDual<T, U>
where
    T: Sub<T, Output = T>,
    U: Sub<U, Output = U>,
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

impl<T: Copy, U> Mul for ForwardDiffDual<T, U>
where
    T: Mul<T, Output = T>,
    U: Mul<T, Output = U> + Add<U, Output = U>,
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

impl<T, U> Neg for ForwardDiffDual<T, U>
where
    T: Neg<Output = T>,
    U: Neg<Output = U>,
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

impl<T: Copy, U> Div for ForwardDiffDual<T, U>
where
    T: Div<T, Output = T> + Mul<T, Output = T>,
    U: Mul<T, Output = U> + Sub<U, Output = U> + Div<U, Output = U> + Div<T, Output = U>,
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

impl<T, U> Rem for ForwardDiffDual<T, U> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl<T, U> Num for ForwardDiffDual<T, U>
where
    T: Copy + Num,
    U: Zero
        + Add<U, Output = U>
        + Sub<U, Output = U>
        + Mul<T, Output = U>
        + Div<U, Output = U>
        + Div<T, Output = U>,
{
    type FromStrRadixErr = T::FromStrRadixErr;

    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(src, radix).map(ForwardDiffDual::make_constant)
    }
}

//
// NUMCAST AND REAL
//

impl<T: NumCast, U: Zero> NumCast for ForwardDiffDual<T, U> {
    fn from<V: num::ToPrimitive>(n: V) -> Option<Self> {
        T::from(n).map(Self::make_constant)
    }
}

impl<T: ToPrimitive, U> ToPrimitive for ForwardDiffDual<T, U> {
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

impl<T, U> Real for ForwardDiffDual<T, U>
where
    T: Real,
    U: Copy + Clone + Mul<T, Output = U> + Div<T, Output = U> + Real + MulAdd<T, U, Output = U>,
{
    /// Get the minimum value of a Tangent.
    /// Does {min, min} make sense here?
    #[inline]
    fn min_value() -> Self {
        Self {
            v: T::min_value(),
            dv: U::min_value(),
        }
    }

    #[inline]
    fn min_positive_value() -> Self {
        Self {
            v: T::min_positive_value(),
            dv: U::min_positive_value(),
        }
    }

    #[inline]
    fn epsilon() -> Self {
        Self {
            v: T::epsilon(),
            dv: U::epsilon(),
        }
    }

    #[inline]
    fn max_value() -> Self {
        Self {
            v: T::max_value(),
            dv: U::max_value(),
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
        ForwardDiffDual::make_constant(self.v.signum())
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
            dv: MulAdd::mul_add(a.dv, self.v, MulAdd::mul_add(self.dv, a.v, b.dv)),
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
                if denom == T::zero() && self.dv == U::zero() {
                    U::zero()
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
                if denom == T::zero() && self.dv == U::zero() {
                    U::zero()
                } else {
                    U::from(2).unwrap() * self.dv / denom
                }
            },
        }
    }

    fn hypot(self, other: Self) -> Self {
        let hypotvalue = self.v.hypot(other.v);
        Self {
            v: hypotvalue,
            dv: (U::from(2).unwrap())
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

impl<T, U> ForwardDiffDual<T, U>
where
    T: Real,
    U: Mul<T, Output = U>
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
