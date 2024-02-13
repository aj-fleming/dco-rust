use num::{traits::real::Real, Num};
use num::{NumCast, One, ToPrimitive, Zero};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

// use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

#[derive(Clone, Copy)]
// #[repr(C)] // (necessary for re-interpreting the struct as something else.)

/// Definition of forward-mode types for automatic differentiation.
/// Behaves "like a real number" as far as rust is concerned.
struct Tangent<T: Real + ToValue> {
    v: T,
    dv: T,
}

trait ToValue {
    type ValueType;
    /// Get the value of a variable or its derivative.
    fn value(self) -> Self::ValueType;
}

impl ToValue for f64{
    type ValueType = Self;

    fn value(self) -> Self::ValueType {
        self
    }
}

impl<T: Real + ToValue> ToValue for Tangent<T>{
    type ValueType = T;

    fn value(self) -> Self::ValueType {
        self.value
    }
}

impl<T: Real> Tangent<T> {
    /// Take some passive value `arg` and turn it into a Tangent with zero derivative (constant)
    #[inline]
    pub fn make_constant(arg: impl Into<T>) -> Tangent<T> {
        Self {
            v: arg.into(),
            dv: T::zero(),
        }
    }

    /// Take some passive value `arg` and turn it into a Tangent with unity derivative (active)
    #[inline]
    pub fn make_active(arg: impl Into<T>) -> Tangent<T> {
        Self {
            v: arg.into(),
            dv: T::one(),
        }
    }
}

impl<T: Real> Num for Tangent<T> {
    type FromStrRadixErr = T::FromStrRadixErr;

    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(src, radix).map(Tangent::make_constant)
    }
}

impl<T: Real> Real for Tangent<T> {
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
            dv: self.v.mul_add(a.dv, self.dv.mul_add(a.v, b.dv)),
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
                let v = sqrtvalue * T::from(2).unwrap();
                if v == T::zero() && self.dv == T::zero() {
                    T::zero()
                } else {
                    self.dv / v
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

    /// Here we use log rules to write log_a(arg) = ln(arg) / ln(a)
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
        todo!()
    }

    fn to_radians(self) -> Self {
        todo!()
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
            Self::make_constant(T::zero())
        }
    }

    fn cbrt(self) -> Self {
        let cbrtvalue = self.v.cbrt();
        Self {
            v: cbrtvalue,
            dv: {
                let v = cbrtvalue * T::from(3).unwrap();
                if v == T::zero() && self.dv == T::zero() {
                    T::zero()
                } else {
                    T::from(2).unwrap() * self.dv / v
                }
            },
        }
    }

    fn hypot(self, other: Self) -> Self {
        todo!()
    }

    fn sin(self) -> Self {
        Self {
            v: self.v.sin(),
            dv: self.dv * self.v.cos(),
        }
    }

    fn cos(self) -> Self {
        Self {
            v: self.v.cos(),
            dv: -self.dv * self.v.sin(),
        }
    }

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
        todo!()
    }

    fn exp_m1(self) -> Self {
        todo!()
    }

    fn ln_1p(self) -> Self {
        todo!()
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

/*
    REAL NUMBER CONSTANTS FOR TANGENTS
*/

impl<T: Real> Zero for Tangent<T> {
    /// The additive identity for a Tangent<T> has value and derivative zero.
    /// We set the derivative to zero to preserve equality under addition in both fields.
    /// One _could_ set the derivative to unity, since we only define equality on the value field.
    #[inline]
    fn zero() -> Self {
        Self {
            v: T::zero(),
            dv: T::zero(),
        }
    }

    // We only need to check if the value field is zero here.
    #[inline]
    fn is_zero(&self) -> bool {
        T::is_zero(&self.v)
    }
}

impl<T: Real> One for Tangent<T> {
    /// The multiplicative identity for a Tangent<T> has value one and derivative zero.
    /// One _could_ set the derivative to unity, since we only define equality on the value field.
    #[inline]
    fn one() -> Self {
        Self {
            v: T::one(),
            dv: T::one(),
        }
    }

    // We only need to check if the value field is one here.
    #[inline]
    fn is_one(&self) -> bool {
        T::is_one(&self.v)
    }
}

/*
    REAL NUMBER UTILITY OPERATIONS ON TANGENTS
*/

impl<T: Real> PartialEq<Tangent<T>> for Tangent<T> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&self.v, &other.v)
    }
}

impl<T: Real> PartialOrd<Tangent<T>> for Tangent<T> {
    fn partial_cmp(&self, other: &Tangent<T>) -> Option<std::cmp::Ordering> {
        PartialOrd::partial_cmp(&self.v, &other.v)
    }
}

impl<T: Real> NumCast for Tangent<T> {
    fn from<U: num::ToPrimitive>(n: U) -> Option<Self> {
        T::from(n).map(Self::make_constant)
    }
}

impl<T: Real> ToPrimitive for Tangent<T> {
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

/*
    REAL NUMBER ARITHMETIC OPERATIONS ON TANGENTS
*/

impl<T: Real> Add for Tangent<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            v: self.v + rhs.v,
            dv: self.dv + rhs.dv,
        }
    }
}

impl<T: Real> Sub for Tangent<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            v: self.v - rhs.v,
            dv: self.dv - rhs.v,
        }
    }
}

impl<T: Real> Mul for Tangent<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            v: self.v * rhs.v,
            dv: self.dv * rhs.v + self.v * rhs.dv,
        }
    }
}

impl<T: Real> Neg for Tangent<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            v: self.v.neg(),
            dv: self.dv.neg(),
        }
    }
}

impl<T: Real> Div for Tangent<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self {
            v: self.v / rhs.v,
            dv: (rhs.v * self.dv - self.v * rhs.dv) / (rhs.v * rhs.v),
        }
    }
}

impl<T: Real> Rem for Tangent<T> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

/*
REAL NUMBER ASSIGNMENT OPERATIONS ON TANGENTS
*/

#[cfg(test)]
mod tests;
