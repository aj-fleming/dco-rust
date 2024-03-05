//
// CONSTANTS AND IDENTITIES
//

use std::ops::Add;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;
use num::traits::real::Real;
use num::Num;
use num::ToPrimitive;
use num::NumCast;
use num::One;
use std::ops::{Div, Rem};

use num::Zero;

use crate::Tangent;

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
// NUMCAST
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

//
// REAL
//

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

    // fn powf(self, n: T) -> Self {
    //     Self {
    //         v: self.v.powf(n.into()),
    //         dv: self.dv * n * self.v.powf(n - T::one()),
    //     }
    // }

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
