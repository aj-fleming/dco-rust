use num::Float;
use std::ops::{Add, Div, Mul, Sub};

// use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

#[derive(Clone, Copy)]
struct Tangent<T: Float> {
    value: T,
    derivative: T,
}

impl<T: Float> Tangent<T> {
    fn sin(&self) -> Self {
        return Self {
            value: self.value.sin(),
            derivative: self.derivative * self.value.cos(),
        };
    }

    fn cos(&self) -> Self {
        return Self {
            value: self.value.cos(),
            derivative: -self.derivative * self.value.sin(),
        };
    }

    fn tan(&self) -> Self {
        return Self {
            value: self.value.tan(),
            derivative: self.derivative / (self.value.cos() * self.value.cos()),
        };
    }

    fn powf(&self, exponent: T) -> Self {
        return Self {
            value: self.value.powf(exponent),
            derivative: self.derivative * exponent * self.value.powf(exponent - T::one()),
        };
    }

    fn exp(&self) -> Self {
        return Self {
            value: self.value.exp(),
            derivative: self.derivative * self.value.exp(),
        };
    }
}

// If we know we can convert t to i32, we might as well implement this. 
// Better ask an expert first.
// impl<T: Float + std::convert::From<i32>> Tangent<T> {
//     fn powi(&self, n: i32) -> Self {
//         return Self {
//             value: self.value.powi(n),
//             derivative: self.derivative * (n.into()) * self.value.powi(n - 1),
//         };
//     }
// }

impl<T: Float> Add for Tangent<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        return Self {
            value: self.value + rhs.value,
            derivative: self.derivative + rhs.derivative,
        };
    }
}

// impl<T: Float> AddAssign for Tangent<T> {
//     fn add_assign(&mut self, rhs: Self) {
//         self.value += rhs.value;
//         self.derivative += rhs.derivative;
//     }
// }

impl<T: Float> Mul for Tangent<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        return Self {
            value: self.value * rhs.value,
            derivative: self.derivative * rhs.value + self.value * rhs.derivative,
        };
    }
}

// impl<T: DifferentiableScalarAssign> MulAssign for Tangent<T> {
//     fn mul_assign(&mut self, rhs: Self) {
//         self.value *= rhs.value;
//         self.derivative *= rhs.value;
//         self.derivative += rhs.derivative * self.value;
//     }
// }

impl<T: Float> Sub for Tangent<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        return Self {
            value: self.value - rhs.value,
            derivative: self.derivative - rhs.value,
        };
    }
}

impl<T: Float> Div for Tangent<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        return Self {
            value: self.value / rhs.value,
            derivative: (rhs.value * self.derivative - self.value * rhs.derivative)
                / (rhs.value * rhs.value),
        };
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, PI};

    use super::*;

    #[test]
    fn test_tangent_trig_fns() {
        let arg1 = Tangent {
            value: PI,
            derivative: 1.0,
        };

        let arg2 = Tangent {
            value: FRAC_PI_2,
            derivative: 1.0,
        };
        
        assert!(arg1.sin().derivative - -1.0 < f64::EPSILON);
        assert!(arg2.sin().derivative < f64::EPSILON);
        
        assert!(arg1.cos().derivative < f64::EPSILON);
        assert!(arg2.cos().derivative - -1.0 < f64::EPSILON);
    }
}
