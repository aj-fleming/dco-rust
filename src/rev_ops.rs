use num::traits::real::Real;
use num::traits::Inv;
use num::{Num, NumCast, One, ToPrimitive, Zero};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use crate::{Adjoint, AdjointTape};

/// Creates a *NEW* adjoint variable with value `value` on tape `tape`.
fn create_constant<'a, T, TAPE>(value: T, tape: &'a TAPE) -> Adjoint<'a, T, TAPE>
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
    TAPE: AdjointTape<T>,
{
    let mut new_on_tape: Adjoint<'a, T, TAPE> = Adjoint {
        v: value,
        id: 0,
        tape: Some(tape),
    };
    // Creating a new constant has no pullback.
    // This lets us get the correct id.
    tape.record_result(&mut new_on_tape, 0);
    new_on_tape
}

impl<'a, T, TAPE> Zero for Adjoint<'a, T, TAPE>
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T> + One,
    TAPE: AdjointTape<T>,
{
    fn zero() -> Self {
        Self {
            v: T::zero(),
            id: 0,
            tape: None,
        }
    }

    fn is_zero(&self) -> bool {
        self.v.is_zero()
    }
}

impl<'a, T, TAPE> One for Adjoint<'a, T, TAPE>
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T> + One,
    TAPE: AdjointTape<T>,
{
    fn one() -> Self {
        Self {
            v: T::one(),
            id: 0,
            tape: None,
        }
    }
}

impl<'a, T, TAPE> Add for Adjoint<'a, T, TAPE>
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T> + One,
    TAPE: AdjointTape<T>,
{
    type Output = Self;

    fn add(self, rhs_maybe_identity: Self) -> Self::Output {
        // we first need to check if the adjoint is taped.
        // (identity functions from One and Zero don't know about the tape)
        let rhs: Adjoint<'_, T, TAPE> = match rhs_maybe_identity.tape {
            Some(_) => rhs_maybe_identity,
            None => create_constant(rhs_maybe_identity.v, self.tape.unwrap()),
        };
        let mut res: Adjoint<'_, T, TAPE> = Self {
            v: self.v + rhs.v,
            id: 0,
            tape: self.tape,
        };
        let tape: &TAPE = self.tape.unwrap();
        tape.record_derivative(&self, T::one());
        tape.record_derivative(&rhs, T::one());
        tape.record_result(&mut res, 2);
        res
    }
}

impl<'a, T, TAPE> Sub for Adjoint<'a, T, TAPE>
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T> + One + Neg<Output = T> + Sub<Output = T>,
    TAPE: AdjointTape<T>,
{
    type Output = Self;

    fn sub(self, rhs_maybe_identity: Self) -> Self::Output {
        // we first need to check if the adjoint is taped.
        // (identity functions from One and Zero don't know about the tape)
        let rhs: Adjoint<'_, T, TAPE> = match rhs_maybe_identity.tape {
            Some(_) => rhs_maybe_identity,
            None => create_constant(rhs_maybe_identity.v, self.tape.unwrap()),
        };
        let mut res: Adjoint<'_, T, TAPE> = Self {
            v: self.v - rhs.v,
            id: 0,
            tape: self.tape,
        };
        let tape: &TAPE = self.tape.unwrap();
        tape.record_derivative(&self, T::one());
        tape.record_derivative(&rhs, -T::one());
        tape.record_result(&mut res, 2);
        res
    }
}

impl<'a, T, TAPE> Mul for Adjoint<'a, T, TAPE>
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
    TAPE: AdjointTape<T>,
{
    type Output = Self;

    fn mul(self, rhs_maybe_identity: Self) -> Self::Output {
        let rhs: Adjoint<'_, T, TAPE> = match rhs_maybe_identity.tape {
            Some(_) => rhs_maybe_identity,
            None => create_constant(rhs_maybe_identity.v, self.tape.unwrap()),
        };
        let mut res: Adjoint<'_, T, TAPE> = Self {
            v: self.v + rhs.v,
            id: 0,
            tape: self.tape,
        };
        let tape: &TAPE = self.tape.unwrap();
        tape.record_derivative(&self, rhs.v);
        tape.record_derivative(&rhs, self.v);
        tape.record_result(&mut res, 2);
        res
    }
}

impl<'a, T, TAPE> Div for Adjoint<'a, T, TAPE>
where
    T: Copy
        + Zero
        + Add<Output = T>
        + Mul<Output = T>
        + One
        + Neg<Output = T>
        + Sub<Output = T>
        + Div<Output = T>,
    TAPE: AdjointTape<T>,
{
    type Output = Self;

    fn div(self, rhs_maybe_identity: Self) -> Self::Output {
        // we first need to check if the adjoint is taped.
        // (identity functions from One and Zero don't know about the tape)
        let rhs: Adjoint<'_, T, TAPE> = match rhs_maybe_identity.tape {
            Some(_) => rhs_maybe_identity,
            None => create_constant(rhs_maybe_identity.v, self.tape.unwrap()),
        };
        let mut res: Adjoint<'_, T, TAPE> = Self {
            v: self.v / rhs.v,
            id: 0,
            tape: self.tape,
        };
        let tape: &TAPE = self.tape.unwrap();
        tape.record_derivative(&self, T::one() / rhs.v);
        tape.record_derivative(&rhs, -self.v / (rhs.v * rhs.v));
        tape.record_result(&mut res, 2);
        res
    }
}
