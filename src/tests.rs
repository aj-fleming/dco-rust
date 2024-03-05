use super::*;
use approx::assert_abs_diff_eq;
use nalgebra as na;
use num::traits::real::Real;
use num::traits::One;

#[test]
fn tangent_initialization() {
    let a: f64 = 1.0;
    let v1 = Tangent::make_active(a);
    let v2 = Tangent::make_constant(a);

    assert_eq!(a, v1.v);
    assert_eq!(f64::one(), v1.dv);
    assert_eq!(a, v2.v);
    assert_eq!(f64::zero(), v2.dv);
}

#[test]
fn polynomials() {
    fn poly2<T: Real>(x: T) -> T {
        x.powi(2) + x
    }

    fn dpoly2<T: Real>(x: T) -> T {
        T::from(2).unwrap() * x + T::one()
    }

    fn poly4(x: Tangent<f64>) -> Tangent<f64> {
        let n: Tangent<f64> = Tangent::make_constant(4.0);
        x.powf(n)
    }

    let arg = 2.0;
    let active_arg = Tangent::make_active(arg);
    assert_eq!(poly2(arg), poly2(active_arg).v);
    assert_eq!(dpoly2(arg), poly2(active_arg).dv);
    assert_eq!(poly4(active_arg).v, active_arg.v.powf(4.0));
}

#[test]
fn trig_identities() {
    fn pythagorean_identity<T: Real>(theta: T) -> T {
        theta.sin() * theta.sin() + theta.cos() * theta.cos()
    }

    let arg: Tangent<f64> = Tangent::make_active(1.0);
    assert_eq!(pythagorean_identity(1.0f64), 1.0);
    assert_eq!(pythagorean_identity(arg).dv, 0.0);

    let v1 = arg.tan();
    let v2 = arg.sin() / arg.cos();
    assert_abs_diff_eq!(v1.v, v2.v);
    assert_abs_diff_eq!(v1.dv, v2.dv);
}

#[test]
fn nalgebra_vectors() {
    let v = na::Vector2::new(1.0, 2.0);
    let arg: Tangent<na::Vector2<f64>> = Tangent::make_constant(v);
    let res = arg + arg;
    assert_eq!(res.v[0], 2.0 * v[0]);
}

#[test]
fn adjoint_initialization() {
    let mut tape: MinimalTape<f64> = MinimalTape::new();
    assert_eq!(tape.num_adjoints(), 0);
    let a1 = tape.new_independent_variable(1.0);
    assert_eq!(a1.v, 1.0);
    assert_eq!(tape.num_adjoints(), 1);
}

#[test]
fn scalar_tape() {}
