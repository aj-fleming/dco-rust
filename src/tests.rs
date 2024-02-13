use super::*;

mod forward_mode {
    use super::*;
    #[test]
    fn insantiation() {
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
            T::from(2).unwrap() * x + T::from(1).unwrap()
        }

        let arg  = 2.0;
        let active_arg = Tangent::make_active(arg);
        assert_eq!(poly2(arg), poly2(active_arg).v);
        assert_eq!(dpoly2(arg), poly2(active_arg).dv);
    }
}
