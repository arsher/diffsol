#![cfg(feature = "autodiff")]

use diffsol::{
    BuilderOp, ClosureAutodiff, FaerContext, FaerMat, FaerVec, NonLinearOpAdjoint,
    NonLinearOpJacobian, NonLinearOpSens, NonLinearOpSensAdjoint, ParameterisedOp, Vector,
};

type M = FaerMat<f64>;
type V = FaerVec<f64>;

fn rectangular(x: &V, p: &V, t: f64, y: &mut V) {
    y[0] = p[0] * x[0] * x[0] + 2.0 * x[1] + t;
    y[1] = 3.0 * x[0] + p[1] * x[1] - t;
    y[2] = x[0] * x[1] + p[0];
}

#[test]
fn rectangular_autodiff_closure_sizes_forward_scratch_from_outputs() {
    let ctx = FaerContext::default();
    let mut op = ClosureAutodiff::<M, _>::new(rectangular, 2, 1, 2, ctx);
    BuilderOp::set_nout(&mut op, 3);
    let p = V::from_vec(vec![4.0, 5.0], ctx);
    let pop = ParameterisedOp::new(&op, &p);
    let x = V::from_vec(vec![2.0, 3.0], ctx);

    let state_direction = V::from_vec(vec![7.0, 11.0], ctx);
    let mut state_jvp = V::zeros(3, ctx);
    pop.jac_mul_inplace(&x, 0.5, &state_direction, &mut state_jvp)
        .unwrap();
    state_jvp.assert_eq_st(&V::from_vec(vec![134.0, 76.0, 43.0], ctx), 1e-12);

    let parameter_direction = V::from_vec(vec![7.0, 11.0], ctx);
    let mut parameter_jvp = V::zeros(3, ctx);
    pop.sens_mul_inplace(&x, 0.5, &parameter_direction, &mut parameter_jvp);
    parameter_jvp.assert_eq_st(&V::from_vec(vec![28.0, 33.0, 7.0], ctx), 1e-12);
}

#[test]
fn rectangular_autodiff_closure_sizes_reverse_scratch_from_outputs() {
    let ctx = FaerContext::default();
    let mut op = ClosureAutodiff::<M, _>::new(rectangular, 2, 1, 2, ctx);
    BuilderOp::set_nout(&mut op, 3);
    let p = V::from_vec(vec![4.0, 5.0], ctx);
    let pop = ParameterisedOp::new(&op, &p);
    let x = V::from_vec(vec![2.0, 3.0], ctx);
    let output_cotangent = V::from_vec(vec![7.0, 11.0, 13.0], ctx);

    let mut state_vjp = V::zeros(2, ctx);
    pop.jac_transpose_mul_inplace(&x, 0.5, &output_cotangent, &mut state_vjp);
    state_vjp.assert_eq_st(&V::from_vec(vec![-184.0, -95.0], ctx), 1e-12);

    let mut parameter_vjp = V::zeros(2, ctx);
    pop.sens_transpose_mul_inplace(&x, 0.5, &output_cotangent, &mut parameter_vjp);
    parameter_vjp.assert_eq_st(&V::from_vec(vec![-41.0, -33.0], ctx), 1e-12);
}
