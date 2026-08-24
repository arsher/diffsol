use crate::{MyRhs, T, V};
use diffsol::{NonLinearOp, NonLinearOpJacobian, OperatorResult};

impl NonLinearOp for MyRhs<'_> {
    fn call_inplace(&self, x: &V, _t: T, y: &mut V) -> OperatorResult {
        y[0] = x[0] * x[0];
        Ok(())
    }
}

impl NonLinearOpJacobian for MyRhs<'_> {
    fn jac_mul_inplace(&self, x: &V, _t: T, v: &V, y: &mut V) -> OperatorResult {
        y[0] = v[0] * (1.0 - 2.0 * x[0]);
        Ok(())
    }
}
