use std::cell::RefCell;

use crate::{
    find_jacobian_non_zeros, jacobian::JacobianColoring, Matrix, MatrixSparsity, NonLinearOp,
    NonLinearOpJacobian, Op, OperatorResult,
};

use super::{BuilderOp, OpStatistics, ParameterisedOp};

/// A nonlinear closure operator whose value and Jacobian-vector callbacks can fail.
///
/// Callback failures retain their [`crate::OperatorError`] classification and
/// source as they propagate through nonlinear and ODE solvers.
pub struct FallibleClosure<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V) -> OperatorResult,
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V) -> OperatorResult,
{
    func: F,
    jacobian_action: G,
    nstates: usize,
    nout: usize,
    nparams: usize,
    coloring: Option<JacobianColoring<M>>,
    sparsity: Option<M::Sparsity>,
    statistics: RefCell<OpStatistics>,
    ctx: M::C,
}

impl<M, F, G> FallibleClosure<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V) -> OperatorResult,
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V) -> OperatorResult,
{
    /// Construct a fallible closure operator.
    pub fn new(
        func: F,
        jacobian_action: G,
        nstates: usize,
        nout: usize,
        nparams: usize,
        ctx: M::C,
    ) -> Self {
        Self {
            func,
            jacobian_action,
            nstates,
            nparams,
            nout,
            statistics: RefCell::new(OpStatistics::default()),
            coloring: None,
            sparsity: None,
            ctx,
        }
    }

    /// Discover and cache the Jacobian sparsity pattern at the supplied point.
    pub fn calculate_sparsity(
        &mut self,
        y0: &M::V,
        t0: M::T,
        p: &M::V,
    ) -> Result<(), crate::LaError> {
        let param_op = ParameterisedOp { op: self, p };
        let non_zeros = find_jacobian_non_zeros(&param_op, y0, t0)?;
        self.sparsity = Some(MatrixSparsity::try_from_indices(
            self.nout(),
            self.nstates(),
            non_zeros.clone(),
        )?);
        self.coloring = Some(JacobianColoring::new(
            self.sparsity.as_ref().unwrap(),
            &non_zeros,
            self.ctx.clone(),
        ));
        Ok(())
    }
}

impl<M, F, G> BuilderOp for FallibleClosure<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V) -> OperatorResult,
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V) -> OperatorResult,
{
    fn calculate_sparsity(&mut self, y0: &M::V, t0: M::T, p: &M::V) -> Result<(), crate::LaError> {
        self.calculate_sparsity(y0, t0, p)
    }

    fn set_nstates(&mut self, nstates: usize) {
        self.nstates = nstates;
    }

    fn set_nout(&mut self, nout: usize) {
        self.nout = nout;
    }

    fn set_nparams(&mut self, nparams: usize) {
        self.nparams = nparams;
    }
}

impl<M, F, G> Op for FallibleClosure<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V) -> OperatorResult,
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V) -> OperatorResult,
{
    type V = M::V;
    type T = M::T;
    type M = M;
    type C = M::C;

    fn context(&self) -> &Self::C {
        &self.ctx
    }

    fn nstates(&self) -> usize {
        self.nstates
    }

    fn nout(&self) -> usize {
        self.nout
    }

    fn nparams(&self) -> usize {
        self.nparams
    }

    fn statistics(&self) -> OpStatistics {
        self.statistics.borrow().clone()
    }
}

impl<M, F, G> NonLinearOp for ParameterisedOp<'_, FallibleClosure<M, F, G>>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V) -> OperatorResult,
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V) -> OperatorResult,
{
    fn call_inplace(&self, x: &M::V, t: M::T, y: &mut M::V) -> OperatorResult {
        self.op.statistics.borrow_mut().increment_call();
        (self.op.func)(x, self.p, t, y)
    }
}

impl<M, F, G> NonLinearOpJacobian for ParameterisedOp<'_, FallibleClosure<M, F, G>>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V) -> OperatorResult,
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V) -> OperatorResult,
{
    fn jac_mul_inplace(&self, x: &M::V, t: M::T, v: &M::V, y: &mut M::V) -> OperatorResult {
        self.op.statistics.borrow_mut().increment_jac_mul();
        (self.op.jacobian_action)(x, self.p, t, v, y)
    }

    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) -> OperatorResult {
        self.op.statistics.borrow_mut().increment_matrix();
        if let Some(coloring) = self.op.coloring.as_ref() {
            coloring.jacobian_inplace(self, x, t, y)
        } else {
            self._default_jacobian_inplace(x, t, y)
        }
    }

    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.op.sparsity.clone()
    }
}
