use std::cell::RefCell;

use crate::{
    find_matrix_non_zeros, jacobian::JacobianColoring, matrix::sparsity::MatrixSparsity, LinearOp,
    Matrix, Op, OperatorResult,
};

use super::{BuilderOp, OpStatistics, ParameterisedOp};

/// A linear closure operator whose matrix-vector callback can fail.
pub struct FallibleLinearClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V) -> OperatorResult,
{
    func: F,
    nstates: usize,
    nout: usize,
    nparams: usize,
    coloring: Option<JacobianColoring<M>>,
    sparsity: Option<M::Sparsity>,
    statistics: RefCell<OpStatistics>,
    ctx: M::C,
}

impl<M, F> FallibleLinearClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V) -> OperatorResult,
{
    /// Construct a fallible linear closure operator.
    pub fn new(func: F, nstates: usize, nout: usize, nparams: usize, ctx: M::C) -> Self {
        Self {
            func,
            nstates,
            statistics: RefCell::new(OpStatistics::default()),
            nout,
            nparams,
            coloring: None,
            sparsity: None,
            ctx,
        }
    }

    /// Discover and cache the matrix sparsity pattern at the supplied time.
    pub fn calculate_sparsity(&mut self, t0: M::T, p: &M::V) -> Result<(), crate::LaError> {
        let op = ParameterisedOp { op: self, p };
        let non_zeros = find_matrix_non_zeros(&op, t0)?;
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

impl<M, F> Op for FallibleLinearClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V) -> OperatorResult,
{
    type V = M::V;
    type T = M::T;
    type M = M;
    type C = M::C;

    fn nstates(&self) -> usize {
        self.nstates
    }

    fn nout(&self) -> usize {
        self.nout
    }

    fn nparams(&self) -> usize {
        self.nparams
    }

    fn context(&self) -> &Self::C {
        &self.ctx
    }

    fn statistics(&self) -> OpStatistics {
        self.statistics.borrow().clone()
    }
}

impl<M, F> BuilderOp for FallibleLinearClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V) -> OperatorResult,
{
    fn calculate_sparsity(
        &mut self,
        _y0: &Self::V,
        t0: Self::T,
        p: &Self::V,
    ) -> Result<(), crate::LaError> {
        self.calculate_sparsity(t0, p)
    }

    fn set_nout(&mut self, nout: usize) {
        self.nout = nout;
    }

    fn set_nparams(&mut self, nparams: usize) {
        self.nparams = nparams;
    }

    fn set_nstates(&mut self, nstates: usize) {
        self.nstates = nstates;
    }
}

impl<M, F> LinearOp for ParameterisedOp<'_, FallibleLinearClosure<M, F>>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V) -> OperatorResult,
{
    fn gemv_inplace(&self, x: &M::V, t: M::T, beta: M::T, y: &mut M::V) -> OperatorResult {
        self.op.statistics.borrow_mut().increment_call();
        (self.op.func)(x, self.p, t, beta, y)
    }

    fn matrix_inplace(&self, t: Self::T, y: &mut Self::M) -> OperatorResult {
        self.op.statistics.borrow_mut().increment_matrix();
        if let Some(coloring) = &self.op.coloring {
            coloring.matrix_inplace(self, t, y)
        } else {
            self._default_matrix_inplace(t, y)
        }
    }

    fn sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.op.sparsity.clone()
    }
}
