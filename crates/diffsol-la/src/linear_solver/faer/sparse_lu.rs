use crate::context::broadcast_batch;
use crate::{
    error::LaError, linear_solver::LinearSolver, linear_solver_error, scalar::IndexType, Context,
    FaerContext, FaerScalar, FaerSparseMat, FaerVec, LinearOp, Matrix,
};

use faer::{
    linalg::solvers::Solve,
    reborrow::{Reborrow, ReborrowMut},
    sparse::linalg::{solvers::Lu, solvers::SymbolicLu},
};

/// A [LinearSolver] that uses the LU decomposition in the [`faer`](https://github.com/sarah-ek/faer-rs) library to solve the linear system.
pub struct FaerSparseLU<T>
where
    T: FaerScalar,
{
    lu: Vec<Lu<IndexType, T>>,
    lu_symbolic: Option<SymbolicLu<IndexType>>,
    matrix: Option<FaerSparseMat<T>>,
}

impl<T> Default for FaerSparseLU<T>
where
    T: FaerScalar,
{
    fn default() -> Self {
        Self {
            lu: Vec::new(),
            matrix: None,
            lu_symbolic: None,
        }
    }
}

impl<T: FaerScalar> LinearSolver<FaerSparseMat<T>> for FaerSparseLU<T> {
    fn set_linearisation<
        C: LinearOp<T = T, V = FaerVec<T>, M = FaerSparseMat<T>, C = FaerContext>,
    >(
        &mut self,
        op: &C,
    ) -> Result<(), LaError> {
        self.lu.clear();
        let matrix = self
            .matrix
            .as_mut()
            .ok_or_else(|| linear_solver_error!(LinearSolverNotSetup))?;
        op.matrix_inplace(matrix)?;
        let symbolic = self
            .lu_symbolic
            .as_ref()
            .ok_or_else(|| linear_solver_error!(LinearSolverNotSetup))?;
        self.lu = matrix
            .data
            .iter()
            .map(|matrix| {
                Lu::try_new_with_symbolic(symbolic.clone(), matrix.rb()).map_err(|error| {
                    linear_solver_error!(
                        Other,
                        format!("Faer sparse numeric factorization failed: {error}")
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(())
    }

    fn solve_in_place(&self, x: &mut FaerVec<T>) -> Result<(), LaError> {
        if self.lu.is_empty() {
            return Err(linear_solver_error!(LuNotInitialized));
        }
        x.context
            .assert_broadcastable_into(self.lu.len(), "sparse_lu_solve");
        let nlu = self.lu.len();
        let nb = x.data.ncols();
        for batch in 0..nb {
            self.lu[broadcast_batch(batch, nlu, nb)].solve_in_place(x.data.rb_mut().col_mut(batch));
        }
        Ok(())
    }

    fn set_sparsity<C: LinearOp<T = T, V = FaerVec<T>, M = FaerSparseMat<T>, C = FaerContext>>(
        &mut self,
        op: &C,
    ) -> Result<(), LaError> {
        self.lu.clear();
        self.lu_symbolic = None;
        self.matrix = None;
        let ncols = op.ncols();
        let nrows = op.nrows();
        let matrix = C::M::new_from_sparsity(nrows, ncols, op.sparsity(), *op.context());
        self.lu_symbolic = Some(
            SymbolicLu::try_new(matrix.data[0].symbolic()).map_err(|error| {
                linear_solver_error!(
                    Other,
                    format!("Faer sparse symbolic analysis failed: {error}")
                )
            })?,
        );
        self.matrix = Some(matrix);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        linear_solver::tests::{diagonal_op, test_grouped_lu_solve, test_narrow_state_lu_solve},
        MatrixCommon, Vector,
    };

    struct StructurallySingularOp {
        matrix: FaerSparseMat<f64>,
    }

    impl LinearOp for StructurallySingularOp {
        type T = f64;
        type V = FaerVec<f64>;
        type M = FaerSparseMat<f64>;
        type C = FaerContext;

        fn nrows(&self) -> IndexType {
            self.matrix.nrows()
        }

        fn ncols(&self) -> IndexType {
            self.matrix.ncols()
        }

        fn context(&self) -> &Self::C {
            self.matrix.context()
        }

        fn matrix_inplace(&self, matrix: &mut Self::M) -> crate::OperatorResult {
            matrix.copy_from(&self.matrix);
            Ok(())
        }

        fn sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
            self.matrix
                .sparsity()
                .map(|sparsity| sparsity.to_owned().unwrap())
        }
    }

    #[test]
    fn test_sparse_lu() {
        let mut s = FaerSparseLU::<f64>::default();
        let op = diagonal_op::<FaerSparseMat<f64>>(2.0);
        s.set_sparsity(&op).unwrap();
        s.set_linearisation(&op).unwrap();
        let b = FaerVec::from_vec(vec![2.0, 4.0], Default::default());
        let x = s.solve(&b).unwrap();
        x.assert_eq_st(
            &FaerVec::from_vec(vec![1.0, 2.0], Default::default()),
            1e-10,
        );
    }

    #[test]
    fn test_grouped_sparse_lu() {
        test_grouped_lu_solve::<FaerSparseMat<f64>, FaerSparseLU<f64>>(FaerContext::with_nbatch(2));
    }

    #[test]
    #[should_panic(expected = "incompatible nbatch")]
    fn test_narrow_state_sparse_lu() {
        test_narrow_state_lu_solve::<FaerSparseMat<f64>, FaerSparseLU<f64>>(
            FaerContext::with_nbatch(2),
        );
    }

    #[test]
    fn sparse_lu_reports_linearisation_before_setup() {
        let mut s = FaerSparseLU::<f64>::default();
        let op = diagonal_op::<FaerSparseMat<f64>>(2.0);

        let error = s.set_linearisation(&op).unwrap_err();

        assert!(matches!(
            error,
            LaError::LinearSolverError(crate::error::LinearSolverError::LinearSolverNotSetup)
        ));
    }

    #[test]
    fn sparse_lu_reports_singular_numeric_factorization() {
        let mut s = FaerSparseLU::<f64>::default();
        let op = StructurallySingularOp {
            matrix: FaerSparseMat::try_from_triplets(
                2,
                2,
                vec![(0, 0)],
                vec![1.0],
                FaerContext::default(),
            )
            .unwrap(),
        };
        s.set_sparsity(&op).unwrap();

        let error = s.set_linearisation(&op).unwrap_err();

        let LaError::LinearSolverError(crate::error::LinearSolverError::Other(message)) = error
        else {
            panic!("unexpected error: {error}");
        };
        assert!(message.contains("Faer sparse numeric factorization failed"));
    }
}
