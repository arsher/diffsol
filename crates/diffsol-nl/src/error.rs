use diffsol_la::error::{LaError, OperatorError};
use thiserror::Error;

/// Error type for the diffsol non-linear solver crate (`diffsol-nl`).
///
/// This wraps the errors that can occur in the non-linear solver layer, as well as
/// the linear-algebra errors produced by the underlying [`diffsol_la`] linear solvers.
/// It is re-exported by the `diffsol` crate and can be converted into `diffsol`'s
/// top-level error type.
#[derive(Error, Debug, Clone)]
pub enum NlError {
    #[error("Non-linear solver error: {0}")]
    NonLinearSolverError(#[from] NonLinearSolverError),
    #[error("Linear algebra error: {0}")]
    LaError(#[from] LaError),
    #[error("Error: {0}")]
    Other(String),
}

impl From<OperatorError> for NlError {
    fn from(error: OperatorError) -> Self {
        Self::LaError(LaError::OperatorError(error))
    }
}

impl NlError {
    /// Return the user-operator error retained by this error, if any.
    pub fn operator_error(&self) -> Option<&OperatorError> {
        match self {
            Self::LaError(LaError::OperatorError(error)) => Some(error),
            _ => None,
        }
    }
}

/// Possible errors that can occur when solving a non-linear problem
#[derive(Error, Debug, Clone)]
pub enum NonLinearSolverError {
    #[error("Initial condition solver did not converge")]
    InitialConditionDidNotConverge,
    #[error("Newton iterations did not converge, maximum iterations reached")]
    NewtonMaxIterations,
    #[error("Newton iteration diverged")]
    NewtonDiverged,
    #[error("Newton linesearch failed to find a suitable step in max iterations")]
    LinesearchFailedMaxIterations,
    #[error("Newton linesearch failed, minimum step size reached")]
    LinesearchFailedMinStep,
    #[error("LU solve failed")]
    LuSolveFailed,
    #[error("Jacobian not reset before calling solve")]
    JacobianNotReset,
    #[error("State has wrong length: expected {expected}, got {found}")]
    WrongStateLength { expected: usize, found: usize },
    #[error("Error: {0}")]
    Other(String),
}

#[macro_export]
macro_rules! non_linear_solver_error {
    ($variant:ident) => {
        $crate::error::NlError::from($crate::error::NonLinearSolverError::$variant)
    };
    ($variant:ident, $($arg:tt)*) => {
        $crate::error::NlError::from($crate::error::NonLinearSolverError::$variant($($arg)*))
    };
}
