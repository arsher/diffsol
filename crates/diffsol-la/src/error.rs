use std::{error::Error, sync::Arc};

use faer::sparse::CreationError;
use thiserror::Error;

/// Classification applied to an error returned by a user-provided operator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperatorErrorKind {
    /// The current trial point was invalid, but a different point may succeed.
    Recoverable,
    /// Continuing the solve cannot correct the failure.
    Fatal,
}

impl std::fmt::Display for OperatorErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Recoverable => f.write_str("recoverable"),
            Self::Fatal => f.write_str("fatal"),
        }
    }
}

/// A classified error returned by a user-provided operator.
///
/// The original error is retained behind an [`Arc`], so callers can clone this
/// value as solver state requires and still downcast to their concrete error.
#[derive(Clone, Debug, Error)]
#[error("{kind} operator error: {source}")]
pub struct OperatorError {
    kind: OperatorErrorKind,
    #[source]
    source: Arc<dyn Error + 'static>,
}

impl OperatorError {
    /// Construct an error for a trial point that may succeed after retrying.
    pub fn recoverable(error: impl Error + 'static) -> Self {
        Self {
            kind: OperatorErrorKind::Recoverable,
            source: Arc::new(error),
        }
    }

    /// Construct an error that must terminate the solve.
    pub fn fatal(error: impl Error + 'static) -> Self {
        Self {
            kind: OperatorErrorKind::Fatal,
            source: Arc::new(error),
        }
    }

    /// Return the error's retry classification.
    pub fn kind(&self) -> OperatorErrorKind {
        self.kind
    }

    /// Return the original concrete error when its type matches `E`.
    pub fn downcast_ref<E: Error + 'static>(&self) -> Option<&E> {
        self.source.downcast_ref()
    }
}

impl LaError {
    /// Return the user-operator error retained by this error, if any.
    pub fn operator_error(&self) -> Option<&OperatorError> {
        match self {
            Self::OperatorError(error) => Some(error),
            _ => None,
        }
    }
}

/// Result returned by fallible operator evaluations.
pub type OperatorResult<T = ()> = Result<T, OperatorError>;

/// Error type for the diffsol linear algebra crate (`diffsol-la`).
///
/// This wraps the errors that can occur in the linear algebra layer: matrix
/// operations, linear solvers, and (optionally) CUDA. It is re-exported by the
/// `diffsol` crate and can be converted into `diffsol`'s top-level error type.
#[derive(Error, Debug, Clone)]
pub enum LaError {
    #[error(transparent)]
    OperatorError(#[from] OperatorError),
    #[error("Linear solver error: {0}")]
    LinearSolverError(#[from] LinearSolverError),
    #[error("Matrix error: {0}")]
    MatrixError(#[from] MatrixError),
    #[cfg(feature = "cuda")]
    #[error("Cuda error: {0}")]
    CudaError(#[from] CudaError),
    #[error("Error: {0}")]
    Other(String),
}

/// Possible errors that can occur when solving a linear problem
#[derive(Error, Debug, Clone)]
pub enum LinearSolverError {
    #[error("LU not initialized")]
    LuNotInitialized,
    #[error("LU solve failed")]
    LuSolveFailed,
    #[error("Linear solver not setup")]
    LinearSolverNotSetup,
    #[error("Linear solver matrix not square")]
    LinearSolverMatrixNotSquare,
    #[error("Linear solver matrix not compatible with vector")]
    LinearSolverMatrixVectorNotCompatible,
    #[error("KLU failed to analyze")]
    KluFailedToAnalyze,
    #[error("KLU failed to factorize")]
    KluFailedToFactorize,
    #[error("Error: {0}")]
    Other(String),
}

/// Possible errors for matrix operations
#[derive(Error, Debug, Clone)]
pub enum MatrixError {
    #[error("Failed to create matrix from triplets: {0}")]
    FailedToCreateMatrixFromTriplets(#[from] CreationError),
    #[error("Cannot union matrices with different shapes")]
    UnionIncompatibleShapes,
    #[error("Cannot create a matrix with zero rows or columns")]
    MatrixShapeError,
    #[error("Index out of bounds")]
    IndexOutOfBounds,
    #[error("Error: {0}")]
    Other(String),
}

#[cfg(feature = "cuda")]
#[derive(Error, Debug, Clone)]
pub enum CudaError {
    #[error("Failed to allocate memory on GPU")]
    CudaMemoryAllocationError,
    #[error("Failed to initialize CUDA: {0}")]
    CudaInitializationError(String),
    #[error("Cuda error: {0}")]
    Other(String),
}

#[cfg(feature = "cuda")]
#[macro_export]
macro_rules! cuda_error {
    ($variant:ident) => {
        $crate::error::LaError::from($crate::error::CudaError::$variant)
    };
    ($variant:ident, $($arg:tt)*) => {
        $crate::error::LaError::from($crate::error::CudaError::$variant($($arg)*))
    };
}

#[macro_export]
macro_rules! linear_solver_error {
    ($variant:ident) => {
        $crate::error::LaError::from($crate::error::LinearSolverError::$variant)
    };
    ($variant:ident, $($arg:tt)*) => {
        $crate::error::LaError::from($crate::error::LinearSolverError::$variant($($arg)*))
    };
}

#[macro_export]
macro_rules! matrix_error {
    ($variant:ident) => {
        $crate::error::LaError::from($crate::error::MatrixError::$variant)
    };
    ($variant:ident, $($arg:tt)*) => {
        $crate::error::LaError::from($crate::error::MatrixError::$variant($($arg)*))
    };
}

#[macro_export]
macro_rules! la_other_error {
    ($msg:expr) => {
        $crate::error::LaError::Other($msg.to_string())
    };
}
