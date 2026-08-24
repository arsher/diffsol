use crate::{
    scale, LinearOp, Matrix, MatrixSparsity, MatrixSparsityRef, NonLinearOpJacobian,
    OdeEquationsImplicit, OperatorResult, Vector, VectorIndex,
};
use num_traits::{One, Zero};
use std::cell::RefCell;

use super::{NonLinearOp, Op};

/// NonLinearOp implementation of consistent initial conditions for an ODE system.
///
/// We calculate consistent initial conditions following the approach of
/// Brown, P. N., Hindmarsh, A. C., & Petzold, L. R. (1998). Consistent initial condition calculation for differential-algebraic systems. SIAM Journal on Scientific Computing, 19(5), 1495-1512.
pub struct InitOp<'a, Eqn: OdeEquationsImplicit> {
    eqn: &'a Eqn,
    pub y0: RefCell<Eqn::V>,
    pub algebraic_indices: <Eqn::V as Vector>::Index,
    neg_mass: Eqn::M,
    jacobian_sparsity: Option<<Eqn::M as Matrix>::Sparsity>,
}

impl<'a, Eqn: OdeEquationsImplicit> InitOp<'a, Eqn> {
    pub fn new(
        eqn: &'a Eqn,
        t0: Eqn::T,
        y0: &Eqn::V,
        algebraic_indices: <Eqn::V as Vector>::Index,
    ) -> OperatorResult<Self> {
        let n = eqn.rhs().nstates();

        let mass = eqn.mass().unwrap().matrix(t0)?;

        // The initial-condition Jacobian contains differential columns from the
        // mass matrix and algebraic columns from the RHS Jacobian.  Use their
        // structural union without evaluating the RHS at the uncorrected initial
        // algebraic state.
        let jacobian_sparsity = if algebraic_indices.is_empty() {
            mass.sparsity().map(|sparsity| sparsity.to_owned())
        } else {
            match (mass.sparsity(), eqn.rhs().jacobian_sparsity()) {
                (Some(mass_sparsity), Some(rhs_sparsity)) => Some(
                    rhs_sparsity
                        .union(mass_sparsity)
                        .map_err(crate::OperatorError::fatal)?,
                ),
                _ => None,
            }
        };

        // equations are:
        // h(t, u, v, du) = 0
        // g(t, u, v) = 0
        // where u are differential states, v are algebraic states.
        // choose h = -M_u du + f(u, v), where M_u are the differential states of the mass matrix
        // want to solve for du, v, so jacobian is
        // J = (-M_u, df/dv)
        //     (0,    dg/dv)
        let [(m_u, _), _, _, _] = mass.split(&algebraic_indices);
        let m_u = m_u * scale(-Eqn::T::one());
        let zero_ll = <Eqn::M as Matrix>::zeros(
            algebraic_indices.len(),
            n - algebraic_indices.len(),
            eqn.context().clone(),
        );
        let zero_ur = <Eqn::M as Matrix>::zeros(
            n - algebraic_indices.len(),
            algebraic_indices.len(),
            eqn.context().clone(),
        );
        let zero_lr = <Eqn::M as Matrix>::zeros(
            algebraic_indices.len(),
            algebraic_indices.len(),
            eqn.context().clone(),
        );
        let neg_mass = Eqn::M::combine(&m_u, &zero_ur, &zero_ll, &zero_lr, &algebraic_indices);

        let y0 = y0.clone();
        let y0 = RefCell::new(y0);
        Ok(Self {
            eqn,
            y0,
            neg_mass,
            algebraic_indices,
            jacobian_sparsity,
        })
    }

    pub fn scatter_soln(&self, soln: &Eqn::V, y: &mut Eqn::V, dy: &mut Eqn::V) {
        let tmp = dy.clone();
        dy.copy_from(soln);
        dy.copy_from_indices(&tmp, &self.algebraic_indices);
        y.copy_from_indices(soln, &self.algebraic_indices);
    }
}

impl<Eqn: OdeEquationsImplicit> Op for InitOp<'_, Eqn> {
    type V = Eqn::V;
    type T = Eqn::T;
    type M = Eqn::M;
    type C = Eqn::C;
    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
    fn context(&self) -> &Self::C {
        self.eqn.context()
    }
}

impl<Eqn: OdeEquationsImplicit> NonLinearOp for InitOp<'_, Eqn> {
    // -M_u du + f(u, v)
    // g(t, u, v)
    fn call_inplace(&self, x: &Eqn::V, t: Eqn::T, y: &mut Eqn::V) -> OperatorResult {
        // input x = (du, v)
        // self.y0 = (u, v)
        let mut y0 = self.y0.borrow_mut();
        y0.copy_from_indices(x, &self.algebraic_indices);

        // y = (f; g)
        self.eqn.rhs().call_inplace(&y0, t, y)?;

        // y = -M x + y
        self.neg_mass.gemv(Eqn::T::one(), x, Eqn::T::one(), y);
        Ok(())
    }
}

impl<Eqn: OdeEquationsImplicit> NonLinearOpJacobian for InitOp<'_, Eqn> {
    // J v
    fn jac_mul_inplace(&self, x: &Eqn::V, t: Eqn::T, v: &Eqn::V, y: &mut Eqn::V) -> OperatorResult {
        if self.algebraic_indices.is_empty() {
            y.fill(Eqn::T::zero());
        } else {
            // Only algebraic state entries vary in the initial-condition solve.
            // Reconstruct that candidate explicitly so every linear-solver reset
            // observes the current nonlinear iterate rather than the original y0.
            let mut y0 = self.y0.borrow_mut();
            y0.copy_from_indices(x, &self.algebraic_indices);
            let mut algebraic_direction = Eqn::V::zeros(self.nstates(), self.eqn.context().clone());
            algebraic_direction.copy_from_indices(v, &self.algebraic_indices);
            self.eqn
                .rhs()
                .jac_mul_inplace(&y0, t, &algebraic_direction, y)?;
        }

        // neg_mass has zero algebraic columns, so multiplying by the complete
        // direction contributes only the differential-rate part.
        self.neg_mass.gemv(Eqn::T::one(), v, Eqn::T::one(), y);
        Ok(())
    }

    fn jacobian_inplace(&self, x: &Eqn::V, t: Eqn::T, y: &mut Eqn::M) -> OperatorResult {
        let algebraic_indices = self.algebraic_indices.clone_as_vec();
        let mut column = Eqn::V::zeros(self.nout(), self.eqn.context().clone());
        if algebraic_indices.is_empty() {
            for j in 0..self.nstates() {
                column.fill(Eqn::T::zero());
                self.neg_mass.add_column_to_vector(j, &mut column);
                y.set_column(j, &column);
            }
            return Ok(());
        }

        let mut is_algebraic = vec![false; self.nstates()];
        for index in algebraic_indices {
            is_algebraic[index] = true;
        }

        // Evaluate the complete RHS Jacobian once at the current algebraic
        // candidate. Calling the default implementation would evaluate one
        // RHS Jacobian-vector product per state column, which is needlessly
        // expensive for large systems and bypasses any coloring in the RHS.
        let mut y0 = self.y0.borrow_mut();
        y0.copy_from_indices(x, &self.algebraic_indices);
        let rhs_jacobian = self.eqn.rhs().jacobian(&y0, t)?;

        for (j, algebraic) in is_algebraic.into_iter().enumerate() {
            column.fill(Eqn::T::zero());
            if algebraic {
                rhs_jacobian.add_column_to_vector(j, &mut column);
            } else {
                self.neg_mass.add_column_to_vector(j, &mut column);
            }
            y.set_column(j, &column);
        }
        Ok(())
    }

    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.jacobian_sparsity.clone()
    }
}

#[cfg(test)]
mod tests {

    use crate::ode_equations::test_models::exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem;
    use crate::op::init::InitOp;
    use crate::vector::Vector;
    use crate::{
        ConstantOp, DenseMatrix, LinearOp, Matrix, NalgebraMat, NalgebraVec, NonLinearOp,
        NonLinearOpJacobian, OdeBuilder, OdeEquations,
    };

    type Mcpu = NalgebraMat<f64>;
    type Vcpu = NalgebraVec<f64>;

    #[test]
    fn test_initop() {
        let (problem, _soln) = exponential_decay_with_algebraic_problem::<Mcpu>(false);
        let y0 = Vcpu::from_vec(vec![1.0, 2.0, 3.0], *problem.context());
        let dy0 = Vcpu::from_vec(vec![4.0, 5.0, 6.0], *problem.context());
        let t = 0.0;
        let (algebraic_indices, _) = problem
            .eqn()
            .mass()
            .unwrap()
            .matrix(t)
            .unwrap()
            .partition_indices_by_zero_diagonal();

        let initop = InitOp::new(&problem.eqn, t, &y0, algebraic_indices).unwrap();
        // check that the init function is correct
        let mut y_out = Vcpu::from_vec(vec![0.0, 0.0, 0.0], *problem.context());

        // -M_u du + f(u, v)
        // g(t, u, v)
        // M = |1 0 0|
        //     |0 1 0|
        //     |0 0 0|
        //
        // y = |1| (u)
        //     |2| (u)
        //     |3| (v)
        // dy = |4| (du)
        //      |5| (du)
        //      |6| (dv)
        // i.e. f(u, v) = -0.1 u = |-0.1|
        //                         |-0.2|
        //      g(u, v) = v - u = |1|
        //      M_u = |1 0|
        //            |0 1|
        //  i.e. F(y) = |-1 * 4 + -0.1 * 1| = |-4.1|
        //              |-1 * 5 + -0.1 * 2|   |-5.2|
        //              |2 - 1|               |1|
        let du_v = Vcpu::from_vec(vec![dy0[0], dy0[1], y0[2]], *problem.context());
        initop.call_inplace(&du_v, t, &mut y_out).unwrap();
        let y_out_expect = Vcpu::from_vec(vec![-4.1, -5.2, 1.0], *problem.context());
        y_out.assert_eq_st(&y_out_expect, 1e-10);

        // df/dv = |0|
        //         |0|
        // dg/dv = |1|
        // J = (-M_u, df/dv) = |-1 0 0|
        //                   = |0 -1 0|
        //     (0,    dg/dv) = |0 0 1|
        let jac = initop.jacobian(&du_v, t).unwrap();
        assert_eq!(jac.get_index(0, 0), -1.0);
        assert_eq!(jac.get_index(0, 1), 0.0);
        assert_eq!(jac.get_index(0, 2), 0.0);
        assert_eq!(jac.get_index(1, 0), 0.0);
        assert_eq!(jac.get_index(1, 1), -1.0);
        assert_eq!(jac.get_index(1, 2), 0.0);
        assert_eq!(jac.get_index(2, 0), 0.0);
        assert_eq!(jac.get_index(2, 1), 0.0);
        assert_eq!(jac.get_index(2, 2), 1.0);
    }

    #[test]
    fn initop_refreshes_algebraic_jacobian_at_candidate() {
        let problem = OdeBuilder::<Mcpu>::new()
            .rhs_implicit(
                |x, _p, _t, y| {
                    y[0] = -x[0];
                    y[1] = x[1] * x[1] - x[0];
                },
                |x, _p, _t, v, y| {
                    y[0] = -v[0];
                    y[1] = 2.0 * x[1] * v[1] - v[0];
                },
            )
            .mass(|v, _p, _t, beta, y| {
                let previous = y.clone();
                y[0] = v[0] + beta * previous[0];
                y[1] = beta * previous[1];
            })
            .init(
                |_p, _t, y| {
                    y[0] = 1.0;
                    y[1] = 1.0;
                },
                2,
            )
            .build()
            .unwrap();
        let y0 = problem.eqn.init().call(problem.t0);
        let mass = problem.eqn.mass().unwrap().matrix(problem.t0).unwrap();
        let (algebraic_indices, _) = mass.partition_indices_by_zero_diagonal();
        let initop = InitOp::new(&problem.eqn, problem.t0, &y0, algebraic_indices).unwrap();

        let first = Vcpu::from_vec(vec![0.0, 1.0], *problem.context());
        let second = Vcpu::from_vec(vec![0.0, 3.0], *problem.context());
        let first_jacobian = initop.jacobian(&first, problem.t0).unwrap();
        let second_jacobian = initop.jacobian(&second, problem.t0).unwrap();

        assert_eq!(first_jacobian.get_index(1, 1), 2.0);
        assert_eq!(second_jacobian.get_index(1, 1), 6.0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_initop_batched() {
        use crate::{
            ode_equations::test_models::exponential_decay_with_algebraic::{
                exponential_decay_with_algebraic_batched,
                exponential_decay_with_algebraic_init_batched,
                exponential_decay_with_algebraic_jacobian_batched,
                exponential_decay_with_algebraic_mass_batched,
            },
            CudaContext, CudaMat, CudaVec, OdeBuilder,
        };

        let nbatch = 2;
        let ctx = CudaContext::default().with_nbatch(nbatch);
        let p_f64 = vec![0.1, 0.2];
        let problem = OdeBuilder::<CudaMat<f64>>::new()
            .context(ctx.clone())
            .p(p_f64)
            .rhs_implicit(
                exponential_decay_with_algebraic_batched::<CudaMat<f64>>,
                exponential_decay_with_algebraic_jacobian_batched::<CudaMat<f64>>,
            )
            .mass(exponential_decay_with_algebraic_mass_batched::<CudaMat<f64>>)
            .init(
                exponential_decay_with_algebraic_init_batched::<CudaMat<f64>>,
                3,
            )
            .build()
            .unwrap();

        let y0 = CudaVec::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], ctx.clone());
        let t = 0.0;
        let (algebraic_indices, _) = problem
            .eqn()
            .mass()
            .unwrap()
            .matrix(t)
            .unwrap()
            .partition_indices_by_zero_diagonal();

        let initop = InitOp::new(&problem.eqn, t, &y0, algebraic_indices).unwrap();

        let du_v = CudaVec::from_vec(vec![4.0, 5.0, 1.0, 4.0, 5.0, 1.0], ctx.clone());
        let mut y_out = CudaVec::zeros(3, ctx.clone());
        initop.call_inplace(&du_v, t, &mut y_out).unwrap();
        let expect = CudaVec::from_vec(vec![-4.1, -5.1, 0.0, -4.2, -5.2, 0.0], ctx.clone());
        y_out.assert_eq_st(&expect, 1e-10);

        let x0 = CudaVec::from_vec(vec![-0.1, -0.1, 1.0, -0.2, -0.2, 1.0], ctx.clone());
        let mut zero_out = CudaVec::zeros(3, ctx);
        initop.call_inplace(&x0, t, &mut zero_out).unwrap();
        let expect_zero = CudaVec::from_vec(vec![0.0; 6], zero_out.context().clone());
        zero_out.assert_eq_st(&expect_zero, 1e-10);
    }
}
