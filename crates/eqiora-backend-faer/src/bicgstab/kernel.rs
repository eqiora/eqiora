//! Single-RHS specialization of faer 0.24.4's BiCGSTAB recurrence.
//!
//! Adapted from faer/src/operator/bicgstab.rs (MIT; see LICENSE-FAER).
//! Unlike that release, every complex contraction is a Hermitian product.
//! Faer owns typed vectors, products, norms and operator/preconditioner actions;
//! the surrounding adapter owns restarts and independent candidate acceptance.
use super::*;
use faer::mat::{MatMut, MatRef};
use faer::matrix_free::LinOp;
use faer::reborrow::Reborrow;

pub(super) struct CycleInfo {
    pub(super) iter_count: usize,
    pub(super) breakdown: bool,
}

pub(super) struct Kernel<'a, S> {
    operator: FaerOperator<'a, S>,
    left: FaerPreconditioner<S>,
    right: FaerPreconditioner<S>,
    workspace: MemBuffer,
    r: Mat<S>,
    p: Mat<S>,
    shadow: Mat<S>,
    y: Mat<S>,
    v: Mat<S>,
    s: Mat<S>,
    z: Mat<S>,
    t: Mat<S>,
    temporary: Mat<S>,
}

impl<'a, S: KrylovScalar> Kernel<'a, S> {
    pub(super) fn new(
        operator: FaerOperator<'a, S>,
        left: FaerPreconditioner<S>,
        right: FaerPreconditioner<S>,
    ) -> Self {
        let n = operator.nrows();
        let workspace = MemBuffer::new(faer::dyn_stack::StackReq::any_of(&[
            operator.apply_scratch(1, Par::Seq),
            left.apply_scratch(1, Par::Seq),
            right.apply_scratch(1, Par::Seq),
        ]));
        Self {
            operator,
            left,
            right,
            workspace,
            r: Mat::zeros(n, 1),
            p: Mat::zeros(n, 1),
            shadow: Mat::zeros(n, 1),
            y: Mat::zeros(n, 1),
            v: Mat::zeros(n, 1),
            s: Mat::zeros(n, 1),
            z: Mat::zeros(n, 1),
            t: Mat::zeros(n, 1),
            temporary: Mat::zeros(n, 1),
        }
    }

    pub(super) fn solve(
        &mut self,
        mut output: MatMut<'_, S>,
        rhs: MatRef<'_, S>,
        threshold: f64,
        limit: usize,
        mut callback: impl FnMut(MatRef<'_, S>),
    ) -> CycleInfo {
        // Keep the library's zero-RHS correction; the wrapper charges a changed
        // nonaccepted guess once, and checks its original physical criterion.
        if rhs.norm_l2() == 0. {
            output.fill(S::zero());
            callback(output.rb());
            return CycleInfo {
                iter_count: 0,
                breakdown: false,
            };
        }
        let stack = MemStack::new(&mut self.workspace);
        self.operator
            .apply(self.r.as_mut(), output.rb(), Par::Seq, stack);
        for i in 0..rhs.nrows() {
            self.r[(i, 0)] = rhs[(i, 0)] - self.r[(i, 0)];
        }
        if self.r.norm_l2() < threshold {
            return CycleInfo {
                iter_count: 0,
                breakdown: false,
            };
        }
        self.p.copy_from(self.r.as_ref());
        self.shadow.copy_from(self.r.as_ref());
        for iter in 0..limit {
            let attempted = iter + 1;
            self.left
                .apply(self.temporary.as_mut(), self.p.as_ref(), Par::Seq, stack);
            self.right
                .apply(self.y.as_mut(), self.temporary.as_ref(), Par::Seq, stack);
            self.operator
                .apply(self.v.as_mut(), self.y.as_ref(), Par::Seq, stack);
            let denominator = self.shadow.col(0).adjoint() * self.v.col(0);
            let rho = self.shadow.col(0).adjoint() * self.r.col(0);
            let alpha = rho * S::recip_impl(&denominator);
            if !alpha.is_finite() {
                return CycleInfo {
                    iter_count: attempted,
                    breakdown: true,
                };
            }
            for i in 0..rhs.nrows() {
                self.s[(i, 0)] = self.r[(i, 0)] - alpha * self.v[(i, 0)];
                output[(i, 0)] += alpha * self.y[(i, 0)];
            }
            // Alpha is a real candidate, including exact happy breakdown.
            // Retain it before any later omega failure can destroy the iterate.
            callback(output.rb());
            if self.s.norm_l2() < threshold {
                return CycleInfo {
                    iter_count: attempted,
                    breakdown: false,
                };
            }
            self.left
                .apply(self.temporary.as_mut(), self.s.as_ref(), Par::Seq, stack);
            self.right
                .apply(self.z.as_mut(), self.temporary.as_ref(), Par::Seq, stack);
            self.operator
                .apply(self.t.as_mut(), self.z.as_ref(), Par::Seq, stack);
            self.left
                .apply(self.y.as_mut(), self.t.as_ref(), Par::Seq, stack);
            // temporary still holds K1*s; y now holds K1*t. The real positive
            // denominator gives the least-residual complex omega, not t^T*t.
            let omega = (self.y.col(0).adjoint() * self.temporary.col(0))
                * S::recip_impl(&(self.y.col(0).adjoint() * self.y.col(0)));
            if !omega.is_finite() {
                return CycleInfo {
                    iter_count: attempted,
                    breakdown: true,
                };
            }
            for i in 0..rhs.nrows() {
                self.r[(i, 0)] = self.s[(i, 0)] - omega * self.t[(i, 0)];
                output[(i, 0)] += omega * self.z[(i, 0)];
                self.p[(i, 0)] -= omega * self.v[(i, 0)];
            }
            callback(output.rb());
            if self.r.norm_l2() < threshold {
                return CycleInfo {
                    iter_count: attempted,
                    breakdown: false,
                };
            }
            // Scalar specialization of the upstream block recurrence. This is
            // equivalent to rho_next/rho * alpha/omega without dividing by omega.
            let beta = (self.shadow.col(0).adjoint() * self.t.col(0)) * S::recip_impl(&denominator);
            if !beta.is_finite() {
                return CycleInfo {
                    iter_count: attempted,
                    breakdown: true,
                };
            }
            for i in 0..rhs.nrows() {
                self.p[(i, 0)] = self.r[(i, 0)] - beta * self.p[(i, 0)];
            }
        }
        CycleInfo {
            iter_count: limit,
            breakdown: false,
        }
    }
}
