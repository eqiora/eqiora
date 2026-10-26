//! Preserve the anchored Q1 energy proof through the actual binary64 assembly.
use super::*;
use eqiora_assembly::{AssemblyTargetId, CsrMatrix};
use eqiora_core::Scalar;
use num_complex::ComplexFloat;
use num_traits::Zero;

impl<S: Coefficient + Send + Sync, P: Projection<S>> PreparedRegionAssembly<S, P>
where
    P::Scalar: Send + Sync,
{
    /// The caller proves original scalar 1D Q1 diffusion and component anchors.
    /// This checks representability of that energy, never numerical matrix rank.
    pub(crate) fn preserves_q1_diffusion_energy(
        &self,
        plan: &AssemblyPlan,
        matrix: &CsrMatrix<P::Scalar>,
    ) -> Result<bool, Diagnostic> {
        if S::DOMAIN != eqiora_core::ScalarDomain::Real
            || P::Scalar::DOMAIN != eqiora_core::ScalarDomain::Real
            || matrix.rows() == 0
            || matrix.rows() != matrix.columns()
        {
            return Ok(false);
        }
        let reduced = plan.target_id(0).expect("reduced assembly target");
        let full = plan.target_id(1).expect("full assembly target");
        let mut incidents = vec![0usize; matrix.rows()];
        let mut minimum = f64::INFINITY;
        let mut maximum = 0_f64;
        for packet in &self.packets {
            let local = match &packet.action {
                RegionPacketAction::Cell { prepared, previous } => prepared.evaluate(previous)?,
                RegionPacketAction::Boundary(facet) => {
                    if facet
                        .affine()
                        .matrix()
                        .iter()
                        .any(|value| *value != P::Scalar::zero())
                    {
                        return Ok(false);
                    }
                    continue;
                }
                RegionPacketAction::Affine(_) => return Ok(false),
            }
            .reoriented(&packet.orientation, &packet.orientation)?;
            if local.rows() != 2 || local.columns() != 2 {
                return Ok(false);
            }
            let energy = local.matrix()[0];
            if !energy.is_finite()
                || energy.im() != 0.
                || energy.re() <= 0.
                || local.matrix() != [energy, -energy, -energy, energy]
            {
                return Ok(false);
            }
            let map = |target: AssemblyTargetId| {
                packet
                    .mappings
                    .iter()
                    .find(|mapping| mapping.target() == target)
                    .map(TargetAssemblyMap::map)
            };
            let full = map(full).expect("prepared full target");
            let [LocalUnknown::Free(left), LocalUnknown::Free(right)] = full.unknowns() else {
                return Ok(false);
            };
            // Equal prescribed values do not imply equal global identities.
            // Quotient self-loops, including both-fixed ones, have no edge energy.
            if left == right {
                return Ok(false);
            }
            let reduced = map(reduced).expect("prepared reduced target");
            let mut active = false;
            for (row, unknown) in reduced.equations().iter().zip(reduced.unknowns()) {
                match (row, unknown) {
                    (Some(row), LocalUnknown::Free(column)) if row == column => {
                        incidents[row.index()] += 1;
                        active = true;
                    }
                    (None, LocalUnknown::Fixed(_)) => {}
                    _ => return Ok(false),
                }
            }
            if active {
                minimum = minimum.min(energy.re());
                maximum = maximum.max(energy.re());
            }
        }
        if incidents.contains(&0) || !minimum.is_finite() {
            return Ok(false);
        }
        let degree = *incidents.iter().max().expect("nonempty reduced target");
        Ok(
            energy_dominates_roundoff(minimum, maximum, degree, matrix.rows())
                && exactly_symmetric(matrix),
        )
    }
}

fn exactly_symmetric<S: Coefficient + Sync>(matrix: &CsrMatrix<S>) -> bool {
    for row in 0..matrix.rows() {
        for slot in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
            let column = matrix.column_indices()[slot];
            let range = matrix.row_offsets()[column]..matrix.row_offsets()[column + 1];
            let Ok(mirror) = matrix.column_indices()[range.clone()].binary_search(&row) else {
                return false;
            };
            if matrix.values()[slot] != matrix.values()[range.start + mirror] {
                return false;
            }
        }
    }
    true
}

fn energy_dominates_roundoff(minimum: f64, maximum: f64, degree: usize, free: usize) -> bool {
    // A path from each free vertex to its component's prescribed vertex has at
    // most N free edges. Cauchy-Schwarz gives lambda_min(A_exact)>=c_min/N^2.
    // Same-sign accumulation of at most D cell contributions per entry, with
    // D*u<=1/2, gives ||A_stored-A_exact||_2<=4*u*D^2*c_max (u=2^-53).
    // Thus a strict energy margin and exact final symmetry preserve SPD.
    // Integer exponent bounds make this conservative comparison exact even for
    // subnormal energies, without overflow, divisions or a numerical rank test.
    let ceil_log2 = |value: usize| (usize::BITS - (value - 1).leading_zeros()) as i32;
    let degree = ceil_log2(degree);
    let free = ceil_log2(free);
    degree <= 52 && exponent(minimum) - exponent(maximum) >= 2 * degree + 2 * free - 50
}

fn exponent(value: f64) -> i32 {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    if exponent == 0 {
        -1074 + (63 - bits.leading_zeros()) as i32
    } else {
        exponent - 1023
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponent_margin_is_sufficient_rather_than_a_condition_number_claim() {
        // N=D=2 gives c_min/c_max>2^-47 in the power-of-two envelope.
        // The 2^-47 case is still mathematically SPD but deliberately unproved.
        assert!(energy_dominates_roundoff(2_f64.powi(-46), 1., 2, 2));
        assert!(!energy_dominates_roundoff(2_f64.powi(-47), 1., 2, 2));
        assert!(!energy_dominates_roundoff(2_f64.powi(-60), 1., 2, 2));
        assert_eq!(exponent(f64::from_bits(1)), -1074);
        assert_eq!(exponent(f64::MIN_POSITIVE), -1022);
        assert!(energy_dominates_roundoff(
            f64::from_bits(1),
            f64::from_bits(1),
            2,
            2
        ));
    }

    #[test]
    fn rounded_down_interface_diagonal_does_not_invalidate_the_energy_margin() {
        let c = 1. + f64::EPSILON;
        assert_eq!(1. + c, 2.);
        // Three free vertices after the left anchor; edge energies [1,1,c].
        // Leading minors are 2,3,c*(3-2*c), all strictly positive. The middle
        // diagonal is below its exact off-diagonal absolute sum nevertheless.
        assert!(c * (3. - 2. * c) > 0.);
        assert!(energy_dominates_roundoff(1., c, 2, 3));
        let matrix = CsrMatrix::from_sorted_csr(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![2., -1., -1., 2., -c, -c, c],
        )
        .unwrap();
        assert!(exactly_symmetric(&matrix));
    }
}
