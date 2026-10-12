use super::*;

/// Use the bound operator's degree for execution and its portable numerical plan.
pub(in crate::numerical_admission) fn quadrature_points<S: Coefficient>(
    reference: eqiora_meshing::ReferenceCell,
    selected: &[eqiora_realization::DomainFieldDiscretization],
    forms: &[crate::form_compiler::region::BoundRegionForm<S>],
) -> Result<usize, Diagnostic> {
    let floor = if reference.family() == eqiora_meshing::ReferenceCellFamily::Simplex {
        simplex_quadrature_points(reference.dimension(), selected)?
    } else {
        2
    };
    Ok(forms.iter().fold(floor, |points, form| {
        points.max(form.minimum_gauss_points())
    }))
}

/// Use the Space owner's minimum while retaining the existing three-point floor.
fn simplex_quadrature_points(
    dimension: usize,
    selected: &[eqiora_realization::DomainFieldDiscretization],
) -> Result<usize, Diagnostic> {
    let dimension =
        NonZeroUsize::new(dimension).ok_or_else(|| invalid("simplex dimension is zero"))?;
    selected
        .iter()
        .flat_map(|domain| domain.field_spaces())
        .try_fold(3, |points, field| {
            field
                .space()
                .minimum_simplex_duffy_points(dimension)
                .map(|minimum| points.max(minimum.get()))
                .ok_or_else(|| invalid("Space has no admitted simplex Duffy quadrature"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_bubble_quadrature_matches_the_independent_dirichlet_integral() {
        use eqiora_realization::{DomainFieldDiscretization, FieldSpaceBinding};
        let selected = [DomainFieldDiscretization::new(
            eqiora_core::Id::new(),
            [FieldSpaceBinding::new(
                eqiora_core::Id::new(),
                Space::simplex_p1_bubble(),
            )],
            None,
        )
        .unwrap()];
        for d in [2usize, 3] {
            let points = simplex_quadrature_points(d, &selected).unwrap();
            let normalization = ((d + 1) as f64).powi((d + 1) as i32);
            // On the unit d-simplex, integral(prod lambda_i^2)=2!^(d+1)/(3d+2)!.
            // This evaluates the normalized bubble algebraically, not through the FE basis.
            let expected = normalization.powi(2) * 2_f64.powi((d + 1) as i32)
                / (1..=(3 * d + 2)).map(|n| n as f64).product::<f64>();
            let integrate = |n| {
                eqiora_meshing::simplex_duffy_gauss_legendre(d, n)
                    .unwrap()
                    .points()
                    .iter()
                    .map(|p| {
                        let bubble = normalization
                            * (1. - p.coordinates.iter().sum::<f64>())
                            * p.coordinates.iter().product::<f64>();
                        p.weight * bubble.powi(2)
                    })
                    .sum::<f64>()
            };
            assert!((integrate(points) - expected).abs() < 1e-12);
            assert!(
                (integrate(points - 1) - expected).abs() > 1e-8,
                "one fewer point must expose underintegration in dimension {d}"
            );
        }
    }
}
