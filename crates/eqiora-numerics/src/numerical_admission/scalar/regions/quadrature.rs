use super::*;

/// Use the bound operator's degree for execution and its portable numerical plan.
pub(in crate::numerical_admission) fn quadrature_points<S: Coefficient>(
    reference: eqiora_meshing::ReferenceCell,
    selected: &[eqiora_realization::DomainFieldDiscretization],
    forms: &[crate::form_compiler::region::BoundRegionForm<S>],
    requested: Option<eqiora_realization::QuadraturePolicy>,
) -> Result<usize, Diagnostic> {
    let floor = if selected
        .iter()
        .flat_map(|domain| domain.field_spaces())
        .all(|field| field.space() == Space::cell_constant())
    {
        1
    } else if reference.family() == eqiora_meshing::ReferenceCellFamily::Simplex {
        simplex_quadrature_points(reference.dimension(), selected)?
    } else {
        2
    };
    let minimum = forms.iter().fold(floor, |points, form| {
        points
            .max(form.minimum_gauss_points())
            .max(form.retained_assembly_points().unwrap_or(0))
    });
    if let Some(policy) = requested {
        use eqiora_realization::QuadraturePolicy;
        let points = match (reference.family(), policy) {
            (
                eqiora_meshing::ReferenceCellFamily::Hypercube,
                QuadraturePolicy::GaussLegendre { points_per_axis },
            ) => points_per_axis.get(),
            (
                eqiora_meshing::ReferenceCellFamily::Simplex,
                QuadraturePolicy::SimplexDuffyGaussLegendre {
                    spatial_dimension,
                    points_per_axis,
                },
            ) if spatial_dimension.get() == reference.dimension() => points_per_axis.get(),
            _ => {
                return Err(invalid(
                    "assembly quadrature provider or family does not match the admitted reference cell",
                ));
            }
        };
        if points < minimum {
            return Err(invalid(
                "assembly quadrature underintegrates the bound polynomial form",
            ));
        }
        // Let the existing rule owner reject unsupported index/allocation/root controls.
        eqiora_meshing::QuadratureRule::gauss_legendre(points)?;
        return Ok(points);
    }
    let quadratic_tensor = reference.family() == eqiora_meshing::ReferenceCellFamily::Hypercube
        && selected.iter().flat_map(|domain| domain.field_spaces()).any(|field| matches!(field.space().family(), eqiora_realization::SpaceFamily::ContinuousLagrange { order } if order.get() == 2));
    // Four points is an accuracy default, not an exactness guarantee for nonpolynomials.
    Ok(minimum.max(if quadratic_tensor { 4 } else { 1 }))
}

impl<S: Coefficient> ExecutableSpatialEquations<S> {
    pub(in crate::numerical_admission) fn assembly_quadrature_policy(
        &self,
        resources: &NativeMeshResources,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        scaling: &eqiora_realization::SystemScaling,
        temporal: Option<&CommonTemporalPolicy>,
        requested: Option<eqiora_realization::QuadraturePolicy>,
    ) -> Result<eqiora_realization::QuadraturePolicy, Diagnostic> {
        let dimension = resources.geometry()?.ambient_dimension();
        let simplex = matches!(
            resources,
            NativeMeshResources::MshSimplicial { .. }
                | NativeMeshResources::AffineTriangleSimplicial { .. }
        );
        let reference = if simplex {
            eqiora_meshing::ReferenceCell::simplex(dimension)?
        } else {
            eqiora_meshing::ReferenceCell::hypercube(dimension)?
        };
        let mut equations = self.clone();
        if let Some(temporal) = temporal.and_then(CommonTemporalPolicy::as_backward_euler) {
            for region in &mut equations.regions {
                region.form = region.form.bind_backward_euler(temporal)?;
            }
        }
        let forms = equations.bind_spaces(reference, selected, scaling, requested)?;
        let points_per_axis =
            NonZeroUsize::new(quadrature_points(reference, selected, &forms, requested)?)
                .expect("positive Gauss point count");
        Ok(if simplex {
            eqiora_realization::QuadraturePolicy::SimplexDuffyGaussLegendre {
                spatial_dimension: NonZeroUsize::new(dimension)
                    .expect("positive reference dimension"),
                points_per_axis,
            }
        } else {
            eqiora_realization::QuadraturePolicy::GaussLegendre { points_per_axis }
        })
    }
}

/// Keep the established simplex accuracy floor; bound forms own exactness.
fn simplex_quadrature_points(
    dimension: usize,
    selected: &[eqiora_realization::DomainFieldDiscretization],
) -> Result<usize, Diagnostic> {
    let reference = eqiora_meshing::ReferenceCell::simplex(dimension)?;
    for field in selected.iter().flat_map(|domain| domain.field_spaces()) {
        // No default for an unknown or unbound Space/reference combination.
        crate::discrete_space::DiscreteSpace::new(field.space(), reference)?;
    }
    Ok(3)
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
            let points = selected[0].field_spaces()[0]
                .space()
                .minimum_simplex_duffy_points(NonZeroUsize::new(d).unwrap())
                .unwrap()
                .get();
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
