//! Static observation meaning belongs to the exact Plan, before primal execution.
use super::{ResolvedCommonPlan, invalid};
use eqiora_core::{Diagnostic, Id, entity::kinds};
use eqiora_meshing::{QuadratureRule, ReferenceCell, ReferenceCellFamily};
use std::collections::HashMap;

impl ResolvedCommonPlan {
    /// Select explicit quadrature for every spatial measure consumed by selected Observables.
    ///
    /// Coordinate factors use tensor-product Gauss rules. Physical measures use
    /// the exact admitted Mesh family, including simplex Duffy rules. The caller
    /// chooses the points per axis; no scientific accuracy is inferred from this
    /// count. Explicit callers may instead supply their own admitted rules.
    /// # Errors
    /// Rejects missing Observables, finite outputs without an integration measure,
    /// unavailable supports and invalid quadrature orders.
    pub fn observation_quadratures(
        &self,
        observables: &[Id<kinds::Observable>],
        points_per_axis: usize,
    ) -> Result<HashMap<Id<kinds::Domain>, QuadratureRule>, Diagnostic> {
        let (measures, only_boundaries) = self.observation_reference_cells(observables)?;
        if measures.is_empty() {
            return Err(invalid(
                "finite Observable does not accept spatial quadrature",
            ));
        }
        measures
            .into_iter()
            .map(|(domain, cell)| {
                let rule = match cell.family() {
                    ReferenceCellFamily::Point => {
                        if points_per_axis != 1 && only_boundaries {
                            return Err(invalid("point measure requires quadrature_points=1"));
                        }
                        QuadratureRule::point()
                    }
                    ReferenceCellFamily::Simplex => eqiora_meshing::simplex_duffy_gauss_legendre(
                        cell.dimension(),
                        points_per_axis,
                    )?,
                    ReferenceCellFamily::Hypercube => {
                        QuadratureRule::tensor_product_gauss_legendre(
                            cell.dimension(),
                            points_per_axis,
                        )?
                    }
                };
                Ok((domain, rule))
            })
            .collect()
    }

    /// Validate explicit rules against all exact measures without executing a primal.
    /// Finite selected values require an empty rule map. Quadrature accuracy is the
    /// caller's declared numerical choice, not inferred from geometric compatibility.
    pub fn validate_observation_quadratures(
        &self,
        observables: &[Id<kinds::Observable>],
        rules: &HashMap<Id<kinds::Domain>, QuadratureRule>,
    ) -> Result<(), Diagnostic> {
        let (measures, _) = self.observation_reference_cells(observables)?;
        if measures.len() != rules.len()
            || measures.iter().any(|(domain, cell)| {
                rules
                    .get(domain)
                    .is_none_or(|rule| rule.reference_cell() != *cell)
            })
        {
            return Err(invalid(
                "observation quadrature differs from the exact selected measures",
            ));
        }
        Ok(())
    }

    fn observation_reference_cells(
        &self,
        observables: &[Id<kinds::Observable>],
    ) -> Result<(HashMap<Id<kinds::Domain>, ReferenceCell>, bool), Diagnostic> {
        use eqiora_schema::kernel::typing::SpatialSupport;
        use eqiora_schema::kernel::{
            ExprNode, KernelNode, ObservableMeasure, ObservableReduction, SymbolRef,
        };
        let program = self.observation_program()?;
        let mut pending = observables.to_vec();
        let mut visited = std::collections::HashSet::new();
        let mut measures = HashMap::new();
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            let Some(KernelNode::Observable(definition)) = program.node(id.erase()) else {
                return Err(invalid("Observable is outside this exact Plan Model"));
            };
            if let ObservableReduction::SpatialIntegral {
                domain, measure, ..
            } = definition.reduction()
            {
                measures.insert(domain, measure);
            }
            if let ObservableReduction::ProbabilityMoment { distribution, .. } =
                definition.reduction()
                && let Some(KernelNode::Relation(law)) = program.node(distribution.erase())
                && let eqiora_schema::kernel::RelationMeaning::ProbabilityDensity {
                    domain,
                    measure,
                    ..
                } = law.meaning()
            {
                measures.insert(*domain, *measure);
            }
            pending.extend(
                definition
                    .expression()
                    .nodes()
                    .iter()
                    .filter_map(|node| match node {
                        ExprNode::Symbol(SymbolRef::Observable(dependency)) => Some(*dependency),
                        _ => None,
                    }),
            );
        }
        let only_boundaries = measures
            .values()
            .all(|measure| *measure == ObservableMeasure::Boundary);
        let cells = measures
            .keys()
            .map(|domain| {
                let support = program
                    .spatial_support(*domain)
                    .ok_or_else(|| invalid("Observable measure Domain is unavailable"))?;
                let dimension = support.intrinsic_dimensions();
                let cell = if dimension == 0 {
                    ReferenceCell::point()
                } else if !matches!(support, SpatialSupport::Coordinates { .. })
                    && self
                        .authenticated_mesh()
                        .is_some_and(|mesh| mesh.simplicial_mesh().is_some())
                {
                    ReferenceCell::simplex(dimension)?
                } else {
                    ReferenceCell::hypercube(dimension)?
                };
                Ok((*domain, cell))
            })
            .collect::<Result<_, Diagnostic>>()?;
        Ok((cells, only_boundaries))
    }
    /// Borrow the admitted program, retaining any Plan-bound physical Geometry.
    pub(crate) fn observation_program(
        &self,
    ) -> Result<std::borrow::Cow<'_, eqiora_sem::KernelProgram>, Diagnostic> {
        use std::borrow::Cow;
        if let Some(plan) = self.as_spatial() {
            Ok(Cow::Borrowed(plan.kernel()))
        } else if let Some(plan) = self.as_algebraic() {
            Ok(Cow::Borrowed(plan.kernel()))
        } else {
            self.model_artifact()
                .to_program()
                .map(Cow::Owned)
                .map_err(|errors| {
                    errors
                        .into_iter()
                        .next()
                        .expect("failed replay has diagnostic")
                })
        }
    }
}
