//! Replay retained functional lineage against an admitted live Observable.
use super::*;
use eqiora_schema::kernel::ObservableDef;

impl AuthoredFormExpressionV1 {
    /// Recompute a retained variation from the exact live energy density.
    ///
    /// This binds functional identity, fixed bindings and the generated body. It does
    /// not establish a strong/weak correspondence, boundary discharge or numerical
    /// compatibility. The caller must obtain the typed density from its live Model
    /// and separately validate test restrictions and the claimed equation.
    ///
    /// # Errors
    /// Rejects stale identities, expressions, fixed bindings or generated terms, and
    /// densities outside the bounded real polynomial fixed spatial measure profile.
    pub fn check_functional_variation(
        &self,
        functional: &ObservableDef,
        density: &TypedResidual<RawId>,
    ) -> Result<(), Diagnostic> {
        let Self::Variation {
            functional_ulid,
            wrt_ulid,
            directions,
            holding,
            value,
        } = self
        else {
            return Err(wire::rejection("expected a retained functional variation"));
        };
        if functional.id().ulid().to_string() != *functional_ulid
            || density.expression() != functional.expression()
        {
            return Err(wire::rejection(
                "variation energy differs from the exact live Observable",
            ));
        }
        let wrt = wrt_ulid
            .parse::<ulid::Ulid>()
            .ok()
            .filter(|id| id.to_string() == *wrt_ulid)
            .map(Id::<kinds::Field>::from_ulid)
            .ok_or_else(|| wire::rejection("variation Field identity is invalid"))?;
        if !(1..=2).contains(&directions.len())
            || directions.iter().any(String::is_empty)
            || (directions.len() == 2 && directions[0] == directions[1])
        {
            return Err(wire::rejection(
                "variation requires one or two independent named directions",
            ));
        }
        let ObservableReduction::SpatialIntegral { domain, measure } = functional.reduction()
        else {
            return Err(wire::rejection(
                "variation requires a fixed spatial Observable",
            ));
        };
        let support = functional_support(density, domain)?;
        if !matches!(
            (measure, support),
            (ObservableMeasure::Volume, SpatialSupport::Volume { .. })
                | (ObservableMeasure::Boundary, SpatialSupport::Boundary { .. })
        ) {
            return Err(wire::rejection(
                "variation measure differs from its live support",
            ));
        }
        functional.validate_type(
            density
                .node_type(density.expression().roots()[0])
                .expect("typed root"),
            Some(support),
        )?;
        let required = functional
            .expression()
            .nodes()
            .iter()
            .filter_map(|node| match node {
                eqiora_schema::kernel::ExprNode::Symbol(SymbolRef::Field(id)) if *id != wrt => {
                    Some(id.erase())
                }
                eqiora_schema::kernel::ExprNode::Symbol(SymbolRef::Parameter(id)) => {
                    Some(id.erase())
                }
                _ => None,
            })
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|id| id.ulid().to_string())
            .collect::<Vec<_>>();
        if *holding != required {
            return Err(wire::rejection(
                "variation fixed bindings differ from its live energy",
            ));
        }
        let expected = derive_value(
            density,
            wrt,
            directions,
            domain,
            functional.value_type().dimension(),
        )?;
        if **value != wire::expression(&expected) {
            return Err(wire::rejection(
                "variation body differs from the live energy derivative",
            ));
        }
        Ok(())
    }
}
