//! Replay retained functional lineage against an admitted live Observable.
use super::*;
use eqiora_schema::kernel::ObservableDef;

impl AuthoredFormExpressionV1 {
    /// Derive one fixed spatial functional through the authored-form calculus.
    ///
    /// The returned integral retains the exact Domain and named ordered Field
    /// directions. Other local inputs stay fixed. Numerical callers bind those
    /// same names to dimensioned coefficients on the exact accepted Result;
    /// this does not prove strong/weak correspondence or admissible boundary traces.
    /// # Errors
    /// Rejects mismatched live expressions, reductions or types, unsupported
    /// local calculus, and invalid first/second direction inventories.
    pub fn derive_spatial_variation(
        functional: &ObservableDef,
        density: &TypedResidual<RawId>,
        wrt: Id<kinds::Field>,
        directions: &[String],
    ) -> Result<Self, Diagnostic> {
        if functional.expression() != density.expression()
            || !(1..=2).contains(&directions.len())
            || directions.iter().any(String::is_empty)
            || (directions.len() == 2 && directions[0] == directions[1])
        {
            return Err(wire::rejection(
                "variation requires the exact live density and independent named directions",
            ));
        }
        let ObservableReduction::SpatialIntegral { domain, measure } = functional.reduction()
        else {
            return Err(wire::rejection(
                "local variation requires a fixed spatial integral",
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
        let root = density
            .node_type(density.expression().roots()[0])
            .ok_or_else(|| wire::rejection("variation density has no typed root"))?;
        functional.validate_type(root, Some(support))?;
        let value = derive_value(
            density,
            wrt,
            directions,
            domain,
            functional.value_type().dimension(),
            &mut 65536,
        )?;
        Ok(wire::expression(&value))
    }

    /// Recompute a retained variation from its exact live energy and dependencies.
    ///
    /// This binds functional identity, fixed bindings and the generated body. It does
    /// not establish a strong/weak correspondence, boundary discharge or numerical
    /// compatibility. The resolver must obtain each typed definition from the live
    /// Model. The caller separately validates test restrictions and the equation.
    ///
    /// # Errors
    /// Rejects stale identities, expressions, fixed bindings or generated terms, and
    /// densities outside the bounded real polynomial fixed spatial measure profile.
    pub fn check_functional_variation(
        &self,
        resolve: &mut dyn FnMut(
            Id<kinds::Observable>,
        )
            -> Result<(ObservableDef, TypedResidual<RawId>), Diagnostic>,
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
        let functional = functional_ulid
            .parse::<ulid::Ulid>()
            .ok()
            .filter(|id| id.to_string() == *functional_ulid)
            .map(Id::<kinds::Observable>::from_ulid)
            .ok_or_else(|| wire::rejection("variation Observable identity is invalid"))?;
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
        let derived = composite::derive(functional, wrt, directions, resolve)?;
        let required = derived
            .holding
            .into_iter()
            .map(|id| id.ulid().to_string())
            .collect::<Vec<_>>();
        if *holding != required {
            return Err(wire::rejection(
                "variation fixed bindings differ from its live energy",
            ));
        }
        let expected = derived.value;
        if **value != wire::expression(&expected) {
            return Err(wire::rejection(
                "variation body differs from the live energy derivative",
            ));
        }
        Ok(())
    }
}
