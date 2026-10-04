//! Whole-Model admission for derived quantities, separate from solve inventory.

use super::*;

impl KernelProgram {
    /// Exact declared output support; a lumped Observable has none.
    /// # Errors
    /// Rejects an Observable outside this admitted Model.
    pub fn observable_output_support(
        &self,
        observable: Id<kinds::Observable>,
    ) -> Result<Option<&SpatialSupport<RawId>>, Diagnostic> {
        if !matches!(
            self.node(observable.erase()),
            Some(KernelNode::Observable(_))
        ) {
            return Err(kernel_error(
                observable.erase(),
                "Observable is outside the selected Model",
            ));
        }
        Ok(self.edges.iter().find_map(|edge| {
            (edge.from() == observable.erase() && edge.kind() == EdgeKind::DefinedOn)
                .then(|| self.spatial_supports.get(&edge.to()))
                .flatten()
        }))
    }

    /// Evaluate the retained expression of one finite Observable in this exact Model.
    /// Symbol values come from the caller's admitted numerical context; this performs
    /// no spatial quadrature or solver acceptance.
    ///
    /// # Errors
    /// Rejects foreign Observables, spatial reductions, unavailable symbols, invalid
    /// arithmetic and a result that differs from the declared Observable type.
    pub fn evaluate_finite_observable(
        &self,
        observable: Id<kinds::Observable>,
        resolve: &mut dyn FnMut(SymbolRef) -> Option<eqiora_core::ValueLiteral>,
    ) -> Result<eqiora_core::ValueLiteral, Diagnostic> {
        let Some(KernelNode::Observable(definition)) = self.node(observable.erase()) else {
            return Err(kernel_error(
                observable.erase(),
                "Observable is outside the selected Model",
            ));
        };
        if !edge_targets(&self.edges, observable.erase(), EdgeKind::DefinedOn).is_empty() {
            return Err(kernel_error(
                observable.erase(),
                "finite evaluation cannot erase an Observable output support",
            ));
        }
        if !matches!(
            definition.reduction(),
            eqiora_schema::kernel::ObservableReduction::Value
        ) {
            return Err(kernel_error(
                observable.erase(),
                "finite evaluation cannot perform spatial quadrature",
            ));
        }
        let values = crate::evaluate::evaluate_expression(
            observable.erase(),
            definition.expression(),
            resolve,
        )?;
        let value = values.into_iter().next().ok_or_else(|| {
            kernel_error(
                observable.erase(),
                "Observable expression has no evaluated root",
            )
        })?;
        if value.value_type() != definition.value_type() {
            return Err(kernel_error(
                observable.erase(),
                "evaluated Observable type differs from its admitted declaration",
            ));
        }
        Ok(value)
    }

    /// Infer an Observable's retained expression in this exact Model.
    /// # Errors
    /// Rejects foreign identities or incompatible expression/measure types.
    pub fn typed_observable(
        &self,
        observable: Id<kinds::Observable>,
    ) -> Result<TypedResidual<RawId>, Vec<Diagnostic>> {
        let Some(KernelNode::Observable(definition)) = self.nodes.get(&observable.erase()) else {
            return Err(vec![kernel_error(
                observable.erase(),
                "Observable is outside the selected Model",
            )]);
        };
        let typed = self.type_derived_residual(
            definition.expression().clone(),
            observable.erase(),
            definition
                .reduction()
                .input_domain()
                .map(Id::erase)
                .or_else(|| {
                    edge_targets(&self.edges, observable.erase(), EdgeKind::DefinedOn)
                        .into_iter()
                        .next()
                }),
            RootContract::Observable,
        )?;
        let root = typed
            .node_type(definition.expression().roots()[0])
            .expect("typed root exists");
        definition
            .validate_type(
                root,
                definition
                    .reduction()
                    .input_domain()
                    .and_then(|id| self.spatial_supports.get(&id.erase())),
                definition
                    .reduction()
                    .domain()
                    .and_then(|id| self.spatial_supports.get(&id.erase())),
                field_support(observable.erase(), &self.edges, &self.spatial_supports).as_ref(),
            )
            .map_err(|error| vec![error])?;
        Ok(typed)
    }
}

pub(super) fn validate(
    nodes: &BTreeMap<RawId, KernelNode>,
    edges: &[Edge],
    spatial_supports: &BTreeMap<RawId, SpatialSupport<RawId>>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    validate_dependency_order(nodes, diagnostics);
    for (&id, node) in nodes {
        let KernelNode::Observable(observable) = node else {
            continue;
        };
        if let eqiora_schema::kernel::ObservableReduction::SpatialIntegral {
            domain,
            measure: eqiora_schema::kernel::ObservableMeasure::SphericalVolume,
            ..
        } = observable.reduction()
        {
            let radial = match spatial_supports.get(&domain.erase()) {
                Some(SpatialSupport::Coordinates { factors, .. }) if factors.len() == 1 => {
                    nodes.get(&factors[0].0)
                }
                _ => None,
            };
            if !matches!(radial, Some(KernelNode::Domain(definition))
                if matches!(definition.kind(), eqiora_schema::kernel::DomainKind::CoordinateInterval { bounds }
                    if bounds.lower().value() == 0.0))
            {
                diagnostics.push(kernel_error(id,
                    "spherical volume measure requires one radial coordinate interval from zero to a positive radius"));
            }
        }
        let output_domains = edge_targets(edges, id, EdgeKind::DefinedOn);
        let output = field_support(id, edges, spatial_supports);
        if (!output_domains.is_empty() && output.is_none())
            || output_domains.len() > 1
            || output_domains
                .iter()
                .any(|id| !spatial_supports.contains_key(id))
        {
            diagnostics.push(kernel_error(
                id,
                "Observable output requires at most one admitted Domain",
            ));
        }
        let scope = observable
            .reduction()
            .input_domain()
            .map(Id::erase)
            .or_else(|| output.as_ref().map(|support| *support.domain()));
        let declared_scopes = edge_targets(edges, id, EdgeKind::AppliesOn);
        if declared_scopes
            != observable
                .reduction()
                .domain()
                .map(Id::erase)
                .into_iter()
                .collect()
        {
            diagnostics.push(kernel_error(
                id,
                "Observable AppliesOn must name exactly its integration Domain",
            ));
        }
        let symbols = validate_expression(
            observable.expression(),
            id,
            TypingEnvironment {
                nodes,
                edges,
                spatial_supports,
            },
            scope,
            RootContract::Observable,
            diagnostics,
        );
        if symbols != edge_targets(edges, id, EdgeKind::DependsOn) {
            diagnostics.push(kernel_error(
                id,
                "Observable dependencies differ from its retained expression symbols",
            ));
        }
        if observable.expression().nodes().iter().any(|node| {
            matches!(
                node,
                ExprNode::Symbol(SymbolRef::Derivative(_) | SymbolRef::Pre(_) | SymbolRef::Next(_))
            )
        }) {
            diagnostics.push(kernel_error(
                id,
                "instantaneous Observable cannot read temporal derivative or event-side symbols",
            ));
        }
        let support = scope.and_then(|id| spatial_supports.get(&id));
        if let Ok(typed) = TypedResidual::infer(
            observable.expression().clone(),
            support.cloned(),
            RootContract::Observable,
            |symbol| symbol_type(symbol, nodes, edges, spatial_supports),
        ) {
            let root = typed
                .node_type(observable.expression().roots()[0])
                .expect("typed root exists");
            if let Err(error) = observable.validate_type(
                root,
                observable
                    .reduction()
                    .input_domain()
                    .and_then(|id| spatial_supports.get(&id.erase())),
                observable
                    .reduction()
                    .domain()
                    .and_then(|id| spatial_supports.get(&id.erase())),
                output.as_ref(),
            ) {
                diagnostics.push(error);
            }
        }
    }
}

// Acyclic reduced-value references do not create new solve unknowns or equations.
// Count unique nominal dependencies rather than their occurrences in a density.
fn validate_dependency_order(
    nodes: &BTreeMap<RawId, KernelNode>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let mut counts = BTreeMap::new();
    let mut dependents: BTreeMap<RawId, Vec<RawId>> = BTreeMap::new();
    for (&id, node) in nodes {
        let KernelNode::Observable(observable) = node else {
            continue;
        };
        let dependencies = observable
            .expression()
            .nodes()
            .iter()
            .filter_map(|node| match node {
                ExprNode::Symbol(SymbolRef::Observable(dependency)) => Some(dependency.erase()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        counts.insert(id, dependencies.len());
        for dependency in dependencies {
            dependents.entry(dependency).or_default().push(id);
        }
    }
    let mut ready = counts
        .iter()
        .filter_map(|(&id, &count)| (count == 0).then_some(id))
        .collect::<Vec<_>>();
    while let Some(id) = ready.pop() {
        for dependent in dependents.get(&id).into_iter().flatten() {
            let count = counts
                .get_mut(dependent)
                .expect("Observable dependent is registered");
            *count -= 1;
            if *count == 0 {
                ready.push(*dependent);
            }
        }
    }
    if let Some((&id, _)) = counts.iter().find(|(_, count)| **count != 0) {
        diagnostics.push(kernel_error(
            id,
            "Observable references must form an acyclic graph of live derived quantities",
        ));
    }
}
