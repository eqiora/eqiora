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

    /// Evaluate explicit lower/upper mathematical limits using the canonical expression evaluator.
    /// This selects only the limit roots; it neither samples the density nor performs quadrature.
    ///
    /// # Errors
    /// Rejects foreign Observables, unavailable inputs, non-finite arithmetic and wrong limit units.
    pub fn evaluate_observable_limits(
        &self,
        observable: Id<kinds::Observable>,
        resolve: &mut dyn FnMut(SymbolRef) -> Option<eqiora_core::ValueLiteral>,
    ) -> Result<Option<[eqiora_core::DynQuantity; 2]>, Diagnostic> {
        let Some(KernelNode::Observable(definition)) = self.node(observable.erase()) else {
            return Err(kernel_error(
                observable.erase(),
                "integral limits require an exact retained Observable",
            ));
        };
        let Some(limits) = definition.reduction().limits() else {
            return Ok(None);
        };
        let typed = self
            .typed_observable(observable)
            .map_err(|errors| errors.into_iter().next().expect("failed typing"))?;
        let values = crate::ExpressionBackend::evaluate(
            &crate::evaluate::ReferenceExpressionBackend,
            observable.erase(),
            definition.expression(),
            &limits,
            resolve,
        )?;
        let mut checked = Vec::with_capacity(2);
        for (value, limit) in values.into_iter().zip(limits) {
            if value.value_type() != &typed.node_type(limit).expect("typed limit").value_type {
                return Err(kernel_error(
                    observable.erase(),
                    "evaluated integral limit has the wrong coordinate unit or type",
                ));
            }
            checked.push(value.real_scalar_value().ok_or_else(|| {
                kernel_error(
                    observable.erase(),
                    "integral limit requires a finite real scalar",
                )
            })?);
        }
        Ok(Some(checked.try_into().expect("two selected limit roots")))
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
        self.evaluate_observable_with_points(observable, None, &mut |input, point| {
            let crate::EvaluationInput::Value(symbol) = input else {
                return Err(kernel_error(
                    observable.erase(),
                    "coordinate partial requires an admitted point reconstruction",
                ));
            };
            if point.is_some() {
                let spatial = match symbol {
                    SymbolRef::Field(id) | SymbolRef::Derivative(id, _) => {
                        self.edges.iter().any(|edge| {
                            edge.from() == id.erase()
                                && edge.kind() == EdgeKind::DefinedOn
                                && matches!(self.node(edge.to()), Some(KernelNode::Domain(_)))
                        })
                    }
                    SymbolRef::Observable(id) => self.observable_output_support(id)?.is_some(),
                    _ => false,
                };
                if spatial {
                    return Err(kernel_error(
                        observable.erase(),
                        "point evaluation of a spatial Field requires an admitted reconstruction",
                    ));
                }
            }
            resolve(symbol).ok_or_else(|| {
                kernel_error(
                    observable.erase(),
                    format!("Observable input is unavailable: {symbol:?}"),
                )
            })
        })
    }

    /// Evaluate an Observable with exact point contexts for spatial reconstruction.
    /// A supported output requires a point on precisely that output support.
    /// Arithmetic and conditional demand remain with the canonical evaluator. The callback
    /// supplies admitted numerical Field values; it must not advance state or infer a representation.
    /// # Errors
    /// Rejects foreign Observables, reductions, invalid coordinates/sides and callback failures.
    pub fn evaluate_observable_with_points(
        &self,
        observable: Id<kinds::Observable>,
        point: Option<&crate::EvaluationPoint>,
        resolve: &mut impl FnMut(
            crate::EvaluationInput,
            Option<&crate::EvaluationPoint>,
        ) -> Result<eqiora_core::ValueLiteral, Diagnostic>,
    ) -> Result<eqiora_core::ValueLiteral, Diagnostic> {
        let Some(KernelNode::Observable(definition)) = self.node(observable.erase()) else {
            return Err(kernel_error(
                observable.erase(),
                "Observable is outside the selected Model",
            ));
        };
        if let Some(point) = point {
            point.validate(self)?;
        }
        if let Some(support) = self.observable_output_support(observable)?
            && point.is_none_or(|point| point.domain().erase() != *support.domain())
        {
            return Err(kernel_error(
                observable.erase(),
                "point evaluation requires the exact Observable output support",
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
        let values = crate::evaluate::evaluate_with_points(
            self,
            observable.erase(),
            definition.expression(),
            point,
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

    /// Evaluate the retained density of an integral at an exact input point.
    /// The numerical owner separately admits regularity, chooses quadrature and
    /// applies the declared measure. This operation performs none of those steps.
    /// # Errors
    /// Rejects foreign Observables, non-integral reductions, foreign input points,
    /// unavailable reconstruction inputs and values differing from the typed density.
    pub fn evaluate_observable_density_with_points(
        &self,
        observable: Id<kinds::Observable>,
        point: &crate::EvaluationPoint,
        resolve: &mut impl FnMut(
            crate::EvaluationInput,
            Option<&crate::EvaluationPoint>,
        ) -> Result<eqiora_core::ValueLiteral, Diagnostic>,
    ) -> Result<eqiora_core::ValueLiteral, Diagnostic> {
        let Some(KernelNode::Observable(definition)) = self.node(observable.erase()) else {
            return Err(kernel_error(
                observable.erase(),
                "density requires an exact retained Observable",
            ));
        };
        let eqiora_schema::kernel::ObservableReduction::SpatialIntegral { input, .. } =
            definition.reduction()
        else {
            return Err(kernel_error(
                observable.erase(),
                "density evaluation requires an integral Observable",
            ));
        };
        if point.domain() != input {
            return Err(kernel_error(
                observable.erase(),
                "density point requires the exact integral input support",
            ));
        }
        point.validate(self)?;
        let typed = self
            .typed_observable(observable)
            .map_err(|errors| errors.into_iter().next().expect("failed typing"))?;
        let values = crate::evaluate::evaluate_with_points(
            self,
            observable.erase(),
            definition.expression(),
            Some(point),
            resolve,
        )?;
        let value = values.into_iter().next().ok_or_else(|| {
            kernel_error(observable.erase(), "integral density has no evaluated root")
        })?;
        let root = definition.expression().roots()[0];
        if value.value_type() != &typed.node_type(root).expect("typed density").value_type {
            return Err(kernel_error(
                observable.erase(),
                "evaluated density differs from its admitted type",
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
                definition.reduction().limits().map(|limits| {
                    limits.map(|id| typed.node_type(id).expect("typed limit exists"))
                }),
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
                ExprNode::Symbol(SymbolRef::Pre(_) | SymbolRef::Next(_))
            )
        }) {
            diagnostics.push(kernel_error(
                id,
                "instantaneous Observable cannot read event-side symbols",
            ));
        }
        let support = scope.and_then(|id| spatial_supports.get(&id));
        if let Ok(typed) = TypedResidual::infer(
            observable.expression().clone(),
            support.cloned(),
            |on| spatial_supports.get(&on.erase()).cloned(),
            RootContract::Observable,
            |symbol| symbol_type(symbol, nodes, edges, spatial_supports),
        ) {
            let root = typed
                .node_type(observable.expression().roots()[0])
                .expect("typed root exists");
            if let Err(error) = observable.validate_type(
                root,
                observable.reduction().limits().map(|limits| {
                    limits.map(|id| typed.node_type(id).expect("typed limit exists"))
                }),
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
