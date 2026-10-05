//! Informational incidence projections; numerical admission remains separate.
use super::*;
use eqiora_ir::{ComponentScalarization, ScalarSymbolCoordinate};

/// One scalar equation occurrence in a continuous structural analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
struct EquationIncidence {
    owner: RawId,
    ordinal: usize,
    coordinates: BTreeSet<Coordinate>,
}

/// A necessary incidence matching, not a numerical rank certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncidenceMatching {
    pub(super) variables: BTreeMap<Coordinate, usize>,
    pub(super) rows: Vec<Vec<usize>>,
    pub(super) matched: Vec<Option<usize>>,
    over: IncidenceBlock,
    under: IncidenceBlock,
}

impl IncidenceMatching {
    /// Candidate coordinates in deterministic column order.
    pub fn coordinates(&self) -> impl Iterator<Item = &ScalarSymbolCoordinate> + '_ {
        self.variables.keys().map(|coordinate| &coordinate.0)
    }
    /// Maximum incidence matching cardinality; this is not numerical rank.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.matched.iter().flatten().count()
    }
    /// Indices into [`EquationAnalysis::equations`] in the coarse excess block.
    pub fn overdetermined_equations(&self) -> impl Iterator<Item = usize> + '_ {
        self.over.0.iter().copied()
    }
    /// Indices into [`EquationAnalysis::equations`] in the coarse deficient block.
    pub fn underdetermined_equations(&self) -> impl Iterator<Item = usize> + '_ {
        self.under.0.iter().copied()
    }
    /// Coordinates participating in the coarse excess block.
    pub fn overdetermined_coordinates(&self) -> impl Iterator<Item = &ScalarSymbolCoordinate> + '_ {
        self.variables
            .iter()
            .filter(|(_, column)| self.over.1.contains(column))
            .map(|(coordinate, _)| &coordinate.0)
    }
    /// Coordinates participating in the coarse deficient block.
    pub fn underdetermined_coordinates(
        &self,
    ) -> impl Iterator<Item = &ScalarSymbolCoordinate> + '_ {
        self.variables
            .iter()
            .filter(|(_, column)| self.under.1.contains(column))
            .map(|(coordinate, _)| &coordinate.0)
    }
}

/// Continuous scalar occurrence analysis before initialization or state selection.
/// Initial equations are separate and cannot repair regular equation balance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EquationAnalysis {
    equations: Vec<EquationIncidence>,
    balance: IncidenceMatching,
    rate_partition: IncidenceMatching,
}

impl EquationAnalysis {
    /// Continuous rows as (exact owner, zero-based scalar row ordinal, coordinates).
    /// Finite rows follow root, row-major component, then real/imaginary order.
    /// Value and derivative coordinates remain distinct. Row indices in matching
    /// blocks refer to this deterministic iterator order.
    pub fn equations(
        &self,
    ) -> impl ExactSizeIterator<
        Item = (
            RawId,
            usize,
            impl Iterator<Item = &ScalarSymbolCoordinate> + '_,
        ),
    > + '_ {
        self.equations.iter().map(|row| {
            (
                row.owner,
                row.ordinal,
                row.coordinates.iter().map(|coordinate| &coordinate.0),
            )
        })
    }
    /// Necessary balance, identifying a Field and its derivative as one unknown.
    #[must_use]
    pub const fn balance(&self) -> &IncidenceMatching {
        &self.balance
    }
    /// Candidate using rates for authored differential Fields and values for
    /// algebraic Fields/Ports. This neither selects states nor admits execution:
    /// a valid coupled constant-mass descriptor may have a deficient candidate.
    #[must_use]
    pub const fn declared_rate_partition(&self) -> &IncidenceMatching {
        &self.rate_partition
    }
}

pub(super) fn analyze(
    program: &KernelProgram,
    plan: &ExecutionPlan,
) -> Result<EquationAnalysis, Diagnostic> {
    let variables = plan
        .differential_orders
        .keys()
        .chain(&plan.algebraic_fields)
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(Variable::Field)
        .chain(plan.continuous_ports.iter().copied().map(Variable::Port))
        .chain(
            plan.physical_unknowns
                .iter()
                .copied()
                .map(Variable::Physical),
        )
        .map(|variable| {
            ScalarSymbolCoordinate::for_value(variable.symbol(), &variable.value_type(program)?)
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?
        .into_iter()
        .flatten()
        .map(Coordinate)
        .collect::<BTreeSet<_>>();
    let mut equations = Vec::new();
    for &owner in &plan.continuous_relations {
        let Some(KernelNode::Relation(relation)) = program.node(owner) else {
            return Err(execution_error("validated Relation is unavailable", 0.0));
        };
        let shaped = relation.expression().nodes().iter().any(|node| {
            if let ExprNode::Symbol(symbol) = node {
                normalized_symbol(*symbol, &plan.signal_sources).is_some()
                    && program.execution_symbol_type(*symbol).is_some_and(|ty| {
                        matches!(
                            ty.scalar_domain(),
                            eqiora_core::ScalarDomain::Real | eqiora_core::ScalarDomain::Complex
                        ) && (!ty.shape().is_scalar()
                            || ty.scalar_domain() == eqiora_core::ScalarDomain::Complex)
                    })
            } else if let ExprNode::Constant(value) = node {
                value.value_type().scalar_domain() == eqiora_core::ScalarDomain::Complex
            } else {
                matches!(node, ExprNode::Complex { .. })
            }
        });
        if shaped {
            let typed = program
                .typed_relation_residual(relation.id())
                .map_err(|errors| errors.into_iter().next().expect("typing failure"))?;
            let operator = ComponentScalarization::lower(&typed)?;
            for (ordinal, row) in operator.rows().iter().enumerate() {
                let coordinates = row
                    .symbols()
                    .iter()
                    .filter_map(|source| {
                        normalized_symbol(source.symbol(), &plan.signal_sources)
                            .map(|symbol| Coordinate(source.with_symbol(symbol)))
                    })
                    .collect();
                equations.push(EquationIncidence {
                    owner,
                    ordinal,
                    coordinates,
                });
            }
            continue;
        }
        for (ordinal, (left, right)) in relation.equation_sides().enumerate() {
            equations.push(EquationIncidence {
                owner,
                ordinal,
                coordinates: scalar_incidence(
                    program,
                    relation.expression(),
                    &[left, right],
                    &plan.signal_sources,
                )?,
            });
        }
    }
    for system in &plan.physical_systems {
        for junction in system.junctions() {
            for (ordinal, &root) in junction.dag().roots().iter().enumerate() {
                equations.push(EquationIncidence {
                    owner: junction.owner().erase(),
                    ordinal,
                    coordinates: scalar_incidence(
                        program,
                        junction.dag(),
                        &[root],
                        &plan.signal_sources,
                    )?,
                });
            }
        }
    }
    let rate_variables = variables
        .iter()
        .map(|coordinate| match coordinate.symbol() {
            SymbolRef::Field(field) if plan.differential_orders.contains_key(&field.erase()) => {
                coordinate.with_symbol(SymbolRef::Derivative(
                    field,
                    plan.differential_orders[&field.erase()],
                ))
            }
            _ => coordinate.clone(),
        })
        .collect();
    let balance = matching(&equations, variables, true);
    let rate_partition = matching(&equations, rate_variables, false);
    Ok(EquationAnalysis {
        equations,
        balance,
        rate_partition,
    })
}

fn matching(
    equations: &[EquationIncidence],
    variables: BTreeSet<Coordinate>,
    merge_rates: bool,
) -> IncidenceMatching {
    let variables = variables
        .into_iter()
        .enumerate()
        .map(|(column, coordinate)| (coordinate, column))
        .collect::<BTreeMap<_, _>>();
    let rows = equations
        .iter()
        .map(|equation| {
            equation
                .coordinates
                .iter()
                .map(|coordinate| match coordinate.symbol() {
                    SymbolRef::Derivative(field, _) if merge_rates => {
                        coordinate.with_symbol(SymbolRef::Field(field))
                    }
                    _ => coordinate.clone(),
                })
                .filter_map(|coordinate| variables.get(&coordinate).copied())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect()
        })
        .collect::<Vec<_>>();
    let matched = maximum_matching(&rows, variables.len());
    let (over, under) = deficient_blocks(&rows, variables.len(), &matched);
    IncidenceMatching {
        variables,
        rows,
        matched,
        over,
        under,
    }
}

fn scalar_incidence(
    program: &KernelProgram,
    dag: &ExprDag,
    roots: &[ExprId],
    sources: &BTreeMap<RawId, RawId>,
) -> Result<BTreeSet<Coordinate>, Diagnostic> {
    incidence::variables(dag, roots, sources)?
        .into_iter()
        .filter(|variable| {
            variable.value_type(program).is_ok_and(|ty| {
                matches!(
                    ty.scalar_domain(),
                    eqiora_core::ScalarDomain::Real | eqiora_core::ScalarDomain::Complex
                )
            })
        })
        .map(|variable| {
            ScalarSymbolCoordinate::for_value(variable.symbol(), &variable.value_type(program)?)
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|coordinates| coordinates.into_iter().flatten().map(Coordinate).collect())
}

fn normalized_symbol(symbol: SymbolRef, sources: &BTreeMap<RawId, RawId>) -> Option<SymbolRef> {
    match symbol {
        SymbolRef::Field(_)
        | SymbolRef::Derivative(..)
        | SymbolRef::Across(_)
        | SymbolRef::Through(_) => Some(symbol),
        SymbolRef::Port(port) => Some(SymbolRef::Port(
            sources
                .get(&port.erase())
                .copied()
                .unwrap_or(port.erase())
                .downcast()
                .expect("Port source"),
        )),
        _ => None,
    }
}

/// Ordering is local to unknown incidence; Semantic IDs need no numeric ordering API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Coordinate(ScalarSymbolCoordinate);
impl std::ops::Deref for Coordinate {
    type Target = ScalarSymbolCoordinate;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Coordinate {
    fn with_symbol(&self, symbol: SymbolRef) -> Self {
        Self(self.0.with_symbol(symbol))
    }
    pub(super) fn variable(&self) -> Variable {
        match self.symbol() {
            SymbolRef::Field(id) => Variable::Field(id.erase()),
            SymbolRef::Derivative(id, order) => Variable::Derivative(id.erase(), order),
            SymbolRef::Next(id) => Variable::NextField(id.erase()),
            SymbolRef::Port(id) => Variable::Port(id.erase()),
            SymbolRef::Across(id) => Variable::Physical(PhysicalUnknown::Across(id)),
            SymbolRef::Through(id) => Variable::Physical(PhysicalUnknown::Through(id)),
            _ => unreachable!("incidence contains only numerical unknowns"),
        }
    }
}
impl Ord for Coordinate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.variable(), self.component_index(), self.is_imaginary()).cmp(&(
            other.variable(),
            other.component_index(),
            other.is_imaginary(),
        ))
    }
}
impl PartialOrd for Coordinate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
