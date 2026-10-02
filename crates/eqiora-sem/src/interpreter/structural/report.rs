//! Informational incidence projections; numerical admission remains separate.
use super::*;

/// One scalar equation occurrence in a continuous structural analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EquationIncidence {
    owner: RawId,
    ordinal: usize,
    coordinates: BTreeSet<Variable>,
}

impl EquationIncidence {
    /// Exact Relation or conserving Connection occurrence.
    #[must_use]
    pub const fn owner(&self) -> RawId {
        self.owner
    }
    /// Zero-based equation ordinal within the owning Relation or junction.
    #[must_use]
    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }
    /// Referenced coordinates, retaining distinct value and derivative slots.
    pub fn coordinates(&self) -> impl Iterator<Item = SymbolRef> + '_ {
        self.coordinates.iter().copied().map(symbol)
    }
}

/// A necessary incidence matching, not a numerical rank certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncidenceMatching {
    pub(super) variables: BTreeMap<Variable, usize>,
    pub(super) rows: Vec<Vec<usize>>,
    pub(super) matched: Vec<Option<usize>>,
    over: IncidenceBlock,
    under: IncidenceBlock,
}

impl IncidenceMatching {
    /// Candidate coordinates in deterministic column order.
    pub fn coordinates(&self) -> impl Iterator<Item = SymbolRef> + '_ {
        self.variables.keys().copied().map(symbol)
    }
    /// Maximum incidence matching cardinality. Cancellation is not analyzed.
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
    pub fn overdetermined_coordinates(&self) -> impl Iterator<Item = SymbolRef> + '_ {
        self.variables
            .iter()
            .filter(|(_, column)| self.over.1.contains(column))
            .map(|(coordinate, _)| symbol(*coordinate))
    }
    /// Coordinates participating in the coarse deficient block.
    pub fn underdetermined_coordinates(&self) -> impl Iterator<Item = SymbolRef> + '_ {
        self.variables
            .iter()
            .filter(|(_, column)| self.under.1.contains(column))
            .map(|(coordinate, _)| symbol(*coordinate))
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
    /// Continuous Relation and conserving junction rows with exact owners.
    #[must_use]
    pub fn equations(&self) -> &[EquationIncidence] {
        &self.equations
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
        .differential_fields
        .union(&plan.algebraic_fields)
        .copied()
        .map(Variable::Field)
        .chain(plan.continuous_ports.iter().copied().map(Variable::Port))
        .chain(
            plan.physical_unknowns
                .iter()
                .copied()
                .map(Variable::Physical),
        )
        .collect::<BTreeSet<_>>();
    let mut equations = Vec::new();
    for &owner in &plan.continuous_relations {
        let Some(KernelNode::Relation(relation)) = program.node(owner) else {
            return Err(execution_error("validated Relation is unavailable", 0.0));
        };
        for (ordinal, (left, right)) in relation.equation_sides().enumerate() {
            equations.push(EquationIncidence {
                owner,
                ordinal,
                coordinates: incidence::variables(
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
                    owner: junction.connection().erase(),
                    ordinal,
                    coordinates: incidence::variables(
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
        .copied()
        .map(|variable| match variable {
            Variable::Field(field) if plan.differential_fields.contains(&field) => {
                Variable::Derivative(field)
            }
            other => other,
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
    variables: BTreeSet<Variable>,
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
                .copied()
                .map(|coordinate| match coordinate {
                    Variable::Derivative(field) if merge_rates => Variable::Field(field),
                    other => other,
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

fn symbol(variable: Variable) -> SymbolRef {
    match variable {
        Variable::Field(id) => SymbolRef::Field(id.downcast().expect("Field coordinate")),
        Variable::Derivative(id) => SymbolRef::Derivative(id.downcast().expect("derivative Field")),
        Variable::NextField(id) => SymbolRef::Next(id.downcast().expect("next Field")),
        Variable::Port(id) => SymbolRef::Port(id.downcast().expect("signal Port")),
        Variable::Physical(PhysicalUnknown::Across(id)) => SymbolRef::Across(id),
        Variable::Physical(PhysicalUnknown::Through(id)) => SymbolRef::Through(id),
    }
}
