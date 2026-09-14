//! Affine numerical realization of one canonical scalar physical subsystem.

mod gauge;

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, GraphPath, Id};
use eqiora_ir::{BoundAffineFailure, BoundAffineScalarIr, ScalarOperatorIr};
use eqiora_schema::kernel::SymbolRef;
use eqiora_sem::{ComposedResidualSystem, KernelProgram, PhysicalUnknown};
use eqiora_solver::{
    CanonicalCsrSystemView, CompleteCsrStorage, FixedOrderInnerProduct, LinearOperatorProperties,
    LinearSolveRequest, ReplicatedLinearExecution, SERIAL_LINEAR_EXECUTION, SolveReport,
};

/// One canonical flat scalar physical subsystem admitted as `A w = b`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarPhysicalAffineProblem {
    composed: ComposedResidualSystem,
    parameter_values: Vec<f64>,
    time: Option<f64>,
    canonical_system: CanonicalCsrSystemView,
}

impl ScalarPhysicalAffineProblem {
    /// Canonical semantic residual system from which this problem was lowered.
    #[must_use]
    pub const fn composed_system(&self) -> &ComposedResidualSystem {
        &self.composed
    }

    /// Parameter values in [`ComposedResidualSystem::parameters`] order.
    #[must_use]
    pub fn parameter_values(&self) -> &[f64] {
        &self.parameter_values
    }

    /// Model time when the composed residual reads time.
    #[must_use]
    pub const fn time(&self) -> Option<f64> {
        self.time
    }

    /// Sole captured complete CSR action and right-hand side.
    #[must_use]
    pub const fn canonical_system(&self) -> &CanonicalCsrSystemView {
        &self.canonical_system
    }

    /// Re-evaluate the original semantic DAGs at one candidate vector.
    ///
    /// # Errors
    /// Returns the reference evaluator's structured input or finite-arithmetic
    /// diagnostic.
    pub fn reference_residuals(&self, values: &[f64]) -> Result<Vec<f64>, Diagnostic> {
        self.composed
            .evaluate_reference(values, &self.parameter_values, self.time)
    }
}

/// Solution accepted by both the captured linear action and original DAGs.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarPhysicalAffineSolution {
    unknowns: Vec<PhysicalUnknown>,
    values: Vec<f64>,
    report: SolveReport,
    reference_residual_norm: f64,
}

impl ScalarPhysicalAffineSolution {
    /// Canonical unknown order matching [`Self::values`].
    #[must_use]
    pub fn unknowns(&self) -> &[PhysicalUnknown] {
        &self.unknowns
    }

    /// Accepted scalar values in canonical unknown order.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Value for one canonical physical unknown.
    #[must_use]
    pub fn value(&self, unknown: PhysicalUnknown) -> Option<f64> {
        self.unknowns
            .iter()
            .position(|candidate| *candidate == unknown)
            .map(|index| self.values[index])
    }

    /// Backend and solver evidence from independent linear-action acceptance.
    #[must_use]
    pub const fn report(&self) -> &SolveReport {
        &self.report
    }

    /// Euclidean residual norm recomputed through the semantic DAG evaluator.
    #[must_use]
    pub const fn reference_residual_norm(&self) -> f64 {
        self.reference_residual_norm
    }
}

/// Compose and structurally admit one flat scalar physical subsystem as an
/// affine general linear system.
///
/// Parameters are read from the exact `KernelProgram` snapshot after
/// composition. The selected columns are the canonical Across/Through order;
/// relation roots precede generated junction roots. No numerical probing,
/// nonlinear iteration, or spatial realization is performed.
///
/// # Errors
/// Returns a semantic composition diagnostic, `EQ0701` for unsupported scalar
/// Operator IR, or `EQ0807` when the roots are not a nonempty square affine
/// system or a known input cannot be bound exactly.
pub fn lower_scalar_physical_affine(
    program: &KernelProgram,
    connection: Id<kinds::Connection>,
    time: Option<f64>,
) -> Result<ScalarPhysicalAffineProblem, Diagnostic> {
    let composed = program.compose_scalar_physical_subsystem(connection)?;
    if composed
        .unknown_types()
        .iter()
        .chain(composed.parameter_types())
        .any(|value_type| {
            value_type.scalar_domain() != eqiora_core::ScalarDomain::Real
                || !value_type.shape().is_scalar()
        })
    {
        return Err(Diagnostic::error(
            codes::NOT_IMPLEMENTED,
            "affine physical execution requires real scalar inputs",
        ));
    }
    validate_time(&composed, time)?;

    let mut selected_symbols = Vec::new();
    selected_symbols
        .try_reserve_exact(composed.unknowns().len())
        .map_err(|_| affine_error("could not reserve canonical physical unknown symbols"))?;
    selected_symbols.extend(composed.unknowns().iter().copied().map(unknown_symbol));

    let mut parameter_values = Vec::new();
    parameter_values
        .try_reserve_exact(composed.parameters().len())
        .map_err(|_| affine_error("could not reserve scalar physical parameter values"))?;
    let binding_count = composed
        .parameters()
        .len()
        .checked_add(usize::from(composed.uses_time()))
        .ok_or_else(|| affine_error("scalar physical binding count overflowed"))?;
    let mut bindings = Vec::new();
    bindings
        .try_reserve_exact(binding_count)
        .map_err(|_| affine_error("could not reserve scalar physical bindings"))?;
    for parameter in composed.parameters() {
        let value = program.value(parameter.erase()).ok_or_else(|| {
            affine_error(format!(
                "canonical physical Parameter {parameter} has no snapshot value"
            ))
        })?;
        parameter_values.push(value.value());
        bindings.push((SymbolRef::Parameter(*parameter), value.value()));
    }
    if let Some(value) = time {
        bindings.push((SymbolRef::Time, value));
    }

    let residual_count = composed
        .relations()
        .iter()
        .map(|residual| residual.dag().roots().len())
        .chain(
            composed
                .junctions()
                .iter()
                .map(|residual| residual.dag().roots().len()),
        )
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| affine_error("scalar physical residual count overflowed"))?;
    if residual_count == 0 || residual_count != selected_symbols.len() {
        return Err(affine_error(format!(
            "scalar physical affine realization requires a nonempty square system, found {residual_count} roots and {} unknowns",
            selected_symbols.len(),
        )));
    }

    let mut storage = AffineCsrStorage::new(residual_count, selected_symbols.len())?;
    for residual in composed.relations() {
        append_affine_group(&mut storage, residual.dag(), &selected_symbols, &bindings)?;
    }
    for residual in composed.junctions() {
        append_affine_group(&mut storage, residual.dag(), &selected_symbols, &bindings)?;
    }
    storage.finish()?;
    gauge::reject_unreferenced_across_groups(program, composed.unknowns(), &storage)?;
    let canonical_system = CanonicalCsrSystemView::new(&storage, LinearOperatorProperties::General)
        .map_err(|diagnostic| {
            affine_error(format!(
                "scalar physical affine system is not canonical: {}",
                diagnostic.message()
            ))
        })?;

    Ok(ScalarPhysicalAffineProblem {
        composed,
        parameter_values,
        time,
        canonical_system,
    })
}

/// Solve from one explicit finite initial guess and reaccept the result through
/// the original semantic residual DAGs.
///
/// The initial guess is a numerical realization input in canonical unknown
/// order; it does not alter Semantic Model meaning or the captured `A w = b`
/// identity.
///
/// # Errors
/// Returns `EQ0802` for an initial-guess shape/non-finite mismatch, the
/// selected backend's capability/numerical diagnostic, or semantic-DAG
/// residual rejection.
pub fn solve_scalar_physical_affine(
    problem: &ScalarPhysicalAffineProblem,
    initial_guess: &[f64],
    solver: LinearSolveRequest<'_>,
) -> Result<ScalarPhysicalAffineSolution, Diagnostic> {
    let linear_problem = problem
        .canonical_system
        .linear_problem()?
        .with_initial_guess(initial_guess)?;
    let accepted = solver.solve(&linear_problem)?;
    let reference_residuals = problem.reference_residuals(accepted.values())?;
    let reference_residual_norm = SERIAL_LINEAR_EXECUTION
        .inner_product(FixedOrderInnerProduct::new(
            &reference_residuals,
            &reference_residuals,
        )?)?
        .sqrt();
    if !reference_residual_norm.is_finite()
        || reference_residual_norm > accepted.report().residual_target()
    {
        return Err(solve_error(format!(
            "linear solution passed CSR acceptance but semantic residual {reference_residual_norm:e} exceeds target {:e}",
            accepted.report().residual_target(),
        )));
    }
    let unknowns = problem.composed.unknowns().to_vec();
    let (values, report) = accepted.into_parts();
    Ok(ScalarPhysicalAffineSolution {
        unknowns,
        values,
        report,
        reference_residual_norm,
    })
}

fn unknown_symbol(unknown: PhysicalUnknown) -> SymbolRef {
    match unknown {
        PhysicalUnknown::Across(port) => SymbolRef::Across(port),
        PhysicalUnknown::Through(port) => SymbolRef::Through(port),
    }
}

fn validate_time(composed: &ComposedResidualSystem, time: Option<f64>) -> Result<(), Diagnostic> {
    match (composed.uses_time(), time) {
        (true, None) => Err(affine_error(
            "time-dependent scalar physical affine realization requires model time",
        )),
        (false, Some(_)) => Err(affine_error(
            "time-independent scalar physical affine realization does not accept model time",
        )),
        (true, Some(value)) if !value.is_finite() => Err(affine_error(
            "scalar physical affine realization requires finite model time",
        )),
        (true, Some(_)) | (false, None) => Ok(()),
    }
}

pub(crate) fn append_affine_group(
    storage: &mut AffineCsrStorage,
    dag: &eqiora_schema::kernel::ExprDag,
    selected_symbols: &[SymbolRef],
    bindings: &[(SymbolRef, f64)],
) -> Result<(), Diagnostic> {
    let ir = ScalarOperatorIr::lower(dag)?;
    let affine = ir
        .bind_affine(selected_symbols, bindings)
        .map_err(bound_affine_error)?;
    storage.append(&affine)
}

fn bound_affine_error(failure: BoundAffineFailure) -> Diagnostic {
    affine_error(format!(
        "scalar physical residual is not bound affine: {failure:?}"
    ))
}

#[derive(Debug)]
pub(crate) struct AffineCsrStorage {
    rows: usize,
    columns: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<f64>,
    right_hand_side: Vec<f64>,
}

impl AffineCsrStorage {
    pub(crate) fn new(rows: usize, columns: usize) -> Result<Self, Diagnostic> {
        let offset_count = rows
            .checked_add(1)
            .ok_or_else(|| affine_error("scalar physical CSR row-offset count overflowed"))?;
        let mut row_offsets = Vec::new();
        row_offsets
            .try_reserve_exact(offset_count)
            .map_err(|_| affine_error("could not reserve scalar physical CSR row offsets"))?;
        row_offsets.push(0);
        let mut right_hand_side = Vec::new();
        right_hand_side
            .try_reserve_exact(rows)
            .map_err(|_| affine_error("could not reserve scalar physical right-hand side"))?;
        Ok(Self {
            rows,
            columns,
            row_offsets,
            column_indices: Vec::new(),
            values: Vec::new(),
            right_hand_side,
        })
    }

    fn append(&mut self, affine: &BoundAffineScalarIr) -> Result<(), Diagnostic> {
        if affine.selected_symbol_count() != self.columns {
            return Err(affine_error(
                "bound affine group changed the canonical unknown count",
            ));
        }
        let nonzero_count = affine
            .coefficients()
            .iter()
            .filter(|value| **value != 0.0)
            .count();
        self.column_indices
            .try_reserve(nonzero_count)
            .map_err(|_| affine_error("could not reserve scalar physical CSR columns"))?;
        self.values
            .try_reserve(nonzero_count)
            .map_err(|_| affine_error("could not reserve scalar physical CSR values"))?;
        for row in 0..affine.residual_count() {
            let coefficients = affine
                .coefficient_row(row)
                .ok_or_else(|| affine_error("bound affine coefficient row is unavailable"))?;
            for (column, value) in coefficients.iter().copied().enumerate() {
                if value != 0.0 {
                    self.column_indices.push(column);
                    self.values.push(value);
                }
            }
            self.row_offsets.push(self.values.len());
            self.right_hand_side.push(-affine.offsets()[row]);
        }
        Ok(())
    }

    pub(crate) fn finish(&self) -> Result<(), Diagnostic> {
        if self.right_hand_side.len() != self.rows || self.row_offsets.len() != self.rows + 1 {
            return Err(affine_error(format!(
                "scalar physical CSR lowering produced {} rows for an admitted {}-row system",
                self.right_hand_side.len(),
                self.rows,
            )));
        }
        Ok(())
    }
}

impl CompleteCsrStorage for AffineCsrStorage {
    fn rows(&self) -> usize {
        self.rows
    }

    fn columns(&self) -> usize {
        self.columns
    }

    fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }

    fn values(&self) -> &[f64] {
        &self.values
    }

    fn right_hand_side(&self) -> &[f64] {
        &self.right_hand_side
    }
}

fn affine_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message).with_graph_path(GraphPath::new([
        "realization",
        "scalar-physical",
        "affine",
    ]))
}

fn solve_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message).with_graph_path(GraphPath::new([
        "evidence",
        "scalar-physical",
        "reference-residual",
    ]))
}

#[cfg(test)]
mod tests {
    use eqiora_core::{DimExponents, OntologyId};
    use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
    use eqiora_schema::kernel::{
        ActivationDef, ConnectionDef, ConnectionSemantics, DomainDef, ExprDagBuilder, KernelNode,
        ParameterDef, PortDef, RelationDef,
    };
    use eqiora_schema::{Model, ModelView};

    use super::*;

    struct Fixture {
        program: KernelProgram,
        connection: Id<kinds::Connection>,
        ports: [Id<kinds::Port>; 2],
        supply: Id<kinds::Parameter>,
        load: Id<kinds::Parameter>,
    }

    fn fixture(scalar_domain: eqiora_core::ScalarDomain) -> Fixture {
        let voltage =
            DimExponents::from_integers([1, 2, -3, -1, 0, 0, 0]).expect("bounded dimension");
        let current =
            DimExponents::from_integers([0, 0, 0, 1, 0, 0, 0]).expect("bounded dimension");
        let resistance =
            DimExponents::from_integers([1, 2, -3, -2, 0, 0, 0]).expect("bounded dimension");
        let domain = Id::new();
        let supply = Id::new();
        let load = Id::new();
        let connection = Id::new();
        let activation = Id::new();
        let mut ports = [Id::new(), Id::new()];
        ports.sort_by_key(|port: &Id<kinds::Port>| port.erase());
        let mut relations = [Id::new(), Id::new()];
        relations.sort_by_key(|relation: &Id<kinds::Relation>| relation.erase());

        let mut source_dag = ExprDagBuilder::new();
        let source_across = source_dag.symbol(SymbolRef::Across(ports[0])).unwrap();
        let supply_value = source_dag.symbol(SymbolRef::Parameter(supply)).unwrap();
        let source_root = source_dag.sub(source_across, supply_value).unwrap();

        let mut load_dag = ExprDagBuilder::new();
        let load_through = load_dag.symbol(SymbolRef::Through(ports[1])).unwrap();
        let resistance_value = load_dag.symbol(SymbolRef::Parameter(load)).unwrap();
        let supply_value = load_dag.symbol(SymbolRef::Parameter(supply)).unwrap();
        let voltage_drop = load_dag.mul(resistance_value, load_through).unwrap();
        let load_root = load_dag.sub(voltage_drop, supply_value).unwrap();

        let nodes = vec![
            KernelNode::from(
                DomainDef::scalar_physical(
                    domain,
                    eqiora_core::ValueType::scalar(scalar_domain, voltage)
                        .expect("fixture uses a numeric scalar domain"),
                    eqiora_core::ValueType::scalar(scalar_domain, current)
                        .expect("fixture uses a numeric scalar domain"),
                )
                .unwrap(),
            ),
            KernelNode::from(ParameterDef::new(
                supply,
                eqiora_core::ValueLiteral::from_real(
                    eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, voltage)
                        .expect("numeric scalar type"),
                    12.0,
                )
                .unwrap(),
            )),
            KernelNode::from(ParameterDef::new(
                load,
                eqiora_core::ValueLiteral::from_real(
                    eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, resistance)
                        .expect("numeric scalar type"),
                    2.0,
                )
                .unwrap(),
            )),
            KernelNode::from(PortDef::scalar_physical(ports[0], domain)),
            KernelNode::from(PortDef::scalar_physical(ports[1], domain)),
            KernelNode::from(
                RelationDef::new(
                    relations[0],
                    {
                        let equation_zero_0 = source_dag
                            .constant(
                                eqiora_core::ValueLiteral::from_real(
                                    eqiora_core::ValueType::scalar(scalar_domain, voltage)
                                        .expect("fixture uses a numeric scalar domain"),
                                    0.0,
                                )
                                .unwrap(),
                            )
                            .unwrap();
                        source_dag.finish([source_root, equation_zero_0])
                    }
                    .unwrap(),
                )
                .unwrap(),
            ),
            KernelNode::from(
                RelationDef::new(
                    relations[1],
                    {
                        let equation_zero_0 = load_dag
                            .constant(
                                eqiora_core::ValueLiteral::from_real(
                                    eqiora_core::ValueType::scalar(scalar_domain, voltage)
                                        .expect("fixture uses a numeric scalar domain"),
                                    0.0,
                                )
                                .unwrap(),
                            )
                            .unwrap();
                        load_dag.finish([load_root, equation_zero_0])
                    }
                    .unwrap(),
                )
                .unwrap(),
            ),
            KernelNode::from(ActivationDef::continuous(activation)),
            KernelNode::from(ConnectionDef::new(
                connection,
                ConnectionSemantics::Conserving,
            )),
        ];
        let model = OntologyId::<Model>::new();
        let view = ModelView::new(model, nodes.iter().map(KernelNode::id), []).unwrap();
        let mut transaction = Transaction::new("two-port affine physical fixture");
        for node in nodes {
            transaction.push(Op::DefineKernelNode { node });
        }
        for (&port, &relation) in ports.iter().zip(&relations) {
            transaction
                .push(Op::Connect {
                    from: relation.erase(),
                    to: port.erase(),
                    edge: EdgeKind::HasPort,
                })
                .push(Op::Connect {
                    from: relation.erase(),
                    to: port.erase(),
                    edge: EdgeKind::DependsOn,
                })
                .push(Op::Connect {
                    from: activation.erase(),
                    to: relation.erase(),
                    edge: EdgeKind::Activates,
                })
                .push(Op::Connect {
                    from: connection.erase(),
                    to: port.erase(),
                    edge: EdgeKind::Connects,
                });
        }
        transaction
            .push(Op::Connect {
                from: relations[0].erase(),
                to: supply.erase(),
                edge: EdgeKind::DependsOn,
            })
            .push(Op::Connect {
                from: relations[1].erase(),
                to: supply.erase(),
                edge: EdgeKind::DependsOn,
            })
            .push(Op::Connect {
                from: relations[1].erase(),
                to: load.erase(),
                edge: EdgeKind::DependsOn,
            })
            .push(Op::DefineOntologyView { view: view.into() });
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        Fixture {
            program: KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
            connection,
            ports,
            supply,
            load,
        }
    }

    #[test]
    fn physical_affine_lowering_binds_parameters_and_reuses_reference_dags() {
        let fixture = fixture(eqiora_core::ScalarDomain::Real);
        let problem =
            lower_scalar_physical_affine(&fixture.program, fixture.connection, None).unwrap();
        assert_eq!(problem.canonical_system().rows(), 4);
        assert_eq!(problem.canonical_system().columns(), 4);
        let parameter_values = problem
            .composed_system()
            .parameters()
            .iter()
            .copied()
            .zip(problem.parameter_values().iter().copied())
            .collect::<std::collections::HashMap<_, _>>();
        assert_eq!(parameter_values[&fixture.supply], 12.0);
        assert_eq!(parameter_values[&fixture.load], 2.0);
        assert_eq!(
            problem.composed_system().unknowns(),
            &[
                PhysicalUnknown::Across(fixture.ports[0]),
                PhysicalUnknown::Through(fixture.ports[0]),
                PhysicalUnknown::Across(fixture.ports[1]),
                PhysicalUnknown::Through(fixture.ports[1]),
            ]
        );
        assert_eq!(
            problem
                .reference_residuals(&[12.0, -6.0, 12.0, 6.0])
                .unwrap(),
            [0.0; 4]
        );
    }

    #[test]
    fn physical_affine_lowering_rejects_superfluous_time_before_ir_work() {
        let fixture = fixture(eqiora_core::ScalarDomain::Real);
        let error = lower_scalar_physical_affine(&fixture.program, fixture.connection, Some(0.0))
            .unwrap_err();
        assert_eq!(error.code(), codes::INVALID_REALIZATION);
        assert!(error.message().contains("does not accept model time"));
    }

    #[test]
    fn physical_affine_lowering_rejects_complex_inputs_before_numeric_binding() {
        let fixture = fixture(eqiora_core::ScalarDomain::Complex);
        let error =
            lower_scalar_physical_affine(&fixture.program, fixture.connection, None).unwrap_err();
        assert_eq!(error.code(), codes::NOT_IMPLEMENTED);
        assert!(error.message().contains("real scalar inputs"));
    }
}
