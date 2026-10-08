//! Authored elastic stationarity bound to the admitted strong continuum.
use crate::canonical_boundary::PhysicalBoundaryDisposition;
use crate::canonical_elasticity::IsotropicElasticityContinuum;
use crate::form_compiler::{authored_polynomial, scalar::typed_relation, vocabulary::*};
use eqiora_compiler::AuthoredFormulationProjection;
use eqiora_core::{Diagnostic, RawId};
use eqiora_schema::kernel::{ExprNode, SymbolRef};
use eqiora_sem::KernelProgram;

pub(crate) fn derive(
    program: &KernelProgram,
    model: &IsotropicElasticityContinuum<2>,
    authored: Option<&AuthoredFormulationProjection>,
) -> Result<Option<crate::form_compiler::PrimalFormDescription>, Diagnostic> {
    let relation = model.equilibrium_relation();
    let typed = typed_relation(program, relation)?;
    let volume = super::execution::recognize_volume(&typed, relation, model.displacement())?;
    let mut boundaries = Vec::new();
    let mut tractions = Vec::new();
    for (_, entry) in model.boundary_inventory().entries() {
        let discharge = match entry.disposition() {
            PhysicalBoundaryDisposition::TraceZero => BoundaryDischarge::ZeroTestTrace,
            PhysicalBoundaryDisposition::FluxZero => BoundaryDischarge::ZeroFlux,
            PhysicalBoundaryDisposition::Prescribed(law)
                if law.quantity() == crate::canonical_boundary::PhysicalBoundaryQuantity::Trace =>
            {
                BoundaryDischarge::ZeroTestTrace
            }
            PhysicalBoundaryDisposition::Prescribed(law)
                if law.quantity() == crate::canonical_boundary::PhysicalBoundaryQuantity::Flux
                    && model.tractions().contains_key(&entry.boundary()) =>
            {
                BoundaryDischarge::PrescribedFlux
            }
            _ => return Ok(None),
        };
        let bindings = model
            .boundary_relations()
            .iter()
            .filter(|binding| binding.boundary() == entry.boundary())
            .collect::<Vec<_>>();
        let [binding] = bindings.as_slice() else {
            return Ok(None);
        };
        let boundary = typed_relation(program, binding.relation())?;
        let dag = boundary.expression();
        let [root] = dag.roots() else {
            return Ok(None);
        };
        let view =
            crate::additive_residual::AdditiveResidualView::derive(dag, *root, binding.relation())?;
        // Canonical continuum admission already proves the complete stress flux.
        // Only an exact homogeneous residual can discharge it without surface work.
        if discharge == BoundaryDischarge::ZeroFlux && view.leaves().len() != 1 {
            return Ok(None);
        }
        let operators = view.leaves().iter().filter(|leaf| match (discharge,dag.node(leaf.value())) {
            (BoundaryDischarge::ZeroTestTrace,Some(ExprNode::Trace(value))) => matches!(dag.node(*value),Some(ExprNode::Symbol(SymbolRef::Field(id))) if id.erase() == model.displacement()),
            (BoundaryDischarge::ZeroFlux | BoundaryDischarge::PrescribedFlux,Some(ExprNode::NormalComponent(_))) => true,
            _ => false,
        }).collect::<Vec<_>>();
        let [operator] = operators.as_slice() else {
            return Ok(None);
        };
        if discharge == BoundaryDischarge::PrescribedFlux {
            let data = view
                .leaves()
                .iter()
                .filter(|leaf| leaf.value() != operator.value())
                .collect::<Vec<_>>();
            let [datum] = data.as_slice() else {
                return Ok(None);
            };
            tractions.push(authored_polynomial::ElasticTractionTerm {
                boundary: entry.boundary(),
                typed: boundary.clone(),
                datum: datum.value(),
                negative: operator.sign().is_opposite(datum.sign()),
            });
        }
        boundaries.push(BoundarySource {
            domain: entry.boundary(),
            relation: binding.relation(),
            operator_node: operator.value(),
            discharge,
        });
    }
    let source = PrimalGalerkinSource {
        domain: model.domain(),
        unknown: model.displacement(),
        volume_relation: relation,
        root: volume.root,
        divergence: volume.divergence,
        divergence_sign: WeakSign::Positive,
        values: &[super::super::vocabulary::PrimalValueTerm {
            source_node: volume.load_gradient,
            sign: WeakSign::Positive,
            trial_dependent: false,
        }],
        conjugate_test: false,
        boundaries: &boundaries,
    };
    let correspondence = PrimalGalerkinCorrespondence::derive(source);
    correspondence
        .replay(source)
        .map_err(|message| reject(relation, message))?;
    if let Some(authored) = authored {
        if !correspondence.replay_authored(authored, program, 2)? {
            return Err(reject(
                relation,
                "elastic stationarity requires a live first energy variation",
            ));
        }
        if !authored_polynomial::matches_elastic_variation(
            authored,
            program,
            &typed,
            volume.stress,
            volume.load_gradient,
            &tractions,
        ) {
            return Err(reject(
                relation,
                "functional variation differs from the admitted elastic strong-law weak residual",
            ));
        }
    }
    Ok(Some((
        correspondence.formulation.kind,
        correspondence.formulation.boundary_treatment.id(),
        correspondence
            .formulation
            .rules
            .into_iter()
            .map(FormulationRule::id)
            .collect(),
    )))
}

fn reject(relation: RawId, message: &str) -> Diagnostic {
    crate::canonical::lowering_error(relation, message)
}
