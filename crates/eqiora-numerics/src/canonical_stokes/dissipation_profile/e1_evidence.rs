//! Sealed-input admission and the `cfg(test)` evidence revalidation seam.
//!
//! Nothing here is a second production constructor or a public surface. The
//! admission path builds exactly one private binding from the byte-identified
//! sealed input; the mutation seam forks from that already admitted positive
//! and re-runs the product predicate that owns the substituted association
//! member. No oracle, expected value, or tolerance is authored here.

use std::num::NonZeroUsize;

use eqiora_artifact::{GeometryDefinitionV1, SimplicialMeshEnvelopeV1};
use eqiora_core::{Diagnostic, DimExponents, RawId};
use eqiora_graph::{GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::ModelView;
use eqiora_schema::kernel::{
    BoundarySide, DomainDef, DomainKind, ExprNode, GeometryDigest, KernelNode,
};
use eqiora_sem::KernelProgram;
use eqiora_solver::{
    LinearSolveRequest, LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
    ReductionPolicy, SolverPlan,
};

use super::e1_sealed_input::{SealedE1Input, e1_stokes_dissipation_sealed_inputs_v1};
use super::topology_content::{chordal_body_polygon_area_m2, chordal_geometry};
use super::{
    StokesDissipationGeometryModelBinding2d, StokesDissipationProfileGeometry2d,
    StokesDissipationTopology2d, StokesDissipationTopologyRole2d, invalid,
};
use crate::canonical_stokes::support::{is_field, relation_expression, relations_on, unique_root};

const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");

/// Exact sealed source labels mapped onto the contract-owned outer roles.
const OUTER_ROLE_FOR_SIDE: [(usize, BoundarySide, &str); 4] = [
    (0, BoundarySide::Lower, "outer_x_lower"),
    (0, BoundarySide::Upper, "outer_x_upper"),
    (1, BoundarySide::Lower, "outer_y_lower"),
    (1, BoundarySide::Upper, "outer_y_upper"),
];

/// One exact substitution forked from the already admitted E1 positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::canonical_stokes) enum E1ProfileTopologyEvidenceMutation2d {
    CircleAlias,
    DesignSwap,
    FakeRefinement,
    PolygonAreaAuthority,
    GeometryRegionSwap,
    MeshCorrespondenceSwap,
    FacetRoleSwap,
    FacetLabelMissing,
    FacetLabelDuplicated,
    FacetLabelAdded,
    FacetEndpointOutOfRange,
    CorrespondenceIndexSwap,
    AngleOrderSwap,
    CellConnectivityDuplicate,
    CellEndpointOutOfRange,
}

/// The association member whose product predicate rejected a mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::canonical_stokes) enum E1ProfileTopologyRejection2d {
    ProfileIdentity,
    DesignAssociation,
    TopologyRole,
    AnalyticAreaAuthority,
    GeometryModelAssociation,
    MeshCorrespondenceAssociation,
    FacetRole,
    TopologyIndex,
}

impl StokesDissipationGeometryModelBinding2d {
    /// Admit the exact sealed E1 positive for one content-bound topology role.
    pub(in crate::canonical_stokes) fn from_e1_sealed_inputs_v1(
        bytes: &[u8],
        role: StokesDissipationTopologyRole2d,
    ) -> Result<Self, Diagnostic> {
        let sealed = SealedE1Input::admit(bytes)?;
        let area_radius_m = sealed.area_radius_m()?;
        let [a2, a4] = sealed.design_coefficients();
        let topology =
            StokesDissipationTopology2d::admit(sealed.topology_source(role)?, area_radius_m)?;
        let cartesian = compile_scaffold(&sealed)?;
        let [area_radius_parameter, a2_parameter, a4_parameter] =
            design_parameters(&cartesian, area_radius_m, [a2, a4])?;
        let profile = StokesDissipationProfileGeometry2d::new(
            area_radius_parameter,
            a2_parameter,
            a4_parameter,
            area_radius_m,
            a2,
            a4,
        )?;
        Self::new(
            profile,
            &topology,
            reference_harmonic_solver(),
            |geometry| geometry_program(&cartesian, geometry),
            Some(sealed.sealed_input_sha256()),
        )
    }

    /// Verified identity of the sealed input this binding was admitted from.
    pub(in crate::canonical_stokes) fn sealed_input_sha256(&self) -> [u8; 32] {
        self.sealed_input
            .expect("a sealed-input binding retains its verified input identity")
    }

    /// Complete one-way replay of the admitted profile/topology association.
    pub(in crate::canonical_stokes) fn revalidate_e1_profile_topology(
        &self,
    ) -> Result<(), Diagnostic> {
        self.revalidate()
    }

    /// Fork one exact substitution from this admitted positive.
    pub(in crate::canonical_stokes) fn revalidate_e1_evidence_mutant(
        &self,
        mutation: E1ProfileTopologyEvidenceMutation2d,
    ) -> Result<(), E1ProfileTopologyRejection2d> {
        use E1ProfileTopologyEvidenceMutation2d as Mutation;
        use E1ProfileTopologyRejection2d as Rejection;
        match mutation {
            Mutation::CircleAlias => {
                self.reject_substituted_design([0.0, 0.0], Rejection::ProfileIdentity)
            }
            Mutation::DesignSwap => {
                let conjugate = self
                    .sealed_input()
                    .map_err(|()| Rejection::DesignAssociation)?
                    .conjugate_design_coefficients();
                self.reject_substituted_design(conjugate, Rejection::DesignAssociation)
            }
            Mutation::FakeRefinement => self.reject_relabelled_role(),
            Mutation::PolygonAreaAuthority => self.reject_polygon_area_authority(),
            Mutation::GeometryRegionSwap => self.reject_geometry_region_swap(),
            Mutation::MeshCorrespondenceSwap => self.reject_mesh_swap(),
            Mutation::FacetRoleSwap
            | Mutation::FacetLabelMissing
            | Mutation::FacetLabelDuplicated
            | Mutation::FacetLabelAdded => self.reject_facet_content(mutation),
            Mutation::FacetEndpointOutOfRange
            | Mutation::CorrespondenceIndexSwap
            | Mutation::AngleOrderSwap
            | Mutation::CellConnectivityDuplicate
            | Mutation::CellEndpointOutOfRange => self.reject_topology_index(mutation),
        }
    }

    fn sealed_input(&self) -> Result<SealedE1Input, ()> {
        SealedE1Input::admit(e1_stokes_dissipation_sealed_inputs_v1()).map_err(|_| ())
    }

    /// Substitute the state a different design would realize, keeping the
    /// retained analytic identity, and require the association to reject.
    fn reject_substituted_design(
        &self,
        coefficients: [f64; 2],
        rejection: E1ProfileTopologyRejection2d,
    ) -> Result<(), E1ProfileTopologyRejection2d> {
        let [area_radius_parameter, a2_parameter, a4_parameter] = self.profile.parameters();
        let substituted = StokesDissipationProfileGeometry2d::new(
            area_radius_parameter,
            a2_parameter,
            a4_parameter,
            self.profile.area_radius_m(),
            coefficients[0],
            coefficients[1],
        )
        .map_err(|_| rejection)?;
        if substituted.identity() == self.profile.identity() {
            return Ok(());
        }
        let state = self
            .topology
            .harmonic_state(&substituted, &self.motion_action)
            .map_err(|_| rejection)?;
        let mutant = Self {
            state,
            ..self.clone()
        };
        reject_when(mutant.revalidate(), rejection)
    }

    fn reject_relabelled_role(&self) -> Result<(), E1ProfileTopologyRejection2d> {
        let mut source = self.topology.source.clone();
        source.role = match source.role {
            StokesDissipationTopologyRole2d::Reference => StokesDissipationTopologyRole2d::Refined,
            StokesDissipationTopologyRole2d::Refined => StokesDissipationTopologyRole2d::Reference,
        };
        reject_when(
            StokesDissipationTopology2d::admit(source, self.profile.area_radius_m()),
            E1ProfileTopologyRejection2d::TopologyRole,
        )
    }

    fn reject_polygon_area_authority(&self) -> Result<(), E1ProfileTopologyRejection2d> {
        let rejection = E1ProfileTopologyRejection2d::AnalyticAreaAuthority;
        let (absolute, relative) = self
            .sealed_input()
            .map_err(|()| rejection)?
            .analytic_area_tolerances()
            .map_err(|_| rejection)?;
        let analytic = self.profile.analytic_area_m2();
        let polygon =
            chordal_body_polygon_area_m2(self.body_vertex_ids(), self.state.coordinates());
        let floor = absolute + relative * analytic.abs().max(polygon.abs());
        if analytic - polygon > floor {
            return Err(rejection);
        }
        Ok(())
    }

    fn reject_geometry_region_swap(&self) -> Result<(), E1ProfileTopologyRejection2d> {
        let rejection = E1ProfileTopologyRejection2d::GeometryModelAssociation;
        let start_design = chordal_geometry(
            &self.profile,
            &self.topology,
            &self.topology.reference_mesh,
            self.topology.sector_count,
            self.topology.source.coordinate_tolerance_m,
        )
        .map_err(|_| rejection)?;
        let mut model = self.model.clone();
        model.geometry_source_digest = Some(start_design.canonical().digest_bytes());
        let mutant = Self {
            model,
            ..self.clone()
        };
        reject_when(mutant.revalidate(), rejection)
    }

    fn reject_mesh_swap(&self) -> Result<(), E1ProfileTopologyRejection2d> {
        let rejection = E1ProfileTopologyRejection2d::MeshCorrespondenceAssociation;
        let start_design = SimplicialMeshEnvelopeV1::from_mesh(&self.topology.reference_mesh)
            .map_err(|_| rejection)?;
        let mesh_artifact = start_design.digest().map_err(|_| rejection)?.sha256_bytes();
        let mutant = Self {
            mesh: start_design,
            mesh_artifact,
            ..self.clone()
        };
        reject_when(mutant.revalidate(), rejection)
    }

    fn reject_facet_content(
        &self,
        mutation: E1ProfileTopologyEvidenceMutation2d,
    ) -> Result<(), E1ProfileTopologyRejection2d> {
        use E1ProfileTopologyEvidenceMutation2d as Mutation;
        let rejection = E1ProfileTopologyRejection2d::FacetRole;
        let mut source = self.topology.source.clone();
        let sectors = self.topology.sector_count;
        match mutation {
            Mutation::FacetRoleSwap => {
                source.boundary_facets[0].label = source.boundary_facets[sectors].label.clone();
            }
            Mutation::FacetLabelMissing => {
                source.boundary_facets.pop();
            }
            Mutation::FacetLabelDuplicated => {
                let duplicate = source.boundary_facets[0].clone();
                source.boundary_facets.push(duplicate);
            }
            Mutation::FacetLabelAdded => {
                let mut added = source.boundary_facets[sectors].clone();
                added.label = "outer_diagonal".to_owned();
                source.boundary_facets.push(added);
            }
            _ => return Ok(()),
        }
        reject_when(
            super::topology_content::require_topology_content(
                &source,
                sectors,
                self.topology.radial_interval_count,
            ),
            rejection,
        )
    }

    fn reject_topology_index(
        &self,
        mutation: E1ProfileTopologyEvidenceMutation2d,
    ) -> Result<(), E1ProfileTopologyRejection2d> {
        use E1ProfileTopologyEvidenceMutation2d as Mutation;
        let rejection = E1ProfileTopologyRejection2d::TopologyIndex;
        let mut source = self.topology.source.clone();
        let sectors = self.topology.sector_count;
        let vertex_count = source.vertices.len();
        match mutation {
            Mutation::FacetEndpointOutOfRange => {
                source.boundary_facets[0].vertices[0] = vertex_count;
            }
            Mutation::CorrespondenceIndexSwap => {
                source.correspondence[0].body_vertex = source.correspondence[1].body_vertex;
                source.correspondence[0].body_facet = source.correspondence[1].body_facet;
            }
            Mutation::AngleOrderSwap => {
                source.ordered_body_angles.swap(0, 1);
            }
            Mutation::CellConnectivityDuplicate => {
                source.cells[0].vertices[1] = source.cells[0].vertices[0];
            }
            Mutation::CellEndpointOutOfRange => {
                source.cells[0].vertices[0] = vertex_count;
            }
            _ => return Ok(()),
        }
        reject_when(
            super::topology_content::require_topology_indices(&source, sectors),
            rejection,
        )
    }
}

fn reject_when<T>(
    outcome: Result<T, Diagnostic>,
    rejection: E1ProfileTopologyRejection2d,
) -> Result<(), E1ProfileTopologyRejection2d> {
    match outcome {
        Ok(_) => Ok(()),
        Err(_) => Err(rejection),
    }
}

fn reference_harmonic_solver() -> LinearSolveRequest<'static> {
    let plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-13,
        1.0e-15,
        NonZeroUsize::new(10_000).expect("positive iteration limit"),
    )
    .expect("valid harmonic solver plan")
    .with_preconditioner(PreconditionerPolicy::Jacobi)
    .with_reduction(ReductionPolicy::Reproducible);
    LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan)
}

/// Author the exact typed Cartesian scaffold from sealed physical inputs.
fn e1_scaffold_source(sealed: &SealedE1Input) -> Result<String, Diagnostic> {
    let area_radius_m = sealed.area_radius_m()?;
    let [a2, a4] = sealed.design_coefficients();
    let half_width = 10.0 * area_radius_m;
    Ok(format!(
        r"model stokes_e1_dissipation_profile() {{
  domain fluid = box({lower:?}, {upper:?}, {lower:?}, {upper:?});
  domain body = boundary(fluid, axis = 0, side = lower);
  domain outer_x_minus = boundary(fluid, axis = 0, side = lower);
  domain outer_x_plus = boundary(fluid, axis = 0, side = upper);
  domain outer_y_minus = boundary(fluid, axis = 1, side = lower);
  domain outer_y_plus = boundary(fluid, axis = 1, side = upper);

  variable velocity: vector<m / s, 2> on fluid in h1;
  variable pressure: kg / (m * s ^ 2) on fluid;
  variable force_potential: kg / (m * s ^ 2) on fluid;
  variable chi: m ^ 2 / s on fluid in smooth;
  parameter mu: kg / (m * s) = {viscosity:?};
  parameter speed: m / s = {speed:?};
  parameter zero_pressure: kg / (m * s ^ 2) = 0;
  parameter equal_area_radius: m = {area_radius_m:?};
  parameter second_mode: 1 = {a2:?};
  parameter fourth_mode: 1 = {a4:?};

  relation force on fluid {{ force_potential - zero_pressure = 0; }}
  relation momentum on fluid {{
    -div(
      2 * mu * symmetric_part(grad(velocity))
      - isotropic_lift(pressure)
    ) - grad(force_potential) = 0;
  }}
  relation incompressibility on fluid {{ div(velocity) = 0; }}
  relation define_chi on fluid {{ chi - speed * coordinate(0) = 0; }}

  relation body_zero on body {{ trace(velocity) = 0; }}
  relation outer_x_minus_value on outer_x_minus {{
    trace(velocity) - trace(grad(chi)) = 0;
  }}
  relation outer_x_plus_value on outer_x_plus {{
    trace(velocity) - trace(grad(chi)) = 0;
  }}
  relation outer_y_minus_value on outer_y_minus {{
    trace(velocity) - trace(grad(chi)) = 0;
  }}
  relation outer_y_plus_value on outer_y_plus {{
    trace(velocity) - trace(grad(chi)) = 0;
  }}
}}
",
        lower = -half_width,
        upper = half_width,
        viscosity = sealed.dynamic_viscosity_pa_s()?,
        speed = sealed.speed_m_per_s()?,
    ))
}

fn compile_scaffold(sealed: &SealedE1Input) -> Result<KernelProgram, Diagnostic> {
    let source = e1_scaffold_source(sealed)?;
    let mut compiled = eqiora_compiler::compile("stokes-e1-dissipation-profile.eqi", &source)
        .map_err(first_diagnostic)?;
    if compiled.len() != 1 {
        return Err(invalid("the E1 scaffold must compile to exactly one Model"));
    }
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).map_err(first_diagnostic)?;
    KernelProgram::from_snapshot(&store.snapshot(), model).map_err(first_diagnostic)
}

/// Identify the three design Parameters by exact dimension and sealed value.
fn design_parameters(
    program: &KernelProgram,
    area_radius_m: f64,
    coefficients: [f64; 2],
) -> Result<[RawId; 3], Diagnostic> {
    let mut area_radius = None;
    let mut modes = [None, None];
    for node in program.nodes() {
        let KernelNode::Parameter(definition) = node else {
            continue;
        };
        let identity = definition.id().erase();
        let Some(initial) = definition.real_scalar_value() else {
            continue;
        };
        let value = program.value(identity).unwrap_or(initial);
        if value.dim() == LENGTH && value.value() == area_radius_m {
            if area_radius.replace(identity).is_some() {
                return Err(invalid("the E1 scaffold has an ambiguous `r_A` Parameter"));
            }
        } else if value.dim() == DimExponents::DIMENSIONLESS {
            for (index, coefficient) in coefficients.into_iter().enumerate() {
                if value.value() == coefficient && modes[index].replace(identity).is_some() {
                    return Err(invalid(
                        "the E1 scaffold has an ambiguous design-coefficient Parameter",
                    ));
                }
            }
        }
    }
    match (area_radius, modes[0], modes[1]) {
        (Some(area_radius), Some(second), Some(fourth)) => Ok([area_radius, second, fourth]),
        _ => Err(invalid(
            "the E1 scaffold does not retain three exact valued design Parameters",
        )),
    }
}

/// Rebind the compiled scaffold onto the derived chordal Geometry.
fn geometry_program(
    cartesian: &KernelProgram,
    geometry: &GeometryDefinitionV1,
) -> Result<KernelProgram, Diagnostic> {
    let canonical = geometry.canonical();
    let velocity = vector_velocity_field(cartesian)?;
    let mut nodes = Vec::new();
    for node in cartesian.nodes() {
        let replacement = match node {
            KernelNode::Domain(domain) => match domain.kind() {
                DomainKind::CartesianBox { .. } => KernelNode::from(DomainDef::geometry_region(
                    domain.id(),
                    GeometryDigest::new(canonical.digest_bytes()),
                    "fluid",
                )?),
                DomainKind::CartesianBoundary { axis, side } => {
                    let entity_set = boundary_entity_set(
                        cartesian,
                        velocity,
                        domain.id().erase(),
                        *axis,
                        *side,
                    )?;
                    KernelNode::from(DomainDef::geometry_boundary(domain.id(), entity_set)?)
                }
                _ => node.clone(),
            },
            _ => node.clone(),
        };
        nodes.push(replacement);
    }
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("private Stokes E1 chordal design binding");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for node in cartesian.nodes() {
        if let Some(value) = cartesian.typed_value(node.id()) {
            transaction.push(Op::SetValue {
                target: node.id(),
                value: value.clone(),
            });
        }
    }
    for edge in cartesian.edges() {
        transaction.push(Op::Connect {
            from: edge.from(),
            to: edge.to(),
            edge: edge.kind(),
        });
    }
    transaction.push(Op::DefineOntologyView {
        view: ModelView::new(cartesian.model(), members, None)
            .map_err(|error| invalid(error.message()))?
            .into(),
    });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).map_err(first_diagnostic)?;
    KernelProgram::from_snapshot_with_geometry(&store.snapshot(), cartesian.model(), &[canonical])
        .map_err(first_diagnostic)
}

/// The exact body role owns the `trace(velocity) = 0` scaffold Relation; the
/// four remaining sides carry the contract-owned outer roles for their side.
fn boundary_entity_set(
    program: &KernelProgram,
    velocity: RawId,
    boundary: RawId,
    axis: usize,
    side: BoundarySide,
) -> Result<&'static str, Diagnostic> {
    let relations = relations_on(program, boundary);
    let [relation] = relations.as_slice() else {
        return Err(invalid(
            "the E1 scaffold requires exactly one Relation on every boundary",
        ));
    };
    let expression = &relation_expression(program, *relation)?;
    let root = unique_root(expression, *relation)?;
    if matches!(expression.node(root), Some(ExprNode::Trace { value, .. }) if is_field(expression, *value, velocity))
    {
        return Ok("body");
    }
    OUTER_ROLE_FOR_SIDE
        .into_iter()
        .find_map(|(role_axis, role_side, role)| {
            (role_axis == axis && role_side == side).then_some(role)
        })
        .ok_or_else(|| invalid("the E1 scaffold has an unknown exact outer side"))
}

fn vector_velocity_field(program: &KernelProgram) -> Result<RawId, Diagnostic> {
    let mut velocity = None;
    for node in program.nodes() {
        if let KernelNode::Field(field) = node
            && field.shape().extents().len() == 1
            && velocity.replace(field.id().erase()).is_some()
        {
            return Err(invalid("the E1 scaffold has an ambiguous velocity Field"));
        }
    }
    velocity.ok_or_else(|| invalid("the E1 scaffold has no vector velocity Field"))
}

fn first_diagnostic(diagnostics: Vec<Diagnostic>) -> Diagnostic {
    diagnostics
        .into_iter()
        .next()
        .unwrap_or_else(|| invalid("the E1 scaffold failed without a diagnostic"))
}
