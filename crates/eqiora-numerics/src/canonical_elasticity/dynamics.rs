//! Method-neutral recognition of first-order isotropic elastodynamics.

use std::collections::BTreeMap;

use eqiora_core::ValueFrame;
use eqiora_core::{Diagnostic, DimExponents, RawId, ValueShape};
use eqiora_graph::EdgeKind;
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, KernelNode, SymbolRef};
use eqiora_sem::KernelProgram;

use super::{
    ElasticityClosure, IsotropicElasticityContinuum, boundary, continuum_representation, has_edge,
    load_definition_root, lower_isotropic_stress_coefficients, lowering_error, relation_expression,
    relations_on, require_closed_elasticity_parts, require_continuous_relation, typed_relation,
    unique_box, unique_root,
};
use crate::linear_elasticity::IsotropicElasticityMaterial;
use crate::spatial_expression::{self, ScalarSpatialExpression};

const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const VELOCITY: DimExponents =
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
const PRESSURE: DimExponents =
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");

/// Exact first-order, method-neutral small-strain elastodynamic model in `D` dimensions.
///
/// The admitted volume meaning is
///
/// ```text
/// derivative(displacement) - velocity = 0
/// density * derivative(velocity) - div(stress(displacement))
///   - grad(load_potential) = 0
/// ```
///
/// The dynamic-solid projection normalizes an exact global sign reversal of
/// either residual. Model identity remains unchanged; broader algebraic
/// equivalence is deliberately not inferred here.
///
/// and every exact Cartesian side has one normalized velocity/traction
/// disposition. No mesh, mass matrix, time method, or solver is part of this
/// contract.
#[derive(Debug, Clone, PartialEq)]
pub struct IsotropicElastodynamicsCartesianModel<const D: usize> {
    continuum: IsotropicElasticityContinuum<D>,
    velocity: RawId,
    kinematic_relation: RawId,
    mass_density: ScalarSpatialExpression<f64>,
    momentum_orientation: f64,
}

impl<const D: usize> IsotropicElastodynamicsCartesianModel<D> {
    pub(crate) const fn momentum_orientation(&self) -> f64 {
        self.momentum_orientation
    }

    /// Method-neutral continuum meaning shared with static elasticity.
    #[must_use]
    pub const fn continuum(&self) -> &IsotropicElasticityContinuum<D> {
        &self.continuum
    }

    /// Velocity-valued spatial-vector Field.
    #[must_use]
    pub const fn velocity(&self) -> RawId {
        self.velocity
    }

    /// Exact Relation binding displacement rate to velocity.
    #[must_use]
    pub(crate) const fn kinematic_relation(&self) -> RawId {
        self.kinematic_relation
    }

    /// Positive, spatially constant mass density in coherent SI units.
    #[must_use]
    pub fn mass_density(&self) -> f64 {
        self.mass_density
            .constant_value()
            .expect("elastodynamic lowerer retains a constant density tape")
    }

    /// Immutable canonical density-coefficient expression.
    ///
    /// Direct and elaborated package Models retain revision-local Parameter
    /// identities; equality of coefficient values does not conflate lineage.
    #[must_use]
    pub const fn mass_density_expression(&self) -> &ScalarSpatialExpression<f64> {
        &self.mass_density
    }
}

/// Lower the exact canonical first-order 2D isotropic-elastodynamic subset.
///
/// Recognition is identity-parametric and package-neutral. Direct boundary
/// Relations and exact velocity/traction package closures normalize through
/// the same shared boundary contract. Live Ports remain explicit for a later
/// coupled Realization.
///
/// # Errors
/// Returns `EQ0703` unless the complete Model is exactly one Cartesian body,
/// one displacement/velocity/load Field triple, the canonical kinematic and
/// momentum Relations, and one complete boundary law per side.
pub fn lower_isotropic_elastodynamics_cartesian_2d(
    program: &KernelProgram,
) -> Result<IsotropicElastodynamicsCartesianModel<2>, Diagnostic> {
    lower_isotropic_elastodynamics_cartesian::<2>(program)
}

/// Lower the exact canonical first-order 3D isotropic-elastodynamic subset.
///
/// # Errors
/// Returns `EQ0703` unless the complete model has three-component kinematics,
/// the canonical isotropic momentum relation, and all six boundary sides.
#[cfg(test)]
pub fn lower_isotropic_elastodynamics_cartesian_3d(
    program: &KernelProgram,
) -> Result<IsotropicElastodynamicsCartesianModel<3>, Diagnostic> {
    lower_isotropic_elastodynamics_cartesian::<3>(program)
}

fn lower_isotropic_elastodynamics_cartesian<const D: usize>(
    program: &KernelProgram,
) -> Result<IsotropicElastodynamicsCartesianModel<D>, Diagnostic> {
    let (domain, bounds) = unique_box::<D>(program)?;
    let lowered = lower_isotropic_elastodynamics_subdomain::<D>(program, domain, bounds, None)?;
    require_closed_elasticity_parts(
        program,
        &[ElasticityClosure {
            domain,
            fields: vec![
                lowered.model.continuum().displacement(),
                lowered.model.velocity,
                lowered.model.continuum().load_potential(),
            ],
            volume_relations: vec![
                lowered.model.continuum().load_definition_relation(),
                lowered.model.kinematic_relation(),
                lowered.model.continuum().equilibrium_relation(),
            ],
            boundary_relations: lowered.model.continuum().boundary_relations(),
            boundary: &lowered.boundary,
        }],
    )?;
    Ok(lowered.model)
}

/// Domain-scoped form used by exact multiphysics network recognizers.
///
/// This deliberately does not close the entire [`KernelProgram`]. Its caller
/// must compose the returned admitted-node inventory with every other
/// recognized subdomain and then prove whole-model closure.
pub(crate) fn lower_isotropic_elastodynamics_subdomain_2d(
    program: &KernelProgram,
    domain: RawId,
    bounds: [[f64; 2]; 2],
) -> Result<LoweredIsotropicElastodynamicsSubdomain2d, Diagnostic> {
    lower_isotropic_elastodynamics_subdomain::<2>(program, domain, bounds, None)
}

pub(crate) fn lower_isotropic_elastodynamics_subdomain_2d_with_boundaries(
    program: &KernelProgram,
    domain: RawId,
    bounds: [[f64; 2]; 2],
    boundaries: BTreeMap<(usize, eqiora_schema::kernel::BoundarySide), RawId>,
) -> Result<LoweredIsotropicElastodynamicsSubdomain2d, Diagnostic> {
    lower_isotropic_elastodynamics_subdomain::<2>(program, domain, bounds, Some(boundaries))
}

pub(crate) fn lower_isotropic_elastodynamics_subdomain<const D: usize>(
    program: &KernelProgram,
    domain: RawId,
    bounds: [[f64; 2]; D],
    boundaries: Option<BTreeMap<(usize, eqiora_schema::kernel::BoundarySide), RawId>>,
) -> Result<LoweredIsotropicElastodynamicsSubdomain<D>, Diagnostic> {
    let (displacement, velocity, load_potential) = exact_fields::<D>(program, domain)?;
    let volume_relations = relations_on(program, domain);
    if volume_relations.len() != 3 {
        return Err(lowering_error(
            domain,
            format!(
                "{D}D isotropic elastodynamics requires one load definition, one kinematic Relation, and one momentum Relation, found {} volume Relations",
                volume_relations.len()
            ),
        ));
    }

    let mut load_definition = None;
    let mut kinematics = None;
    let mut momentum = None;
    for relation in &volume_relations {
        require_continuous_relation(program, *relation)?;
        let expression = &relation_expression(program, *relation)?;
        let root = unique_root(expression, *relation)?;
        if let Some(source) = load_definition_root(expression, root, load_potential) {
            require_unique(
                &mut load_definition,
                (*relation, source),
                *relation,
                "load definition",
            )?;
        } else if kinematic_root(expression, root, displacement, velocity) {
            require_unique(&mut kinematics, *relation, *relation, "kinematic Relation")?;
        } else if let Some(parts) = momentum_root(expression, root, velocity, load_potential) {
            require_unique(
                &mut momentum,
                (*relation, parts),
                *relation,
                "momentum Relation",
            )?;
        } else {
            return Err(lowering_error(
                *relation,
                "volume Relation is not the canonical load definition, displacement/velocity kinematics, or first-order isotropic momentum",
            ));
        }
    }

    let (load_relation, load_root) =
        load_definition.ok_or_else(|| lowering_error(domain, "load definition is missing"))?;
    let kinematic_relation =
        kinematics.ok_or_else(|| lowering_error(domain, "kinematic Relation is missing"))?;
    let (momentum_relation, momentum_parts) =
        momentum.ok_or_else(|| lowering_error(domain, "momentum Relation is missing"))?;

    let load_expression = &relation_expression(program, load_relation)?;
    let load_potential_expression =
        spatial_expression::lower(program, load_expression, load_root, load_relation, D)?;
    let momentum_expression = &relation_expression(program, momentum_relation)?;
    let mass_density = spatial_expression::lower::<f64>(
        program,
        momentum_expression,
        momentum_parts.density,
        momentum_relation,
        D,
    )?;
    let Some(density) = mass_density.constant_value() else {
        return Err(lowering_error(
            momentum_relation,
            "elastodynamic mass density must be spatially constant",
        ));
    };
    if !density.is_finite() || density <= 0.0 {
        return Err(lowering_error(
            momentum_relation,
            "elastodynamic mass density must be finite and strictly positive",
        ));
    }

    let momentum_typed = typed_relation(program, momentum_relation)?;
    debug_assert_eq!(momentum_typed.expression(), momentum_expression);
    let (two_mu, lambda) = lower_isotropic_stress_coefficients(
        program,
        &momentum_typed,
        momentum_parts.stress,
        displacement,
        momentum_relation,
    )?;
    let shear_modulus = two_mu
        .clone()
        .multiply(ScalarSpatialExpression::constant(D, 0.5));
    let Some(mu) = shear_modulus.constant_value() else {
        return Err(lowering_error(
            momentum_relation,
            "shear modulus must be spatially constant",
        ));
    };
    let Some(lambda_value) = lambda.constant_value() else {
        return Err(lowering_error(
            momentum_relation,
            "first Lame parameter must be spatially constant",
        ));
    };
    let Some(material) = IsotropicElasticityMaterial::<D>::new(mu, lambda_value) else {
        return Err(lowering_error(
            momentum_relation,
            format!(
                "{D}D isotropic elastodynamics requires finite `mu > 0` and `lambda + 2 mu / D > 0`"
            ),
        ));
    };

    let lowered_boundary = match boundaries {
        Some(boundaries) => boundary::lower_dimension_with_boundaries::<D>(
            program,
            domain,
            velocity,
            displacement,
            &two_mu,
            &lambda,
            boundaries,
        )?,
        None => boundary::lower_dimension::<D>(
            program,
            domain,
            velocity,
            displacement,
            &two_mu,
            &lambda,
        )?,
    };
    let continuum = IsotropicElasticityContinuum::new(
        domain,
        displacement,
        load_potential,
        load_relation,
        momentum_relation,
        bounds,
        material,
        shear_modulus,
        lambda,
        load_potential_expression,
        lowered_boundary.inventory.clone(),
        lowered_boundary.boundary_relations.clone(),
        lowered_boundary.tractions.clone(),
    )
    .ok_or_else(|| {
        lowering_error(
            domain,
            format!("{D}D elastodynamics has no admitted linear-continuum profile"),
        )
    })?;
    let model = IsotropicElastodynamicsCartesianModel {
        continuum,
        velocity,
        kinematic_relation,
        mass_density,
        momentum_orientation: momentum_parts.orientation,
    };
    Ok(LoweredIsotropicElastodynamicsSubdomain {
        model,
        representation: continuum_representation(program, displacement)
            .expect("field validation establishes one continuum Representation"),
        volume_relations: [load_relation, kinematic_relation, momentum_relation],
        boundary: lowered_boundary,
    })
}

/// Exact semantic identities admitted by one dynamic-solid subdomain.
#[derive(Debug)]
pub(crate) struct LoweredIsotropicElastodynamicsSubdomain<const D: usize> {
    pub(crate) model: IsotropicElastodynamicsCartesianModel<D>,
    pub(crate) representation: RawId,
    pub(crate) volume_relations: [RawId; 3],
    pub(crate) boundary: boundary::LoweredElasticityBoundary<D>,
}

pub(crate) type LoweredIsotropicElastodynamicsSubdomain2d =
    LoweredIsotropicElastodynamicsSubdomain<2>;
#[derive(Debug, Clone, Copy)]
struct MomentumParts {
    density: ExprId,
    stress: ExprId,
    orientation: f64,
}

fn exact_fields<const D: usize>(
    program: &KernelProgram,
    domain: RawId,
) -> Result<(RawId, RawId, RawId), Diagnostic> {
    let fields = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Field(field)
                if has_edge(program, field.id().erase(), domain, EdgeKind::DefinedOn)
                    && continuum_representation(program, field.id().erase()).is_some() =>
            {
                Some(field)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if fields.len() != 3 {
        return Err(lowering_error(
            domain,
            format!(
                "{D}D isotropic elastodynamics requires exactly three continuum Fields, found {}",
                fields.len()
            ),
        ));
    }
    let components = u32::try_from(D).expect("supported dimensions fit portable component count");
    let vector_shape =
        ValueShape::new([components]).expect("supported dimensions are representable");
    let unique = |dimension, shape: &ValueShape, frame| {
        fields
            .iter()
            .filter(|field| {
                field.dimension() == dimension && field.shape() == shape && field.frame() == frame
            })
            .map(|field| field.id().erase())
            .collect::<Vec<_>>()
    };
    let displacement = unique(LENGTH, &vector_shape, ValueFrame::SpatialCartesian);
    let velocity = unique(VELOCITY, &vector_shape, ValueFrame::SpatialCartesian);
    let scalar_shape = ValueShape::scalar();
    let potential = unique(PRESSURE, &scalar_shape, ValueFrame::Invariant);
    if let ([displacement], [velocity], [potential]) = (
        displacement.as_slice(),
        velocity.as_slice(),
        potential.as_slice(),
    ) {
        let representation = continuum_representation(program, *displacement);
        if representation == continuum_representation(program, *velocity)
            && representation == continuum_representation(program, *potential)
        {
            return Ok((*displacement, *velocity, *potential));
        }
    }
    Err(lowering_error(
        domain,
        format!(
            "Fields must be exactly one length-valued displacement, one velocity-valued spatial Cartesian `[{D}]` Field, and one pressure-valued invariant scalar load potential on the same continuum Representation"
        ),
    ))
}

fn kinematic_root(
    expression: &ExprDag,
    root: ExprId,
    displacement: RawId,
    velocity: RawId,
) -> bool {
    if let Some(ExprNode::Neg(inner)) = expression.node(root) {
        return kinematic_root_oriented(expression, *inner, displacement, velocity);
    }
    kinematic_root_oriented(expression, root, displacement, velocity)
}

fn kinematic_root_oriented(
    expression: &ExprDag,
    root: ExprId,
    displacement: RawId,
    velocity: RawId,
) -> bool {
    let Some(ExprNode::Sub(left, right)) = expression.node(root) else {
        return false;
    };
    (is_derivative(expression, *left, displacement) && is_field(expression, *right, velocity))
        || (is_field(expression, *left, velocity)
            && is_derivative(expression, *right, displacement))
}

fn momentum_root(
    expression: &ExprDag,
    root: ExprId,
    velocity: RawId,
    load_potential: RawId,
) -> Option<MomentumParts> {
    if let Some(ExprNode::Neg(inner)) = expression.node(root) {
        return momentum_root_oriented(expression, *inner, velocity, load_potential).map(|parts| {
            MomentumParts {
                orientation: -1.0,
                ..parts
            }
        });
    }
    momentum_root_oriented(expression, root, velocity, load_potential).or_else(|| {
        let ExprNode::Sub(load, internal_balance) = expression.node(root)? else {
            return None;
        };
        momentum_parts(
            expression,
            *internal_balance,
            *load,
            velocity,
            load_potential,
        )
        .map(|parts| MomentumParts {
            orientation: -1.0,
            ..parts
        })
    })
}

fn momentum_root_oriented(
    expression: &ExprDag,
    root: ExprId,
    velocity: RawId,
    load_potential: RawId,
) -> Option<MomentumParts> {
    let ExprNode::Sub(internal_balance, load) = expression.node(root)? else {
        return None;
    };
    momentum_parts(
        expression,
        *internal_balance,
        *load,
        velocity,
        load_potential,
    )
}

fn momentum_parts(
    expression: &ExprDag,
    internal_balance: ExprId,
    load: ExprId,
    velocity: RawId,
    load_potential: RawId,
) -> Option<MomentumParts> {
    let ExprNode::Gradient(load_argument) = expression.node(load)? else {
        return None;
    };
    if !is_field(expression, *load_argument, load_potential) {
        return None;
    }
    let ExprNode::Sub(inertia, divergence) = expression.node(internal_balance)? else {
        return None;
    };
    let ExprNode::Divergence(stress) = expression.node(*divergence)? else {
        return None;
    };
    inertia_density(expression, *inertia, velocity).map(|density| MomentumParts {
        density,
        stress: *stress,
        orientation: 1.0,
    })
}

fn is_derivative(expression: &ExprDag, value: ExprId, field: RawId) -> bool {
    matches!(
        expression.node(value),
        Some(ExprNode::Symbol(SymbolRef::Derivative(id, std::num::NonZeroU32::MIN))) if id.erase() == field
    )
}

fn inertia_density(expression: &ExprDag, value: ExprId, velocity: RawId) -> Option<ExprId> {
    let ExprNode::Mul(left, right) = expression.node(value)? else {
        return None;
    };
    for (derivative, density) in [(*left, *right), (*right, *left)] {
        if matches!(expression.node(derivative), Some(ExprNode::Symbol(SymbolRef::Derivative(field, std::num::NonZeroU32::MIN))) if field.erase() == velocity)
        {
            return Some(density);
        }
    }
    None
}

fn is_field(expression: &ExprDag, value: ExprId, field: RawId) -> bool {
    matches!(
        expression.node(value),
        Some(ExprNode::Symbol(SymbolRef::Field(id))) if id.erase() == field
    )
}

fn require_unique<T>(
    slot: &mut Option<T>,
    value: T,
    owner: RawId,
    name: &str,
) -> Result<(), Diagnostic> {
    if slot.replace(value).is_some() {
        Err(lowering_error(owner, format!("{name} is not unique")))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use eqiora_compiler::compile;
    use eqiora_graph::{GraphStore, InMemoryGraphStore};
    use eqiora_schema::kernel::BoundarySide;
    use eqiora_sem::KernelProgram;

    use super::lower_isotropic_elastodynamics_cartesian_3d;
    use crate::canonical_boundary::PhysicalBoundaryDisposition;
    use crate::canonical_elasticity::{ElasticityIntegrationMeasure, IsotropicElasticityReduction};

    const SOURCE_3D: &str = r#"
model dynamic_solid_3d() {
  domain solid = box(0, 1, -1, 1, -2, 2);
  domain x_lower = boundary(solid, axis = 0, side = lower);
  domain x_upper = boundary(solid, axis = 0, side = upper);
  domain y_lower = boundary(solid, axis = 1, side = lower);
  domain y_upper = boundary(solid, axis = 1, side = upper);
  domain z_lower = boundary(solid, axis = 2, side = lower);
  domain z_upper = boundary(solid, axis = 2, side = upper);

  state displacement: vector<m, 3> on solid;
  state velocity: vector<m / s, 3> on solid in h1;
  variable load: kg / (m * s ^ 2) on solid;
  parameter density: kg / m ^ 3 = 3;
  parameter mu: kg / (m * s ^ 2) = 4;
  parameter lambda: kg / (m * s ^ 2) = 5;
  parameter zero_pressure: kg / (m * s ^ 2) = 0;

  relation load_definition on solid { load - zero_pressure = 0; }
  relation kinematics on solid { derivative(displacement) - velocity = 0; }
  relation momentum on solid {
    density * derivative(velocity)
      - div(
        2 * mu * symmetric_part(grad(displacement))
        + lambda * isotropic_lift(div(displacement))
      )
      - grad(load) = 0;
  }

  relation x_lower_zero on x_lower { trace(velocity) = 0; }
  relation x_upper_zero on x_upper { trace(velocity) = 0; }
  relation y_lower_zero on y_lower { trace(velocity) = 0; }
  relation y_upper_zero on y_upper { trace(velocity) = 0; }
  relation z_lower_zero on z_lower { trace(velocity) = 0; }
  relation z_upper_zero on z_upper { trace(velocity) = 0; }
}
"#;

    fn compile_program(source: &str) -> KernelProgram {
        let mut compiled = compile("dynamic-solid-3d.eqi", source).expect("source compiles");
        let (transaction, model, _) = compiled.remove(0).into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).expect("transaction commits");
        KernelProgram::from_snapshot(&store.snapshot(), model).expect("model validates")
    }

    #[test]
    fn lowers_three_dimensional_elastodynamics_through_the_shared_contract() {
        let model = lower_isotropic_elastodynamics_cartesian_3d(&compile_program(SOURCE_3D))
            .expect("exact 3D elastodynamics lowers");
        let continuum = model.continuum();
        assert_eq!(
            continuum.reduction(),
            IsotropicElasticityReduction::FullThreeDimensional
        );
        assert_eq!(
            continuum.integration_measure(),
            ElasticityIntegrationMeasure::Volume
        );
        assert_eq!(continuum.bounds(), &[[0.0, 1.0], [-1.0, 1.0], [-2.0, 2.0]]);
        assert_eq!(
            continuum.conservative_body_force(&[0.2, 0.3, 0.4]).unwrap(),
            [0.0; 3]
        );
        for axis in 0..3 {
            for side in [BoundarySide::Lower, BoundarySide::Upper] {
                assert!(matches!(
                    continuum
                        .boundary_inventory()
                        .boundary(axis, side)
                        .expect("complete 3D solid boundary inventory")
                        .disposition(),
                    PhysicalBoundaryDisposition::TraceZero
                ));
            }
        }
        assert_eq!(continuum.boundary_relations().len(), 6);
    }

    #[test]
    fn rejects_an_incomplete_three_dimensional_solid_boundary() {
        let incomplete = SOURCE_3D
            .replace(
                "  domain z_upper = boundary(solid, axis = 2, side = upper);\n",
                "",
            )
            .replace(
                "  relation z_upper_zero on z_upper { trace(velocity) = 0; }\n",
                "",
            );
        assert!(
            lower_isotropic_elastodynamics_cartesian_3d(&compile_program(&incomplete)).is_err()
        );
    }
}
