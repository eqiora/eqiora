#![cfg_attr(not(test), allow(dead_code))]
#![cfg_attr(test, allow(clippy::extra_unused_lifetimes, clippy::type_complexity))]
mod execution;
#[cfg(test)]
mod oracle;
use std::borrow::Cow;

use eqiora_assembly::LocalContribution;
use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id, RawId};
use eqiora_meshing::{
    AffineGeometryMap, GeometryMap, QuadratureRule, ReferenceCell, ReferenceCellFamily,
};
use eqiora_schema::kernel::typing::TypedResidual;
use eqiora_schema::kernel::{BoundarySide, ExprId, ExprNode, SymbolRef};
use eqiora_sem::KernelProgram;

use self::execution::{
    LocalFormExecution, LocalFormInput2d, LocalFormProgram2d,
    compile_derived_local_form_program_2d, compile_witness_local_form_program_2d,
};
use crate::affine_fem::physical_gradient;
use crate::canonical_boundary::PhysicalBoundaryDisposition;
use crate::discrete_space::{DiscreteSpace, HypercubeQ1Space};
use crate::form_compiler::vocabulary::{
    BoundarySource, PrimalGalerkinCorrespondence, PrimalGalerkinSource,
};
pub(super) use crate::form_compiler::vocabulary::{
    DIVERGENCE_BY_PARTS, SOURCE_PAIRING, TEST_PAIRING,
};
use crate::spatial_expression::ScalarSpatialExpression;
const DIMENSION: usize = 2;
const COMPONENTS: usize = 2;
const SCALAR_DOFS: usize = 4;
const LOCAL_DOFS: usize = SCALAR_DOFS * COMPONENTS;
const MAX_DAG_NODES: usize = 4_096;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VolumeNodes {
    root: ExprId,
    divergence: ExprId,
    stress: ExprId,
    load_gradient: ExprId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BoundaryRole {
    domain: RawId,
    relation: RawId,
    axis: usize,
    side: BoundarySide,
    trace_node: ExprId,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DerivedCartesianQ1ElasticityForm2d {
    domain: RawId,
    displacement: RawId,
    balance_relation: RawId,
    material_parameters: [Id<kinds::Parameter>; 2],
    parameters: [Id<kinds::Parameter>; 3],
    load: ScalarSpatialExpression,
    volume: VolumeNodes,
    boundaries: Vec<BoundaryRole>,
    certificate: PrimalGalerkinCorrespondence,
    typed_balance: TypedResidual<RawId>,
    program: LocalFormProgram2d,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ElasticityDerivationSource<'a> {
    typed_balance: &'a TypedResidual<RawId>,
    domain: RawId,
    displacement: RawId,
    material_parameters: [Id<kinds::Parameter>; 2],
    balance_relation: RawId,
    volume: VolumeNodes,
    boundaries: &'a [BoundaryRole],
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AdmittedCartesianQ1ElasticityForm2d<'form> {
    form: Option<&'form DerivedCartesianQ1ElasticityForm2d>,
    quadrature: QuadratureRule,
    program: Cow<'form, LocalFormProgram2d>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CartesianElasticityDifferentialActions2d {
    contribution: LocalContribution,
    residual: [f64; LOCAL_DOFS],
    state_direction_action: [f64; LOCAL_DOFS],
    parameter_direction_action: [f64; LOCAL_DOFS],
    state_transpose_action: [f64; LOCAL_DOFS],
    parameter_transpose_action: [f64; 3],
}

impl DerivedCartesianQ1ElasticityForm2d {
    #[cfg(test)]
    pub(super) const fn correspondence(&self) -> &PrimalGalerkinCorrespondence {
        &self.certificate
    }

    pub(crate) fn admit_quadrature(
        &self,
        quadrature: &QuadratureRule,
    ) -> Result<AdmittedCartesianQ1ElasticityForm2d<'_>, Diagnostic> {
        self.validate_certificate()?;
        let expected = QuadratureRule::tensor_product_gauss_legendre(DIMENSION, 2)?;
        if quadrature != &expected {
            return Err(tape_error(
                "derived elasticity requires the certified two-point tensor quadrature",
            ));
        }
        let regenerated =
            compile_derived_local_form_program_2d(self.derivation_source(), &self.certificate)?;
        if regenerated != self.program {
            return Err(tape_error(
                "derived elasticity instruction tape is stale or foreign",
            ));
        }
        Ok(AdmittedCartesianQ1ElasticityForm2d {
            form: Some(self),
            quadrature: quadrature.clone(),
            program: Cow::Borrowed(&self.program),
        })
    }

    fn validate_certificate(&self) -> Result<(), Diagnostic> {
        if self.parameters[..2] != self.material_parameters {
            return Err(tape_error(
                "derived elasticity identity replay differs from its certificate",
            ));
        }
        if self.load.parameter_fields() != [self.parameters[2]] {
            return Err(tape_error(
                "derived elasticity load Parameter identity is stale",
            ));
        }
        self.derivation_source().validate(&self.certificate)
    }

    fn derivation_source(&self) -> ElasticityDerivationSource<'_> {
        ElasticityDerivationSource {
            typed_balance: &self.typed_balance,
            domain: self.domain,
            displacement: self.displacement,
            material_parameters: self.material_parameters,
            balance_relation: self.balance_relation,
            volume: self.volume,
            boundaries: &self.boundaries,
        }
    }

    fn validate_geometry(&self, geometry: &AffineGeometryMap) -> Result<(), Diagnostic> {
        let admitted_origin = geometry
            .origin()
            .iter()
            .all(|value| matches!(value.to_bits(), bits if bits == 0.25_f64.to_bits() || bits == 0.75_f64.to_bits()));
        if geometry.reference_cell() != ReferenceCell::hypercube(DIMENSION)?
            || geometry.physical_dimension() != DIMENSION
            || geometry.jacobian() != [0.25_f64, 0.0_f64, 0.0_f64, 0.25_f64]
            || !admitted_origin
        {
            return Err(tape_error(
                "derived elasticity geometry is outside the certified unit-square patch",
            ));
        }
        Ok(())
    }

    fn action_load(
        &self,
        parameters: [f64; 3],
        direction: [f64; 3],
    ) -> Result<ActionLoad, Diagnostic> {
        let base = self
            .load
            .bind_parameter_point(&self.parameters, &parameters)?;
        let value = affine_body_force(&base)?;
        let directed_point: [f64; 3] =
            std::array::from_fn(|index| parameters[index] + direction[index]);
        let directed = self
            .load
            .bind_parameter_point(&self.parameters, &directed_point)?;
        let directed_force = affine_body_force(&directed)?;
        let tangent = std::array::from_fn(|axis| directed_force[axis] - value[axis]);
        let mut jacobian = [[0.0; COMPONENTS]; 3];
        for parameter in 0..3 {
            let mut point = parameters;
            point[parameter] += 1.0;
            let shifted = self.load.bind_parameter_point(&self.parameters, &point)?;
            let shifted_force = affine_body_force(&shifted)?;
            for axis in 0..COMPONENTS {
                jacobian[parameter][axis] = shifted_force[axis] - value[axis];
            }
        }
        Ok(ActionLoad {
            value,
            tangent,
            jacobian,
        })
    }
}

impl ElasticityDerivationSource<'_> {
    fn validate(&self, certificate: &PrimalGalerkinCorrespondence) -> Result<(), Diagnostic> {
        let boundary_sources = boundary_sources(self.boundaries);
        if self.typed_balance.expression().nodes().len() > MAX_DAG_NODES
            || execution::recognize_volume(
                self.typed_balance,
                self.balance_relation,
                self.displacement,
            )? != self.volume
            || self.boundaries.len() != 2 * DIMENSION
            || certificate
                .replay(correspondence_source(
                    self.domain,
                    self.displacement,
                    self.balance_relation,
                    self.volume,
                    &boundary_sources,
                ))
                .is_err()
        {
            return Err(tape_error(
                "elasticity certificate source identities or resource bounds are stale",
            ));
        }
        Ok(())
    }
}

impl<'form> AdmittedCartesianQ1ElasticityForm2d<'form> {
    #[allow(private_interfaces)]
    pub(super) fn executable_program(&self) -> &LocalFormProgram2d {
        self.program.as_ref()
    }

    pub(crate) fn evaluate(
        &self,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
        shear_modulus: f64,
        first_lame_parameter: f64,
        body_force_potential: Option<&ScalarSpatialExpression>,
    ) -> Result<LocalContribution, Diagnostic> {
        if let Some(form) = self.form {
            let potential = body_force_potential.ok_or_else(|| {
                tape_error("derived elasticity requires its certificate-owned load")
            })?;
            if !potential.is_same_coefficient_as(&form.load) {
                return Err(tape_error(
                    "derived elasticity received a foreign load certificate",
                ));
            }
            form.validate_geometry(geometry)?;
        }
        cumulative_local_form(
            self.executable_program(),
            geometry,
            quadrature,
            &self.quadrature,
            [shear_modulus, first_lame_parameter],
            body_force_potential,
            EvaluationRequest::Primal,
        )
        .map(|result| result.contribution)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_with_actions(
        &self,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
        parameters: [f64; 3],
        state: &[f64; 8],
        state_direction: &[f64; 8],
        parameter_direction: [f64; 3],
        cotangent: &[f64; 8],
    ) -> Result<CartesianElasticityDifferentialActions2d, Diagnostic> {
        let form = self.form.ok_or_else(|| {
            tape_error("elasticity differential actions require a derivation certificate")
        })?;
        form.validate_geometry(geometry)?;
        let load = form.action_load(parameters, parameter_direction)?;
        cumulative_local_form(
            self.executable_program(),
            geometry,
            quadrature,
            &self.quadrature,
            [parameters[0], parameters[1]],
            None,
            EvaluationRequest::Actions {
                state,
                state_direction,
                parameter_direction,
                cotangent,
                load: &load,
            },
        )
    }
}

macro_rules! action_accessor {
    ($name:ident, $field:ident, $value:ty) => {
        pub(crate) const fn $name(&self) -> &$value {
            &self.$field
        }
    };
}

impl CartesianElasticityDifferentialActions2d {
    action_accessor!(contribution, contribution, LocalContribution);
    action_accessor!(residual, residual, [f64; 8]);
    action_accessor!(state_direction_action, state_direction_action, [f64; 8]);
    action_accessor!(
        parameter_direction_action,
        parameter_direction_action,
        [f64; 8]
    );
    action_accessor!(state_transpose_action, state_transpose_action, [f64; 8]);
    action_accessor!(
        parameter_transpose_action,
        parameter_transpose_action,
        [f64; 3]
    );
}

pub(crate) fn derive_cartesian_q1_elasticity_form_2d(
    program: &KernelProgram,
) -> Result<DerivedCartesianQ1ElasticityForm2d, Diagnostic> {
    let model = crate::canonical_elasticity::lower_isotropic_elasticity_cartesian_2d(program)
        .map_err(|error| tape_error(error.message()))?;
    if model.bounds() != &[[0.0_f64, 1.0_f64], [0.0_f64, 1.0_f64]]
        || model.shear_modulus().to_bits() != 3.0_f64.to_bits()
        || model.first_lame_parameter().to_bits() != 2.0_f64.to_bits()
    {
        return Err(tape_error(
            "derived elasticity requires the exact frozen unit-square Model",
        ));
    }
    let balance_relation = model.equilibrium_relation();
    let typed_balance = typed_relation(program, balance_relation)?;
    let volume =
        execution::recognize_volume(&typed_balance, balance_relation, model.displacement())?;
    let material_parameters = exact_material_parameters(&model)?;
    let (load_parameter, load_value) = exact_load_parameter(program, &model)?;
    if load_value.to_bits() != 1.0_f64.to_bits() {
        return Err(tape_error(
            "derived elasticity requires the frozen pressure-gradient value",
        ));
    }
    let parameters = [
        material_parameters[0],
        material_parameters[1],
        load_parameter,
    ];
    let boundaries = exact_boundaries(program, &model)?;
    let certificate = build_certificate(
        model.domain(),
        model.displacement(),
        balance_relation,
        volume,
        &boundaries,
    );
    let derivation_source = ElasticityDerivationSource {
        typed_balance: &typed_balance,
        domain: model.domain(),
        displacement: model.displacement(),
        material_parameters,
        balance_relation,
        volume,
        boundaries: &boundaries,
    };
    let tape = compile_derived_local_form_program_2d(derivation_source, &certificate)?;
    let form = DerivedCartesianQ1ElasticityForm2d {
        domain: model.domain(),
        displacement: model.displacement(),
        balance_relation,
        material_parameters,
        parameters,
        load: model.load_potential_expression().clone(),
        volume,
        boundaries,
        certificate,
        typed_balance,
        program: tape,
    };
    form.validate_certificate()?;
    Ok(form)
}

pub(crate) fn compile_cartesian_q1_elasticity_form_2d(
    quadrature: &QuadratureRule,
) -> Result<AdmittedCartesianQ1ElasticityForm2d<'static>, Diagnostic> {
    let space = HypercubeQ1Space::new(DIMENSION)?;
    if quadrature.reference_cell().family() != ReferenceCellFamily::Hypercube
        || quadrature.reference_cell() != space.reference_cell()
    {
        return Err(realization_error(
            "compiled Cartesian Q1 elasticity requires dimension-matched hypercube quadrature",
        ));
    }
    let program = compile_witness_local_form_program_2d()?;
    Ok(AdmittedCartesianQ1ElasticityForm2d {
        form: None,
        quadrature: quadrature.clone(),
        program: Cow::Owned(program),
    })
}

#[derive(Debug, Clone, Copy)]
struct ActionLoad {
    value: [f64; COMPONENTS],
    tangent: [f64; COMPONENTS],
    jacobian: [[f64; COMPONENTS]; 3],
}

enum EvaluationRequest<'a> {
    Primal,
    Actions {
        state: &'a [f64; LOCAL_DOFS],
        state_direction: &'a [f64; LOCAL_DOFS],
        parameter_direction: [f64; 3],
        cotangent: &'a [f64; LOCAL_DOFS],
        load: &'a ActionLoad,
    },
}

#[allow(clippy::too_many_arguments)]
fn cumulative_local_form(
    program: &LocalFormProgram2d,
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
    admitted_quadrature: &QuadratureRule,
    material: [f64; 2],
    potential: Option<&ScalarSpatialExpression>,
    request: EvaluationRequest<'_>,
) -> Result<CartesianElasticityDifferentialActions2d, Diagnostic> {
    validate_realization(geometry, quadrature, admitted_quadrature)?;
    let inverse = geometry.inverse_jacobian()?;
    let space = HypercubeQ1Space::new(DIMENSION)?;
    let mut matrix = [0.0; LOCAL_DOFS * LOCAL_DOFS];
    let mut rhs = [0.0; LOCAL_DOFS];
    let mut residual = [0.0; LOCAL_DOFS];
    let mut state_action = [0.0; LOCAL_DOFS];
    let mut parameter_action = [0.0; LOCAL_DOFS];
    let mut state_transpose = [0.0; LOCAL_DOFS];
    let mut parameter_transpose = [0.0; 3];
    let mut physical = [0.0; DIMENSION];
    let zero_parameters =
        potential.map_or_else(Vec::new, |value| vec![0.0; value.parameter_fields().len()]);

    for point in quadrature.points() {
        let basis = space.tabulate(&point.coordinates)?;
        geometry.map_point(&point.coordinates, &mut physical)?;
        let gradients = basis
            .reference_gradients()
            .as_chunks::<DIMENSION>()
            .0
            .iter()
            .map(|gradient| physical_gradient(gradient, &inverse, DIMENSION))
            .collect::<Vec<_>>();
        let body_force = match &request {
            EvaluationRequest::Primal => potential.map_or(Ok([0.0; COMPONENTS]), |value| {
                potential_gradient(value, &physical, &zero_parameters)
            })?,
            EvaluationRequest::Actions { load, .. } => load.value,
        };
        let state = match &request {
            EvaluationRequest::Primal => &[0.0; LOCAL_DOFS],
            EvaluationRequest::Actions { state, .. } => *state,
        };
        let jets = interpolate_jets(state, &gradients);
        let direction_jets = match &request {
            EvaluationRequest::Primal => [[0.0; DIMENSION]; COMPONENTS],
            EvaluationRequest::Actions {
                state_direction, ..
            } => interpolate_jets(state_direction, &gradients),
        };
        let scale = point.weight * geometry.measure_scale();

        for test in 0..SCALAR_DOFS {
            let inputs = bind_inputs(
                program,
                material,
                jets,
                [gradients[test][0], gradients[test][1]],
                basis.values()[test],
                body_force,
                scale,
            )?;
            for row_component in 0..COMPONENTS {
                let mut root_cotangent = [0.0; COMPONENTS];
                root_cotangent[row_component] = 1.0;
                let executed = execute_sweep(program, &inputs, None, Some(&root_cotangent))?;
                if row_component == 0 {
                    for component in 0..COMPONENTS {
                        let row = local_dof(test, component);
                        match request {
                            EvaluationRequest::Primal => rhs[row] -= executed.roots()[component],
                            EvaluationRequest::Actions { .. } => {
                                residual[row] += executed.roots()[component];
                            }
                        }
                    }
                }
                scatter_state_cotangent(
                    program,
                    executed.input_cotangents().expect("reverse requested"),
                    &gradients,
                    local_dof(test, row_component),
                    &mut matrix,
                );
            }

            if let EvaluationRequest::Actions {
                parameter_direction,
                cotangent,
                load,
                ..
            } = &request
            {
                let state_tangent = bind_state_tangent(program, direction_jets);
                let state_forward = execute_sweep(program, &inputs, Some(&state_tangent), None)?;
                let parameter_tangent =
                    bind_parameter_tangent(program, *parameter_direction, load.tangent);
                let parameter_forward =
                    execute_sweep(program, &inputs, Some(&parameter_tangent), None)?;
                for component in 0..COMPONENTS {
                    let row = local_dof(test, component);
                    state_action[row] +=
                        state_forward.root_tangents().expect("forward requested")[component];
                    parameter_action[row] += parameter_forward
                        .root_tangents()
                        .expect("forward requested")[component];
                }
                let root_cotangent = [cotangent[local_dof(test, 0)], cotangent[local_dof(test, 1)]];
                let reverse = execute_sweep(program, &inputs, None, Some(&root_cotangent))?;
                scatter_state_action(
                    program,
                    reverse.input_cotangents().expect("reverse requested"),
                    &gradients,
                    &mut state_transpose,
                );
                scatter_parameter_action(
                    program,
                    reverse.input_cotangents().expect("reverse requested"),
                    load,
                    &mut parameter_transpose,
                );
            }
        }
    }

    if let EvaluationRequest::Actions { state, .. } = request {
        for row in 0..LOCAL_DOFS {
            let mut applied = 0.0;
            for column in 0..LOCAL_DOFS {
                applied += matrix[row * LOCAL_DOFS + column] * state[column];
            }
            rhs[row] = applied - residual[row];
        }
    }
    Ok(CartesianElasticityDifferentialActions2d {
        contribution: LocalContribution::new(
            LOCAL_DOFS,
            LOCAL_DOFS,
            matrix.to_vec(),
            rhs.to_vec(),
        )?,
        residual,
        state_direction_action: state_action,
        parameter_direction_action: parameter_action,
        state_transpose_action: state_transpose,
        parameter_transpose_action: parameter_transpose,
    })
}

fn execute_sweep<'a>(
    program: &'a LocalFormProgram2d,
    inputs: &'a [f64],
    tangent: Option<&'a [f64]>,
    cotangent: Option<&'a [f64]>,
) -> Result<LocalFormExecution, Diagnostic> {
    program.execute(inputs, tangent, cotangent)
}

fn bind_inputs(
    program: &LocalFormProgram2d,
    material: [f64; 2],
    jets: [[f64; DIMENSION]; COMPONENTS],
    shape_gradient: [f64; DIMENSION],
    shape_value: f64,
    body_force: [f64; COMPONENTS],
    scale: f64,
) -> Result<Vec<f64>, Diagnostic> {
    program
        .inputs
        .iter()
        .map(|input| match *input {
            LocalFormInput2d::Material(index) => material.get(usize::from(index)).copied(),
            LocalFormInput2d::StateJet { component, axis } => jets
                .get(usize::from(component))
                .and_then(|values| values.get(usize::from(axis)))
                .copied(),
            LocalFormInput2d::ShapeGradient(axis) => shape_gradient.get(usize::from(axis)).copied(),
            LocalFormInput2d::ShapeValue => Some(shape_value),
            LocalFormInput2d::BodyForce(component) => {
                body_force.get(usize::from(component)).copied()
            }
            LocalFormInput2d::QuadratureScale => Some(scale),
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| tape_error("generated program requested an invalid dense input"))
}

fn bind_state_tangent(
    program: &LocalFormProgram2d,
    jets: [[f64; DIMENSION]; COMPONENTS],
) -> Vec<f64> {
    program
        .inputs
        .iter()
        .map(|input| match *input {
            LocalFormInput2d::StateJet { component, axis } => {
                jets[usize::from(component)][usize::from(axis)]
            }
            _ => 0.0,
        })
        .collect()
}

fn bind_parameter_tangent(
    program: &LocalFormProgram2d,
    direction: [f64; 3],
    body_force: [f64; COMPONENTS],
) -> Vec<f64> {
    program
        .inputs
        .iter()
        .map(|input| match *input {
            LocalFormInput2d::Material(index) => direction[usize::from(index)],
            LocalFormInput2d::BodyForce(component) => body_force[usize::from(component)],
            _ => 0.0,
        })
        .collect()
}

fn interpolate_jets(
    state: &[f64; LOCAL_DOFS],
    gradients: &[Vec<f64>],
) -> [[f64; DIMENSION]; COMPONENTS] {
    let mut jets = [[0.0; DIMENSION]; COMPONENTS];
    for (local_node, gradient) in gradients.iter().enumerate() {
        for component in 0..COMPONENTS {
            for axis in 0..DIMENSION {
                jets[component][axis] += state[local_dof(local_node, component)] * gradient[axis];
            }
        }
    }
    jets
}

fn scatter_state_cotangent(
    program: &LocalFormProgram2d,
    cotangents: &[f64],
    gradients: &[Vec<f64>],
    row: usize,
    matrix: &mut [f64; LOCAL_DOFS * LOCAL_DOFS],
) {
    for (input, cotangent) in program.inputs.iter().zip(cotangents) {
        if let LocalFormInput2d::StateJet { component, axis } = *input {
            for (local_node, gradient) in gradients.iter().enumerate() {
                let column = local_dof(local_node, usize::from(component));
                matrix[row * LOCAL_DOFS + column] += cotangent * gradient[usize::from(axis)];
            }
        }
    }
}

fn scatter_state_action(
    program: &LocalFormProgram2d,
    cotangents: &[f64],
    gradients: &[Vec<f64>],
    output: &mut [f64; LOCAL_DOFS],
) {
    for (input, cotangent) in program.inputs.iter().zip(cotangents) {
        if let LocalFormInput2d::StateJet { component, axis } = *input {
            for (local_node, gradient) in gradients.iter().enumerate() {
                output[local_dof(local_node, usize::from(component))] +=
                    cotangent * gradient[usize::from(axis)];
            }
        }
    }
}

fn scatter_parameter_action(
    program: &LocalFormProgram2d,
    cotangents: &[f64],
    load: &ActionLoad,
    output: &mut [f64; 3],
) {
    for (input, cotangent) in program.inputs.iter().zip(cotangents) {
        match *input {
            LocalFormInput2d::Material(index) => output[usize::from(index)] += cotangent,
            LocalFormInput2d::BodyForce(component) => {
                for (parameter, value) in output.iter_mut().enumerate() {
                    *value += cotangent * load.jacobian[parameter][usize::from(component)];
                }
            }
            _ => {}
        }
    }
}

fn exact_material_parameters(
    model: &crate::canonical_elasticity::IsotropicElasticityContinuum<2>,
) -> Result<[Id<kinds::Parameter>; 2], Diagnostic> {
    let [mu] = model.shear_modulus_expression().parameter_fields() else {
        return Err(tape_error(
            "shear modulus does not retain one exact Parameter",
        ));
    };
    let [lambda] = model.first_lame_parameter_expression().parameter_fields() else {
        return Err(tape_error(
            "first Lame parameter does not retain one exact Parameter",
        ));
    };
    if mu == lambda {
        return Err(tape_error("Lame Parameter roles are ambiguous"));
    }
    Ok([*mu, *lambda])
}

fn exact_load_parameter(
    program: &KernelProgram,
    model: &crate::canonical_elasticity::IsotropicElasticityContinuum<2>,
) -> Result<(Id<kinds::Parameter>, f64), Diagnostic> {
    let typed = typed_relation(program, model.load_definition_relation())?;
    let expression = typed.expression();
    let [root] = expression.roots() else {
        return Err(tape_error("load definition must have one typed root"));
    };
    let Some(ExprNode::Sub(field, product)) = expression.node(*root) else {
        return Err(tape_error(
            "load definition root is outside the frozen class",
        ));
    };
    let Some(ExprNode::Symbol(SymbolRef::Field(load))) = expression.node(*field) else {
        return Err(tape_error("load definition does not bind its exact Field"));
    };
    let Some(ExprNode::Mul(parameter, coordinate)) = expression.node(*product) else {
        return Err(tape_error(
            "load definition is not an ordered affine product",
        ));
    };
    let Some(ExprNode::Symbol(SymbolRef::Parameter(parameter))) = expression.node(*parameter)
    else {
        return Err(tape_error(
            "load definition has no exact Parameter identity",
        ));
    };
    if load.erase() != model.load_potential()
        || !matches!(
            expression.node(*coordinate),
            Some(ExprNode::SpatialCoordinate(0))
        )
        || model.load_potential_expression().parameter_fields() != [*parameter]
    {
        return Err(tape_error(
            "load definition identity or coordinate is stale",
        ));
    }
    let value = program
        .value(parameter.erase())
        .ok_or_else(|| tape_error("load Parameter has no revision-local value"))?
        .value();
    Ok((*parameter, value))
}

fn exact_boundaries(
    program: &KernelProgram,
    model: &crate::canonical_elasticity::IsotropicElasticityContinuum<2>,
) -> Result<Vec<BoundaryRole>, Diagnostic> {
    let mut roles = Vec::with_capacity(2 * DIMENSION);
    for axis in 0..DIMENSION {
        for side in [BoundarySide::Lower, BoundarySide::Upper] {
            let boundary = model
                .boundary_inventory()
                .boundary(axis, side)
                .ok_or_else(|| tape_error("elasticity boundary inventory is incomplete"))?;
            if boundary.disposition() != PhysicalBoundaryDisposition::TraceZero {
                return Err(tape_error(
                    "derived elasticity requires complete homogeneous essential boundary",
                ));
            }
            let relation = model
                .boundary_relations()
                .iter()
                .find(|binding| binding.boundary() == boundary.boundary())
                .map(|binding| binding.relation())
                .ok_or_else(|| tape_error("elasticity boundary Relation is absent"))?;
            let typed = typed_relation(program, relation)?;
            let [root] = typed.expression().roots() else {
                return Err(tape_error("elasticity boundary must have one typed root"));
            };
            let Some(ExprNode::Trace(field)) = typed.expression().node(*root) else {
                return Err(tape_error("elasticity boundary is not an exact zero trace"));
            };
            if !matches!(
                typed.expression().node(*field),
                Some(ExprNode::Symbol(SymbolRef::Field(value)))
                    if value.erase() == model.displacement()
            ) {
                return Err(tape_error("elasticity boundary traces a foreign Field"));
            }
            roles.push(BoundaryRole {
                domain: boundary.boundary(),
                relation,
                axis,
                side,
                trace_node: *root,
            });
        }
    }
    Ok(roles)
}

#[allow(clippy::too_many_arguments)]
fn build_certificate(
    domain: RawId,
    displacement: RawId,
    balance_relation: RawId,
    volume: VolumeNodes,
    boundaries: &[BoundaryRole],
) -> PrimalGalerkinCorrespondence {
    let boundary_sources = boundary_sources(boundaries);
    PrimalGalerkinCorrespondence::derive(PrimalGalerkinSource {
        domain,
        unknown: displacement,
        volume_relation: balance_relation,
        root: volume.root,
        divergence: volume.divergence,
        divergence_sign: super::vocabulary::WeakSign::Positive,
        source: volume.load_gradient,
        boundaries: &boundary_sources,
    })
}

fn boundary_sources(boundaries: &[BoundaryRole]) -> Vec<BoundarySource> {
    boundaries
        .iter()
        .map(|boundary| BoundarySource {
            domain: boundary.domain,
            relation: boundary.relation,
            trace_node: boundary.trace_node,
        })
        .collect()
}

fn correspondence_source<'a>(
    domain: RawId,
    displacement: RawId,
    balance_relation: RawId,
    volume: VolumeNodes,
    boundaries: &'a [BoundarySource],
) -> PrimalGalerkinSource<'a> {
    PrimalGalerkinSource {
        domain,
        unknown: displacement,
        volume_relation: balance_relation,
        root: volume.root,
        divergence: volume.divergence,
        divergence_sign: super::vocabulary::WeakSign::Positive,
        source: volume.load_gradient,
        boundaries,
    }
}

fn typed_relation(
    program: &KernelProgram,
    relation: RawId,
) -> Result<TypedResidual<RawId>, Diagnostic> {
    let relation = relation
        .downcast::<kinds::Relation>()
        .ok_or_else(|| tape_error("selected elasticity owner is not a Relation"))?;
    program
        .typed_relation_residual(relation)
        .map_err(|diagnostics| {
            diagnostics
                .into_iter()
                .next()
                .unwrap_or_else(|| tape_error("typed Relation replay produced no diagnostic"))
        })
}

fn affine_body_force(expression: &ScalarSpatialExpression) -> Result<[f64; 2], Diagnostic> {
    expression
        .affine_gradient()
        .and_then(|gradient| gradient.try_into().ok())
        .ok_or_else(|| tape_error("certificate-owned load is not the frozen affine tape"))
}

fn potential_gradient(
    potential: &ScalarSpatialExpression,
    coordinates: &[f64; DIMENSION],
    zero_parameter_tangent: &[f64],
) -> Result<[f64; COMPONENTS], Diagnostic> {
    let mut gradient = [0.0; COMPONENTS];
    for axis in 0..DIMENSION {
        let mut coordinate_tangent = [0.0; DIMENSION];
        coordinate_tangent[axis] = 1.0;
        gradient[axis] = potential
            .evaluate_jvp(coordinates, &coordinate_tangent, zero_parameter_tangent)?
            .1;
    }
    Ok(gradient)
}

fn validate_realization(
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
    admitted_quadrature: &QuadratureRule,
) -> Result<(), Diagnostic> {
    if quadrature != admitted_quadrature
        || geometry.reference_cell() != quadrature.reference_cell()
        || geometry.reference_cell() != ReferenceCell::hypercube(DIMENSION)?
        || geometry.physical_dimension() != DIMENSION
    {
        return Err(realization_error(
            "compiled elasticity geometry, Q1 space, or quadrature record drifted",
        ));
    }
    Ok(())
}

const fn local_dof(node: usize, component: usize) -> usize {
    node * COMPONENTS + component
}

fn tape_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(
        codes::INVALID_SPATIAL_LOWERING,
        format!("FEM form compiler elasticity tape gate: {}", message.into()),
    )
}

fn realization_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_DISCRETIZATION, message)
}
