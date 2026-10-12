//! Bind the native flow profile to exact equation-derived Region actions.

use eqiora_core::{Diagnostic, DimExponents, DynQuantity, RawId};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::{AffineGeometryMap, GeometryMap, ReferenceCell};
use eqiora_realization::Space;
use eqiora_schema::kernel::KernelNode;
use eqiora_sem::KernelProgram;

use super::invalid;
use crate::form_compiler::region::{
    BoundRegionForm, CompiledRegionForm, PreparedRegionCell, RegionFieldBinding, RegionTimeBinding,
};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct StepForm {
    pub form: BoundRegionForm<f64>,
    pub vector: RawId,
    pub scalar: RawId,
    pub length: f64,
    pub origin: [f64; 2],
}

impl StepForm {
    pub fn reference(density: f64, viscosity: f64, step: f64) -> Result<Self, Diagnostic> {
        // The low-level numerical API supplies physical constants and a load callback.
        // Its reference mathematics goes through the same compiler as authored Models.
        let source = format!(
            "public operator outer_product(input left: spatial[1], input right: spatial[1]): spatial[2] = component(left, 0) * component(right, 1);
model Step() {{
          domain body = box(0, 1, 0, 1);
          parameter density: kg / m ^ 3 = {density};
          parameter viscosity: kg / (m * s) = {viscosity};
          state a: vector<m / s, 2> on body;
          variable b: kg / (m * s ^ 2) on body;
          relation balance on body {{ density * derivative(a)
            + div(density * outer_product(left = a, right = a))
            - div(2 * viscosity * symmetric_part(grad(a)) - isotropic_lift(b)) = 0; }}
          relation constraint on body {{ div(a) = 0; }}
        }}"
        );
        let (transaction, model, _) = eqiora_compiler::compile("mini-step.eqi", &source)
            .map_err(|_| invalid("reference step equations did not compile"))?
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store
            .commit(transaction)
            .map_err(|_| invalid("reference step graph commit failed"))?;
        let program = KernelProgram::from_snapshot(&store.snapshot(), model)
            .map_err(|mut errors| errors.remove(0))?;
        let domain = program
            .nodes()
            .find_map(|node| match node {
                KernelNode::Domain(value) => Some(value.id().erase()),
                _ => None,
            })
            .expect("reference equation domain");
        Self::bind(&program, domain, step, [1.0, 1.0, 1.0], [0.0; 2])
    }

    pub fn bind(
        program: &KernelProgram,
        domain: RawId,
        step: f64,
        scales: [f64; 3],
        origin: [f64; 2],
    ) -> Result<Self, Diagnostic> {
        let compiled = CompiledRegionForm::<f64>::derive(program, domain, 2)?;
        let [length, vector_scale, scalar_scale] = scales;
        let length_dim = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("length");
        let measure_dim = length_dim.pow(2, 1).expect("area");
        let mut bindings = Vec::new();
        let mut vector = None;
        let mut scalar = None;
        for (field, value_type) in compiled.fields() {
            let is_scalar = value_type.shape().is_scalar();
            if is_scalar {
                scalar = Some(field);
            } else {
                vector = Some(field);
            }
            bindings.push(RegionFieldBinding {
                field,
                space: if is_scalar {
                    Space::continuous_lagrange(std::num::NonZeroU16::MIN)
                } else {
                    Space::simplex_p1_bubble()
                },
                scale: DynQuantity::new(
                    if is_scalar {
                        scalar_scale
                    } else {
                        vector_scale
                    },
                    value_type.dimension(),
                ),
            });
        }
        if bindings.len() != 2 || vector.is_none() || scalar.is_none() {
            return Err(invalid(
                "MINI ordering requires exactly one vector and one scalar equation Field",
            ));
        }
        let rows = compiled
            .rows()
            .map(|(relation, _, value_type)| {
                let value = if value_type.shape().is_scalar() {
                    -1.0 / (vector_scale * length)
                } else {
                    1.0 / (scalar_scale * length)
                };
                let dimension = value_type
                    .dimension()
                    .mul(measure_dim)
                    .and_then(|dim| dim.pow(-1, 1))
                    .expect("bounded weak-row units");
                (relation, DynQuantity::new(value, dimension))
            })
            .collect();
        let time_dim = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("time");
        let form = compiled.bind(
            ReferenceCell::simplex(2)?,
            &bindings,
            &rows,
            Some(&RegionTimeBinding {
                step: DynQuantity::new(step, time_dim),
                states: Vec::new(),
            }),
        )?;
        Ok(Self {
            form,
            vector: vector.unwrap(),
            scalar: scalar.unwrap(),
            length,
            origin,
        })
    }

    pub fn prepare_cell(
        &self,
        normalized: &AffineGeometryMap,
        quadrature: &eqiora_meshing::QuadratureRule,
        load: &impl Fn([f64; 2]) -> Result<[f64; 2], Diagnostic>,
    ) -> Result<PreparedRegionCell<f64>, Diagnostic> {
        let mut prepared = self
            .form
            .prepare_cell(&self.geometry(normalized)?, quadrature)?;
        let paired = self
            .form
            .pair_load(self.vector, normalized, quadrature, |point| {
                load([point[0], point[1]]).map(Vec::from)
            })?;
        prepared.add_load(paired.rhs())?;
        Ok(prepared)
    }

    pub fn natural_facet(
        &self,
        normalized_cell: &AffineGeometryMap,
        facet: (
            &AffineGeometryMap,
            eqiora_meshing::EntityIncidence,
            &[usize],
        ),
        rule: &eqiora_meshing::QuadratureRule,
        point: &[f64],
        traction: [f64; 2],
        derivative: bool,
    ) -> Result<crate::form_compiler::region::RegionLinearization, Diagnostic> {
        let (normalized_facet, incidence, parent_vertices) = facet;
        let scalar = self
            .form
            .fields()
            .iter()
            .find(|layout| layout.field == self.scalar)
            .expect("bound scalar");
        let cell = self.geometry(normalized_cell)?;
        let facet = self.geometry(normalized_facet)?;
        self.form.natural_facet_action(
            self.vector,
            &cell,
            (&facet, incidence, parent_vertices),
            rule,
            point,
            derivative,
            |_, _| Ok(traction.iter().map(|value| scalar.scale * value).collect()),
        )
    }

    pub fn geometry(
        &self,
        normalized: &AffineGeometryMap,
    ) -> Result<AffineGeometryMap, Diagnostic> {
        AffineGeometryMap::new(
            normalized.reference_cell(),
            normalized.physical_dimension(),
            normalized
                .origin()
                .iter()
                .enumerate()
                .map(|(i, value)| self.origin[i] + self.length * value)
                .collect(),
            normalized
                .jacobian()
                .iter()
                .map(|value| self.length * value)
                .collect(),
        )
    }
}
