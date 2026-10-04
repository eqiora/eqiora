//! A numerical tensor grid binds exact dimensioned coordinate factors, without a physical frame.
use super::{Diagnostic, invalid};
use eqiora_artifact::CartesianMeshEnvelopeV1;
use eqiora_core::{DimExponents, DynQuantity, Id, entity::kinds};
use eqiora_meshing::{CartesianMesh, MeshTopology};
use eqiora_schema::kernel::{AxisBounds, DomainKind, KernelNode, typing::SpatialSupport};
use eqiora_sem::KernelProgram;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use ulid::Ulid;

mod diffusion;
mod diffusion_plan;
mod equations;
mod observe;
mod plan;
mod polynomial;
mod projection;
pub(super) use equations::CellEquations;
pub(super) use plan::{execute, portable};
pub(super) use projection::CellProjection;
mod sampling;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CoordinateSource {
    domain: String,
    factors: Vec<Factor>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Factor {
    domain: String,
    dimension: [(i32, i32); 7],
    lower: f64,
    upper: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct CoordinateGrid {
    pub(super) source: CoordinateSource,
    pub(super) mesh: CartesianMeshEnvelopeV1,
}

impl CoordinateSource {
    pub(super) fn digest(&self) -> Result<eqiora_artifact::ArtifactDigest, Diagnostic> {
        use sha2::{Digest, Sha256};
        let bytes = serde_json::to_vec(self)
            .map_err(|error| invalid(format!("cannot encode coordinate factor source: {error}")))?;
        let mut hash = Sha256::new();
        hash.update(b"eqiora.coordinate-factor-source/v1\0");
        hash.update(bytes);
        Ok(eqiora_artifact::ArtifactDigest::from_sha256(
            hash.finalize().into(),
        ))
    }

    fn from_program(
        program: &KernelProgram,
        domain: Id<kinds::Domain>,
    ) -> Result<Self, Diagnostic> {
        let Some(SpatialSupport::Coordinates { factors, .. }) = program.spatial_support(domain)
        else {
            return Err(invalid(
                "coordinate grid requires an exact coordinate Domain",
            ));
        };
        let factors = factors
            .iter()
            .map(|(factor, _, _)| {
                let Some(KernelNode::Domain(definition)) = program.node(*factor) else {
                    return Err(invalid("coordinate grid factor is outside its Model"));
                };
                let DomainKind::CoordinateInterval { bounds } = definition.kind() else {
                    return Err(invalid("coordinate grid requires bounded interval factors"));
                };
                Ok(Factor {
                    domain: definition.id().ulid().to_string(),
                    dimension: bounds.lower().dim().exponents(),
                    lower: bounds.lower().value(),
                    upper: bounds.upper().value(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            domain: domain.ulid().to_string(),
            factors,
        })
    }

    fn validate(&self) -> Result<(), Diagnostic> {
        parse_domain(&self.domain)?;
        if self.factors.is_empty() {
            return Err(invalid("coordinate grid source requires nonempty factors"));
        }
        let mut ids = BTreeSet::new();
        for factor in &self.factors {
            if !ids.insert(parse_domain(&factor.domain)?.erase()) {
                return Err(invalid("coordinate grid repeats an exact factor"));
            }
            let dimension = DimExponents::from_rationals(factor.dimension)
                .filter(|dimension| dimension.exponents() == factor.dimension)
                .ok_or_else(|| invalid("coordinate grid has noncanonical factor units"))?;
            AxisBounds::new(
                DynQuantity::new(factor.lower, dimension),
                DynQuantity::new(factor.upper, dimension),
            )?;
        }
        Ok(())
    }

    pub(super) fn require_program(&self, program: &KernelProgram) -> Result<(), Diagnostic> {
        let expected = Self::from_program(program, parse_domain(&self.domain)?)?;
        if self != &expected {
            return Err(invalid(
                "coordinate grid source differs from the exact Model factors, units or bounds",
            ));
        }
        Ok(())
    }
}

impl CoordinateGrid {
    pub(super) fn new(
        program: &KernelProgram,
        domain: Id<kinds::Domain>,
        cells: &[usize],
    ) -> Result<Self, Diagnostic> {
        let source = CoordinateSource::from_program(program, domain)?;
        source.validate()?;
        let bounds = source
            .factors
            .iter()
            .map(|factor| [factor.lower, factor.upper])
            .collect::<Vec<_>>();
        let mesh = CartesianMesh::uniform(&bounds, cells)?;
        Self::from_parts(source, CartesianMeshEnvelopeV1::from_mesh(&mesh)?)
    }

    pub(super) fn from_parts(
        source: CoordinateSource,
        mesh: CartesianMeshEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        source.validate()?;
        if mesh.mesh().topological_dimension() != source.factors.len() {
            return Err(invalid(
                "coordinate grid dimension differs from its exact factor inventory",
            ));
        }
        for (index, factor) in source.factors.iter().enumerate() {
            let axis = mesh
                .mesh()
                .axis_coordinates(index)
                .ok_or_else(|| invalid("coordinate grid omitted a factor axis"))?;
            if axis.first() != Some(&factor.lower) || axis.last() != Some(&factor.upper) {
                return Err(invalid(
                    "coordinate grid endpoints differ from its exact factor bounds",
                ));
            }
        }
        Ok(Self { source, mesh })
    }
}

fn parse_domain(value: &str) -> Result<Id<kinds::Domain>, Diagnostic> {
    let id = Ulid::from_string(value)
        .map_err(|_| invalid("coordinate grid contains an invalid Domain identity"))?;
    if id.to_string() != value {
        return Err(invalid(
            "coordinate grid contains a noncanonical Domain identity",
        ));
    }
    Ok(Id::from_ulid(id))
}
