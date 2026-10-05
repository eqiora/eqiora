use super::*;
use eqiora_core::{ScalarDomain, ValueType};
use eqiora_ir::{ComponentScalarization, ScalarSymbolCoordinate};
use eqiora_schema::kernel::{FieldRole, KernelNode, SymbolRef};
use std::collections::HashMap;

mod embedding;
use embedding::Embedding;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct SourcePencil {
    pub relation: Id<kinds::Relation>,
    pub mode: Id<kinds::Field>,
    pub eigenvalue: Id<kinds::Field>,
    pub operator: ValueLiteral,
    pub metric: ValueLiteral,
    pub projected: Option<ProjectedPencil>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ProjectedPencil {
    pub embedding: Embedding,
    pub operator: ValueLiteral,
    pub metric: ValueLiteral,
}

impl SourcePencil {
    pub(super) fn lower(kernel: &KernelProgram) -> Result<Self, Diagnostic> {
        let fields = kernel
            .nodes()
            .filter_map(|node| match node {
                KernelNode::Field(field) => Some(field),
                _ => None,
            })
            .collect::<Vec<_>>();
        if fields
            .iter()
            .any(|field| field.role() != FieldRole::Variable)
        {
            return Err(invalid("spectral Fields must have Variable roles"));
        }
        let scalar_fields = fields
            .iter()
            .filter(|field| field.value_type().coordinate_basis().is_none())
            .collect::<Vec<_>>();
        let [eigenvalue] = scalar_fields.as_slice() else {
            return Err(invalid(
                "spectral source requires one real scalar eigenvalue Field",
            ));
        };
        let lambda_type = eigenvalue.value_type();
        if lambda_type
            != &ValueType::scalar(ScalarDomain::Real, lambda_type.dimension())
                .map_err(|e| invalid(e.to_string()))?
        {
            return Err(invalid(
                "spectral eigenvalue Field must be a real invariant scalar",
            ));
        }
        let relations = kernel
            .nodes()
            .filter_map(|node| match node {
                KernelNode::Relation(relation) => Some(relation),
                _ => None,
            })
            .collect::<Vec<_>>();
        if relations
            .iter()
            .any(|r| r.is_initial() || r.has_constraints() || r.equation_sides().count() != 1)
        {
            return Err(invalid(
                "spectral source requires complete global equalities",
            ));
        }
        let pencils = relations
            .iter()
            .filter(|r| embedding::field_ids(r).contains(&eigenvalue.id()))
            .collect::<Vec<_>>();
        let [pencil] = pencils.as_slice() else {
            return Err(invalid(
                "spectral source requires one original eigenvalue-dependent equality",
            ));
        };
        let mode_fields = fields
            .iter()
            .filter(|f| f.id() != eigenvalue.id() && embedding::field_ids(pencil).contains(&f.id()))
            .collect::<Vec<_>>();
        let [mode] = mode_fields.as_slice() else {
            return Err(invalid(
                "original spectral equality requires one finite mode Field",
            ));
        };
        let mode_type = mode.value_type();
        let embedding = Embedding::lower(
            kernel,
            &fields,
            &relations,
            pencil.id(),
            mode.id(),
            eigenvalue.id(),
        )?;
        if kernel.nodes().any(|node| {
            matches!(
                node,
                KernelNode::Domain(_)
                    | KernelNode::Port(_)
                    | KernelNode::Connection(_)
                    | KernelNode::ClockDomain(_)
            )
        }) {
            return Err(invalid(
                "finite Hermitian admission does not omit spatial, port or clocked source meaning",
            ));
        }
        let relation = pencil.id();
        let typed = kernel.typed_relation_residual(relation).map_err(|errors| {
            errors
                .into_iter()
                .next()
                .unwrap_or_else(|| invalid("spectral relation typing failed"))
        })?;
        let root = *typed
            .expression()
            .roots()
            .first()
            .ok_or_else(|| invalid("spectral residual is empty"))?;
        let root_type = typed
            .node_type(root)
            .ok_or_else(|| invalid("spectral residual type is unavailable"))?;
        let basis = mode_type
            .coordinate_basis()
            .expect("selected coordinate Field");
        if typed.expression().roots().len() != 1
            || root_type.support.is_some()
            || root_type.value_type.coordinate_basis() != Some(basis)
            || root_type.value_type.scalar_domain() != mode_type.scalar_domain()
        {
            return Err(invalid(
                "spectral residual must act in the mode's exact global basis and scalar domain",
            ));
        }
        let a_dimension = root_type
            .value_type
            .dimension()
            .div(mode_type.dimension())
            .ok_or_else(|| invalid("spectral operator dimension overflowed"))?;
        let b_dimension = a_dimension
            .div(lambda_type.dimension())
            .ok_or_else(|| invalid("spectral metric dimension overflowed"))?;
        let a_type = ValueType::linear_map(basis, basis, mode_type.scalar_domain(), a_dimension)
            .map_err(|e| invalid(e.to_string()))?;
        let b_type = ValueType::linear_map(basis, basis, mode_type.scalar_domain(), b_dimension)
            .map_err(|e| invalid(e.to_string()))?;
        let (expected_lambda, expected_mode) = a_type
            .hermitian_eigenpair_types(&b_type)
            .map_err(|e| invalid(e.to_string()))?;
        if &expected_lambda != lambda_type || &expected_mode != mode_type {
            return Err(invalid(
                "source spectral types must express the eigenvalue units and dimensionless B-unit normalization of the mode",
            ));
        }
        let selected = ScalarSymbolCoordinate::for_value(SymbolRef::Field(mode.id()), mode_type)?;
        let spectral =
            ScalarSymbolCoordinate::for_value(SymbolRef::Field(eigenvalue.id()), lambda_type)?
                .remove(0);
        let rows = ComponentScalarization::lower(&typed)?;
        if rows.rows().len() != selected.len() {
            return Err(invalid("spectral coordinate action must be square"));
        }
        let bindings = bindings(kernel, &rows, &[mode.id(), eigenvalue.id()])?;
        let mut a = Vec::new();
        let mut c = Vec::new();
        for row in rows.rows() {
            let (constant, spectral_part) =
                row.bind_affine_pencil(&selected, &spectral, &bindings)?;
            a.extend_from_slice(constant.coefficients());
            c.extend_from_slice(spectral_part.coefficients());
        }
        // An equality has no privileged left/right orientation. Choose the
        // representative (s A, -s C) whose metric can be positive definite.
        // Its first admitted coordinate has a strictly positive quadratic form.
        // This fixes only the whole-equation sign, never individual rows or a
        // regularizing shift. Full Hermitian input checks and positive pivots
        // on every admitted coordinate still follow in HermitianEigenproblem.
        let unoriented_c = assemble(b_type.clone(), &c, selected.len(), 1.)?;
        let first = if let Some(embedding) = &embedding {
            embedding.first_quadratic(&unoriented_c)?
        } else {
            c[0]
        };
        let orientation = if first > 0. { -1. } else { 1. };
        let operator = assemble(a_type, &a, selected.len(), orientation)?;
        let metric = assemble(b_type, &c, selected.len(), -orientation)?;
        let projected = embedding
            .map(|embedding| {
                let (operator, metric) =
                    HermitianEigenproblem::pullback(&operator, &metric, &embedding.map)?;
                let (_, expected_coordinate) = operator
                    .value_type()
                    .hermitian_eigenpair_types(metric.value_type())
                    .map_err(|e| invalid(e.to_string()))?;
                if &expected_coordinate != embedding.coordinate_type() {
                    return Err(invalid(
                        "admitted coordinate Field has incorrect spectral normalization units",
                    ));
                }
                Ok(ProjectedPencil {
                    embedding,
                    operator,
                    metric,
                })
            })
            .transpose()?;
        Ok(Self {
            relation,
            mode: mode.id(),
            eigenvalue: eigenvalue.id(),
            operator,
            metric,
            projected,
        })
    }
}

fn assemble(
    ty: ValueType,
    rows: &[f64],
    real_dimension: usize,
    sign: f64,
) -> Result<ValueLiteral, Diagnostic> {
    let complex = ty.scalar_domain() == ScalarDomain::Complex;
    let (source, target) = ty.map_bases().expect("typed map");
    let n = target.extent() as usize;
    let columns = source.extent() as usize;
    let mut entries = Vec::new();
    for row in 0..n {
        for column in 0..columns {
            let (real, imaginary) = if complex {
                let real = rows[(2 * row) * real_dimension + 2 * column];
                let imaginary = rows[(2 * row + 1) * real_dimension + 2 * column];
                if rows[(2 * row) * real_dimension + 2 * column + 1] != -imaginary
                    || rows[(2 * row + 1) * real_dimension + 2 * column + 1] != real
                {
                    return Err(invalid(
                        "spectral action is real-linear but not complex-linear; conjugate-dependent pencils are not Hermitian eigenproblems",
                    ));
                }
                (real, imaginary)
            } else {
                (rows[row * real_dimension + column], 0.)
            };
            entries.push((sign * real, sign * imaginary));
        }
    }
    ValueLiteral::new(ty, entries).map_err(|e| invalid(e.to_string()))
}

fn bindings(
    kernel: &KernelProgram,
    rows: &ComponentScalarization,
    fields: &[Id<kinds::Field>],
) -> Result<Vec<(ScalarSymbolCoordinate, f64)>, Diagnostic> {
    let mut bindings = HashMap::new();
    for coordinate in rows.rows().iter().flat_map(|row| row.symbols()) {
        match coordinate.symbol() {
            SymbolRef::Field(id) if fields.contains(&id) => {}
            SymbolRef::Parameter(id) => {
                let value = kernel
                    .typed_value(id.erase())
                    .ok_or_else(|| invalid("spectral Parameter value is missing"))?;
                let index = coordinate
                    .component_index()
                    .iter()
                    .zip(value.value_type().shape().extents())
                    .fold(0, |flat, (&index, extent)| {
                        flat * extent.get() as usize + index as usize
                    });
                let (real, imaginary) = value
                    .component(index)
                    .ok_or_else(|| invalid("spectral Parameter component is unavailable"))?;
                bindings.insert(
                    coordinate.clone(),
                    if coordinate.is_imaginary() {
                        imaginary
                    } else {
                        real
                    },
                );
            }
            _ => {
                return Err(invalid(
                    "spectral source contains an unresolved or unsupported dependency",
                ));
            }
        }
    }
    Ok(bindings.into_iter().collect())
}
