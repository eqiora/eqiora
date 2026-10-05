use super::*;
use eqiora_core::{ScalarDomain, ValueType};
use eqiora_ir::{ComponentScalarization, ScalarSymbolCoordinate};
use eqiora_schema::kernel::{FieldRole, KernelNode, SymbolRef};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct SourcePencil {
    pub relation: Id<kinds::Relation>,
    pub mode: Id<kinds::Field>,
    pub eigenvalue: Id<kinds::Field>,
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
        if fields.len() != 2
            || fields
                .iter()
                .any(|field| field.role() != FieldRole::Variable)
        {
            return Err(invalid(
                "finite Hermitian source requires exactly a mode Field and a real eigenvalue Field",
            ));
        }
        let mode = fields
            .iter()
            .find(|field| field.value_type().coordinate_basis().is_some())
            .ok_or_else(|| invalid("spectral mode requires an exact finite coordinate basis"))?;
        let eigenvalue = fields
            .iter()
            .find(|field| field.id() != mode.id())
            .ok_or_else(|| invalid("spectral roles must be distinct"))?;
        let mode_type = mode.value_type();
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
        if relations.len() != 1
            || relations[0].is_initial()
            || relations[0].has_constraints()
            || relations[0].equation_sides().count() != 1
        {
            return Err(invalid(
                "finite Hermitian source requires one complete equality; additional constraints need an admitted constrained-space projection",
            ));
        }
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
        let relation = relations[0].id();
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
        let mut bindings = HashMap::new();
        for coordinate in rows.rows().iter().flat_map(|row| row.symbols()) {
            match coordinate.symbol() {
                SymbolRef::Field(id) if id == mode.id() || id == eigenvalue.id() => {}
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
        let bindings = bindings.into_iter().collect::<Vec<_>>();
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
        // Every positive-definite metric has a strictly positive first diagonal;
        // this fixes only the whole-equation sign, never individual rows or a
        // regularizing shift. Full Hermitian and positive-pivot admission still
        // follows in HermitianEigenproblem, including all remaining coordinates.
        let orientation = if c[0] > 0. { -1. } else { 1. };
        let operator = assemble(a_type, &a, selected.len(), orientation)?;
        let metric = assemble(b_type, &c, selected.len(), -orientation)?;
        Ok(Self {
            relation,
            mode: mode.id(),
            eigenvalue: eigenvalue.id(),
            operator,
            metric,
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
    let n = if complex {
        real_dimension / 2
    } else {
        real_dimension
    };
    let mut entries = Vec::new();
    for row in 0..n {
        for column in 0..n {
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
