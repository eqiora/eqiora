use super::*;
use eqiora_schema::kernel::{ExprNode, FieldDef, RelationDef};
use num_complex::Complex64;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Embedding {
    pub map: ValueLiteral,
    pub steps: Vec<EmbeddingStep>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EmbeddingStep {
    pub relation: Id<kinds::Relation>,
    pub target: Id<kinds::Field>,
    pub coordinate: Id<kinds::Field>,
    pub coordinate_type: ValueType,
    pub map: ValueLiteral,
    target_operator: ValueLiteral,
    coordinate_operator: ValueLiteral,
}

pub(super) fn field_ids(relation: &RelationDef) -> Vec<Id<kinds::Field>> {
    let mut fields = Vec::new();
    for node in relation.expression().nodes() {
        if let ExprNode::Symbol(SymbolRef::Field(id)) = node
            && !fields.contains(id)
        {
            fields.push(*id);
        }
    }
    fields
}

impl Embedding {
    pub(super) fn lower(
        kernel: &KernelProgram,
        fields: &[&FieldDef],
        relations: &[&RelationDef],
        pencil: Id<kinds::Relation>,
        mode: Id<kinds::Field>,
        eigenvalue: Id<kinds::Field>,
    ) -> Result<Option<Self>, Diagnostic> {
        let mut remaining = relations
            .iter()
            .filter(|r| r.id() != pencil)
            .copied()
            .collect::<Vec<_>>();
        let mut visited = vec![mode, eigenvalue];
        let mut target = mode;
        let mut steps = Vec::new();
        let mut map = None;
        while !remaining.is_empty() {
            let matches = remaining
                .iter()
                .enumerate()
                .filter(|(_, r)| field_ids(r).contains(&target))
                .collect::<Vec<_>>();
            let [(index, relation)] = matches.as_slice() else {
                return Err(invalid(
                    "spectral coordinate embeddings require one unambiguous source chain",
                ));
            };
            let ids = field_ids(relation);
            let coordinates = ids
                .iter()
                .copied()
                .filter(|id| *id != target)
                .collect::<Vec<_>>();
            let [coordinate] = coordinates.as_slice() else {
                return Err(invalid(
                    "coordinate equality must connect two finite Fields",
                ));
            };
            if visited.contains(coordinate) {
                return Err(invalid(
                    "spectral coordinate embedding contains a cycle or eigenvalue dependence",
                ));
            }
            let field = |id| {
                fields
                    .iter()
                    .copied()
                    .find(|f| f.id() == id)
                    .ok_or_else(|| invalid("embedding refers to a missing Field"))
            };
            let step = lower_step(kernel, relation, field(target)?, field(*coordinate)?)?;
            map = Some(if let Some(previous) = map {
                compose(&previous, &step.map)?
            } else {
                step.map.clone()
            });
            target = *coordinate;
            visited.push(target);
            steps.push(step);
            let index = *index;
            remaining.remove(index);
        }
        if visited.len() != fields.len() {
            return Err(invalid(
                "spectral source contains a Field outside the original pencil and coordinate chain",
            ));
        }
        Ok(map.map(|map| Self { map, steps }))
    }

    pub(super) fn coordinate_type(&self) -> &ValueType {
        &self
            .steps
            .last()
            .expect("nonempty embedding")
            .coordinate_type
    }

    pub(in crate::numerical_admission::eigen) fn lift(
        &self,
        coordinate: &ValueLiteral,
    ) -> Result<LiftedMode, Diagnostic> {
        if coordinate.value_type() != self.coordinate_type() {
            return Err(invalid(
                "spectral candidate has the wrong admitted coordinate type",
            ));
        }
        let mut value = coordinate.clone();
        let mut fields = vec![(
            self.steps.last().expect("embedding").coordinate,
            value.clone(),
        )];
        let mut residual: f64 = 0.;
        for step in self.steps.iter().rev() {
            let (source, target) = step.map.value_type().map_bases().expect("embedding map");
            let k = source.extent() as usize;
            let ty = ValueType::coordinates(
                target,
                value.value_type().scalar_domain(),
                step.map
                    .value_type()
                    .dimension()
                    .mul(value.value_type().dimension())
                    .ok_or_else(|| invalid("lifted mode dimension overflowed"))?,
            )
            .map_err(|e| invalid(e.to_string()))?;
            let lifted = ValueLiteral::new(
                ty,
                (0..target.extent() as usize).map(|i| {
                    let entry: Complex64 = (0..k)
                        .map(|j| coefficient(&step.map, i * k + j) * coefficient(&value, j))
                        .sum();
                    (entry.re, entry.im)
                }),
            )
            .map_err(|e| invalid(e.to_string()))?;
            residual = residual.max(step.relative_residual(&lifted, &value)?);
            value = lifted;
            fields.push((step.target, value.clone()));
        }
        fields.reverse();
        Ok((fields, residual))
    }

    pub(super) fn first_quadratic(&self, matrix: &ValueLiteral) -> Result<f64, Diagnostic> {
        let (source, target) = self.map.value_type().map_bases().expect("embedding map");
        let (n, k) = (target.extent() as usize, source.extent() as usize);
        let mut value = Complex64::new(0., 0.);
        for i in 0..n {
            for j in 0..n {
                value += coefficient(&self.map, i * k).conj()
                    * coefficient(matrix, i * n + j)
                    * coefficient(&self.map, j * k);
            }
        }
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(invalid(
                "spectral orientation produced nonfinite arithmetic",
            ));
        }
        Ok(value.re)
    }
}

fn lower_step(
    kernel: &KernelProgram,
    relation: &RelationDef,
    target: &FieldDef,
    coordinate: &FieldDef,
) -> Result<EmbeddingStep, Diagnostic> {
    let target_type = target.value_type();
    let coordinate_type = coordinate.value_type();
    let target_basis = target_type
        .coordinate_basis()
        .ok_or_else(|| invalid("embedding target requires a finite basis"))?;
    let source_basis = coordinate_type
        .coordinate_basis()
        .ok_or_else(|| invalid("embedding coordinates require a finite basis"))?;
    if target_type.scalar_domain() != coordinate_type.scalar_domain() {
        return Err(invalid(
            "spectral embedding must retain a real or complex linear coordinate space",
        ));
    }
    let typed = kernel
        .typed_relation_residual(relation.id())
        .map_err(|mut errors| errors.remove(0))?;
    let root = typed.expression().roots()[0];
    let root_type = &typed
        .node_type(root)
        .ok_or_else(|| invalid("embedding root type is missing"))?
        .value_type;
    if root_type.coordinate_basis() != Some(target_basis)
        || root_type.scalar_domain() != target_type.scalar_domain()
    {
        return Err(invalid(
            "embedding equality must act in its target Field's exact basis",
        ));
    }
    let mut selected =
        ScalarSymbolCoordinate::for_value(SymbolRef::Field(target.id()), target_type)?;
    let output_size = selected.len();
    let input =
        ScalarSymbolCoordinate::for_value(SymbolRef::Field(coordinate.id()), coordinate_type)?;
    selected.extend_from_slice(&input);
    let rows = ComponentScalarization::lower(&typed)?;
    if rows.rows().len() != output_size {
        return Err(invalid(
            "embedding equality has an inconsistent target shape",
        ));
    }
    let bindings = super::bindings(kernel, &rows, &[target.id(), coordinate.id()])?;
    let mut left = Vec::new();
    let mut right = Vec::new();
    for row in rows.rows() {
        let affine = row.bind_affine(&selected, &bindings)?;
        if affine.offsets().iter().any(|value| *value != 0.) {
            return Err(invalid(
                "spectral coordinate embeddings must be homogeneous",
            ));
        }
        left.extend_from_slice(&affine.coefficients()[..output_size]);
        right.extend_from_slice(&affine.coefficients()[output_size..]);
    }
    let ty = |basis, dimension| {
        ValueType::linear_map(basis, target_basis, target_type.scalar_domain(), dimension)
            .map_err(|e| invalid(e.to_string()))
    };
    let left_type = ty(
        target_basis,
        root_type
            .dimension()
            .div(target_type.dimension())
            .ok_or_else(|| invalid("embedding dimension overflowed"))?,
    )?;
    let right_type = ty(
        source_basis,
        root_type
            .dimension()
            .div(coordinate_type.dimension())
            .ok_or_else(|| invalid("embedding dimension overflowed"))?,
    )?;
    let left = super::assemble(left_type, &left, output_size, 1.)?;
    let right = super::assemble(right_type, &right, input.len(), 1.)?;
    let n = target_basis.extent() as usize;
    let k = source_basis.extent() as usize;
    let mut entries = Vec::new();
    for i in 0..n {
        let diagonal = coefficient(&left, i * n + i);
        if diagonal == Complex64::new(0., 0.)
            || (0..n).any(|j| i != j && coefficient(&left, i * n + j) != Complex64::new(0., 0.))
        {
            return Err(invalid(
                "coordinate equality must explicitly define its target; coupled target elimination is not admitted",
            ));
        }
        for j in 0..k {
            let value = -coefficient(&right, i * k + j) / diagonal;
            entries.push((value.re, value.im));
        }
    }
    let map = ValueLiteral::new(
        ty(
            source_basis,
            target_type
                .dimension()
                .div(coordinate_type.dimension())
                .ok_or_else(|| invalid("embedding dimension overflowed"))?,
        )?,
        entries,
    )
    .map_err(|e| invalid(e.to_string()))?;
    Ok(EmbeddingStep {
        relation: relation.id(),
        target: target.id(),
        coordinate: coordinate.id(),
        coordinate_type: coordinate_type.clone(),
        map,
        target_operator: left,
        coordinate_operator: right,
    })
}

impl EmbeddingStep {
    fn relative_residual(
        &self,
        target: &ValueLiteral,
        coordinate: &ValueLiteral,
    ) -> Result<f64, Diagnostic> {
        let n = target
            .value_type()
            .shape()
            .component_count()
            .expect("typed target");
        let k = coordinate
            .value_type()
            .shape()
            .component_count()
            .expect("typed coordinate");
        let (mut left_norm, mut right_norm, mut residual) = (0_f64, 0_f64, 0_f64);
        for i in 0..n {
            let left: Complex64 = (0..n)
                .map(|j| coefficient(&self.target_operator, i * n + j) * coefficient(target, j))
                .sum();
            let right: Complex64 = (0..k)
                .map(|j| {
                    coefficient(&self.coordinate_operator, i * k + j) * coefficient(coordinate, j)
                })
                .sum();
            left_norm = left_norm.hypot(left.norm());
            right_norm = right_norm.hypot(right.norm());
            residual = residual.hypot((left + right).norm());
        }
        let scale = left_norm + right_norm;
        let relative = if scale == 0. && residual == 0. {
            0.
        } else {
            residual / scale
        };
        if !scale.is_finite() || !relative.is_finite() {
            return Err(invalid(
                "source coordinate equality verification produced nonfinite arithmetic",
            ));
        }
        Ok(relative)
    }
}

fn coefficient(value: &ValueLiteral, index: usize) -> Complex64 {
    let (re, im) = value.component(index).expect("typed finite shape");
    Complex64::new(re, im)
}

fn compose(left: &ValueLiteral, right: &ValueLiteral) -> Result<ValueLiteral, Diagnostic> {
    let (inner, target) = left.value_type().map_bases().expect("map");
    let (source, right_target) = right.value_type().map_bases().expect("map");
    if inner != right_target {
        return Err(invalid("coordinate chain has incompatible nominal bases"));
    }
    let ty = ValueType::linear_map(
        source,
        target,
        left.value_type().scalar_domain(),
        left.value_type()
            .dimension()
            .mul(right.value_type().dimension())
            .ok_or_else(|| invalid("embedding dimension overflowed"))?,
    )
    .map_err(|e| invalid(e.to_string()))?;
    let (n, k, m) = (
        target.extent() as usize,
        source.extent() as usize,
        inner.extent() as usize,
    );
    let mut entries = Vec::new();
    for i in 0..n {
        for j in 0..k {
            let value: Complex64 = (0..m)
                .map(|r| coefficient(left, i * m + r) * coefficient(right, r * k + j))
                .sum();
            entries.push((value.re, value.im));
        }
    }
    ValueLiteral::new(ty, entries).map_err(|e| invalid(e.to_string()))
}
