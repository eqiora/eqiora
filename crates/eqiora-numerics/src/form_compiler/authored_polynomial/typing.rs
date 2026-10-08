//! Live symbol types select channels and exact component coverage for comparison.
use super::*;
use eqiora_core::ScalarDomain;
use eqiora_schema::kernel::KernelNode;

pub(super) fn symbol_types(program: &KernelProgram) -> BTreeMap<String, ValueType> {
    program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Field(value) => {
                Some((value.id().ulid().to_string(), value.value_type().clone()))
            }
            KernelNode::Parameter(value) => {
                Some((value.id().ulid().to_string(), value.value_type().clone()))
            }
            _ => None,
        })
        .collect()
}
impl Context<'_> {
    pub(super) fn atom(&self, atom: Atom) -> Option<Polynomial> {
        let (symbol, indices, gradient) = match &atom {
            Atom::Field(id, indices)
            | Atom::Parameter(id, indices)
            | Atom::TraceField(id, indices) => (id.as_str(), indices, false),
            Atom::FieldGradient(id, indices) => (id.as_str(), indices, true),
            Atom::Test(indices) | Atom::TraceTest(indices) => (self.field, indices, false),
            Atom::TestGradient(indices) => (self.field, indices, true),
            Atom::Measure(_) | Atom::Coordinate(..) => return Some(Polynomial::atom(atom)),
        };
        let ty = self.symbols.get(symbol)?;
        if ty.array_rank() != 0
            || !matches!(
                ty.scalar_domain(),
                ScalarDomain::Real | ScalarDomain::Complex
            )
        {
            return None;
        }
        let mut shape = ty
            .shape()
            .extents()
            .iter()
            .map(|extent| extent.get() as usize)
            .collect::<Vec<_>>();
        if gradient {
            shape.push(self.dimensions);
        }
        if shape.len() != indices.len()
            || shape
                .iter()
                .zip(indices)
                .any(|(extent, index)| index >= extent)
        {
            return None;
        }
        Some(Polynomial::symbol(
            atom,
            ty.scalar_domain() == ScalarDomain::Complex,
        ))
    }
    pub(super) fn shape(&mut self, value: &E, depth: usize) -> Option<Vec<usize>> {
        self.step(depth)?;
        match value {
            E::Components { shape, values, .. } => {
                let shape = shape.iter().map(|n| *n as usize).collect::<Vec<_>>();
                let count = shape.iter().try_fold(1usize, |a, b| a.checked_mul(*b))?;
                (shape.len() == 2 && !shape.contains(&0) && count == values.len()).then_some(shape)
            }
            E::Component { value, indices } => {
                let shape = self.shape(value, depth + 1)?;
                (shape.len() == indices.len()
                    && shape.iter().zip(indices).all(|(n, i)| (*i as usize) < *n))
                .then(Vec::new)
            }
            E::Apply { left, right } => {
                let matrix = self.shape(left, depth + 1)?;
                let vector = self.shape(right, depth + 1)?;
                match (matrix.as_slice(), vector.as_slice()) {
                    ([rows, cols], [n]) if cols == n => Some(vec![*rows]),
                    _ => None,
                }
            }
            E::Field { ulid } | E::Parameter { ulid } => {
                let ty = self.symbols.get(ulid)?;
                (ty.array_rank() == 0).then(|| {
                    ty.shape()
                        .extents()
                        .iter()
                        .map(|extent| extent.get() as usize)
                        .collect()
                })
            }
            E::Test { field_ulid } if field_ulid == self.field => self.shape(
                &E::Field {
                    ulid: field_ulid.clone(),
                },
                depth + 1,
            ),
            E::Direction { name, field_ulid } if name == self.name && field_ulid == self.field => {
                self.shape(
                    &E::Field {
                        ulid: field_ulid.clone(),
                    },
                    depth + 1,
                )
            }
            E::Trace { value } | E::Conjugate { value } | E::Neg { value } => {
                self.shape(value, depth + 1)
            }
            E::Gradient { value } => {
                let mut shape = self.shape(value, depth + 1)?;
                shape.push(self.dimensions);
                Some(shape)
            }
            E::Number { .. }
            | E::Rational { .. }
            | E::Coordinate { .. }
            | E::Dot { .. }
            | E::Inner { .. } => Some(vec![]),
            E::Complex { real, imag } => (self.shape(real, depth + 1)?.is_empty()
                && self.shape(imag, depth + 1)?.is_empty())
            .then(Vec::new),
            E::Mul { left, right } => {
                let a = self.shape(left, depth + 1)?;
                let b = self.shape(right, depth + 1)?;
                if a.is_empty() {
                    Some(b)
                } else if b.is_empty() {
                    Some(a)
                } else {
                    None
                }
            }
            E::Add { left, right } | E::Sub { left, right } => {
                let a = self.shape(left, depth + 1)?;
                (a == self.shape(right, depth + 1)?).then_some(a)
            }
            _ => None,
        }
    }
    pub(super) fn contract(
        &mut self,
        left: &E,
        right: &E,
        conjugated: bool,
        depth: usize,
    ) -> Option<Polynomial> {
        let shape = self.shape(left, depth + 1)?;
        if shape != self.shape(right, depth + 1)? {
            return None;
        }
        match shape.as_slice() {
            [] if conjugated => self
                .scalar(left, depth + 1)?
                .conjugate()
                .ok()?
                .checked_mul(&self.scalar(right, depth + 1)?)
                .ok(),
            [extent] => {
                self.remaining = self.remaining.checked_sub(*extent)?;
                let mut sum = Polynomial::constant(ExactRational::integer(0));
                for axis in 0..*extent {
                    let left = self.vector(left, axis, depth + 1)?;
                    let left = if conjugated {
                        left.conjugate().ok()?
                    } else {
                        left
                    };
                    let term = left
                        .checked_mul(&self.vector(right, axis, depth + 1)?)
                        .ok()?;
                    sum = sum.checked_add(&term).ok()?;
                }
                Some(sum)
            }
            _ => None,
        }
    }
}
