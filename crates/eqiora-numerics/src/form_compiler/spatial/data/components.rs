//! Typed real coordinates shared by prescribed expressions and coefficient Fields.
use super::*;
use eqiora_core::{ScalarDomain, ValueType};

#[derive(Debug, Clone, PartialEq)]
pub(in crate::form_compiler) struct Components<S: Coefficient> {
    value_type: ValueType,
    storage: Storage<S>,
}

#[derive(Debug, Clone, PartialEq)]
enum Storage<S: Coefficient> {
    Zero,
    // Preserve scalar composition, including its degree and coefficient identity.
    Scalar(Data<S>),
    // Component-major coordinates, with adjacent real/imaginary pairs when complex.
    Coordinates(Vec<Data<S>>),
}

impl<S: Coefficient> Components<S> {
    fn parts(value_type: &ValueType) -> Result<usize, Diagnostic> {
        match value_type.scalar_domain() {
            ScalarDomain::Real => Ok(1),
            ScalarDomain::Complex => Ok(2),
            _ => Err(super::super::invalid(
                "prescribed data requires real or complex components",
            )),
        }
    }

    pub(super) fn new(
        value_type: ValueType,
        coordinates: Vec<Data<S>>,
    ) -> Result<Self, Diagnostic> {
        let count = value_type
            .shape()
            .component_count()
            .and_then(|count| count.checked_mul(Self::parts(&value_type).ok()?))
            .ok_or_else(|| {
                super::super::invalid("prescribed coordinate count is not representable")
            })?;
        if coordinates.len() != count {
            return Err(super::super::invalid(
                "prescribed coordinates differ from their complete value type",
            ));
        }
        Ok(Self {
            value_type,
            storage: Storage::Coordinates(coordinates),
        })
    }

    pub(in crate::form_compiler) fn prescribed(
        context: &Context<'_, S>,
        value_type: &ValueType,
        expression: Option<ExprId>,
    ) -> Result<Self, Diagnostic> {
        Self::parts(value_type)?;
        let Some(id) = expression else {
            return Ok(Self {
                value_type: value_type.clone(),
                storage: Storage::Zero,
            });
        };
        let typed = super::super::super::scalar::typed_relation(context.program, context.owner)?;
        let source_type = &typed
            .node_type(id)
            .ok_or_else(|| super::super::invalid("missing coefficient type"))?
            .value_type;
        if source_type
            .scalar_domain()
            .common(value_type.scalar_domain())
            != Some(value_type.scalar_domain())
            || source_type
                .clone()
                .with_scalar_domain(value_type.scalar_domain())
                .as_ref()
                != Some(value_type)
        {
            return Err(super::super::invalid(
                "coefficient definition differs from its exact Field type",
            ));
        }
        // Preserve scalar composition's polynomial degree for quadrature selection.
        // Every component uses the same typed Parameter and coordinate inputs.
        if source_type.shape().is_scalar()
            && let Ok(data) = context.data(id, 0)
        {
            return Ok(Self {
                value_type: value_type.clone(),
                storage: Storage::Scalar(data),
            });
        }
        let mut result = pointwise::PointwiseData::lower(context, id, 0)?;
        if source_type.scalar_domain() != value_type.scalar_domain() {
            let zero = Data::constant(context.dimension, <S as From<f64>>::from(0.));
            if let Storage::Coordinates(coordinates) = result.storage {
                result.storage = Storage::Coordinates(
                    coordinates
                        .into_iter()
                        .flat_map(|real| [real, zero.clone()])
                        .collect(),
                );
            }
        }
        result.value_type = value_type.clone();
        Ok(result)
    }

    pub(in crate::form_compiler) fn coordinate(
        &self,
        index: &[u32],
        imaginary: bool,
        dimension: usize,
    ) -> Result<Data<S>, Diagnostic> {
        let flat = self.value_type.shape().flat_index(index).ok_or_else(|| {
            super::super::invalid("prescribed coordinate differs from its exact Field shape")
        })?;
        self.part(flat, imaginary, dimension)
    }

    fn part(&self, flat: usize, imaginary: bool, dimension: usize) -> Result<Data<S>, Diagnostic> {
        let parts = Self::parts(&self.value_type)?;
        if imaginary && parts != 2 {
            return Err(super::super::invalid(
                "real prescribed data has no imaginary source coordinate",
            ));
        }
        let index = flat
            .checked_mul(parts)
            .and_then(|index| index.checked_add(usize::from(imaginary)))
            .ok_or_else(|| {
                super::super::invalid("prescribed coordinate offset is not representable")
            })?;
        match &self.storage {
            Storage::Zero => Ok(Data::constant(dimension, <S as From<f64>>::from(0.))),
            Storage::Scalar(data) if parts == 1 => Ok(data.clone()),
            Storage::Scalar(data) => Ok(Data(Arc::new(Node::Part {
                imaginary,
                data: data.clone(),
            }))),
            Storage::Coordinates(coordinates) => coordinates.get(index).cloned().ok_or_else(|| {
                super::super::invalid("prescribed component lies outside its exact Field")
            }),
        }
    }

    fn component(&self, flat: usize, dimension: usize) -> Result<Data<S>, Diagnostic> {
        let real = self.part(flat, false, dimension)?;
        if self.value_type.scalar_domain() == ScalarDomain::Real {
            return Ok(real);
        }
        let unit = S::imaginary_unit().ok_or_else(|| super::super::invalid("complex prescribed values require their complete numeric domain; select real coordinates explicitly"))?;
        Ok(real.add(
            self.part(flat, true, dimension)?
                .multiply(Data::constant(dimension, unit)),
        ))
    }

    pub(in crate::form_compiler) fn scalar(&self, dimension: usize) -> Result<Data<S>, Diagnostic> {
        if !self.value_type.shape().is_scalar() {
            return Err(super::super::invalid(
                "scalar coefficient requires an exact scalar value type",
            ));
        }
        if self.value_type.scalar_domain() == ScalarDomain::Complex && S::imaginary_unit().is_none()
        {
            return Err(super::super::invalid(
                "complex prescribed values require their complete numeric domain; select real coordinates explicitly",
            ));
        }
        match &self.storage {
            Storage::Scalar(data) => return Ok(data.clone()),
            Storage::Zero => return Ok(Data::constant(dimension, <S as From<f64>>::from(0.))),
            Storage::Coordinates(_) => {}
        }
        self.component(0, dimension)
    }

    pub(super) fn values(&self, dimension: usize) -> Result<Vec<Data<S>>, Diagnostic> {
        (0..self
            .value_type
            .shape()
            .component_count()
            .expect("checked value type"))
            .map(|flat| self.component(flat, dimension))
            .collect()
    }
}
