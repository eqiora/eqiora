use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{
    Diagnostic, DimExponents, FiniteBasis, Id, InvalidValueType, ScalarDomain, ValueType,
};

/// One nominal, nonempty ordered orthonormal basis of mathematical components.
/// This is independent of a Realization graph discretization Space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteSpaceDef {
    id: Id<kinds::FiniteSpace>,
    labels: Vec<String>,
    factors: Option<[FiniteBasis; 2]>,
}

impl FiniteSpaceDef {
    /// Define an ordered basis with unique, nonempty labels.
    pub fn new(
        id: Id<kinds::FiniteSpace>,
        labels: impl IntoIterator<Item = String>,
    ) -> Result<Self, Diagnostic> {
        let labels: Vec<_> = labels.into_iter().collect();
        let mut seen = std::collections::BTreeSet::new();
        if labels.is_empty()
            || u32::try_from(labels.len()).is_err()
            || labels
                .iter()
                .any(|label| label.trim().is_empty() || !seen.insert(label))
        {
            return Err(Diagnostic::error(
                codes::INVALID_KERNEL_DEFINITION,
                "finite space requires a nonempty ordered basis with unique nonempty labels",
            ));
        }
        Ok(Self {
            id,
            labels,
            factors: None,
        })
    }
    /// Alias an ordered tensor product of two atomic primal orthonormal bases.
    /// Alias identity is retained by the declaration; coordinate type equality uses factors.
    pub fn product(
        id: Id<kinds::FiniteSpace>,
        left: FiniteBasis,
        right: FiniteBasis,
    ) -> Result<Self, Diagnostic> {
        FiniteBasis::product(left, right).map_err(|error| {
            Diagnostic::error(codes::INVALID_KERNEL_DEFINITION, error.to_string())
        })?;
        if left.is_dual() || right.is_dual() {
            return Err(Diagnostic::error(
                codes::INVALID_KERNEL_DEFINITION,
                "product declarations require primal atomic factors",
            ));
        }
        Ok(Self {
            id,
            labels: Vec::new(),
            factors: Some([left, right]),
        })
    }
    /// Ordered product factors; atomic label declarations have none.
    pub const fn factors(&self) -> Option<[FiniteBasis; 2]> {
        self.factors
    }
    /// Exact nominal identity, never derived from labels or cardinality.
    #[must_use]
    pub const fn id(&self) -> Id<kinds::FiniteSpace> {
        self.id
    }
    /// Ordered basis labels.
    #[must_use]
    pub fn labels(&self) -> Option<&[String]> {
        self.factors.is_none().then_some(&self.labels)
    }
    /// Exact primal basis, preserving declaration identity and label order.
    #[must_use]
    pub fn basis(&self) -> FiniteBasis {
        match self.factors {
            Some([left, right]) => FiniteBasis::product(left, right).expect("checked product"),
            None => FiniteBasis::new(self.id, self.labels.len() as u32).expect("checked basis"),
        }
    }
    /// Numeric coordinates using this declaration's exact cardinality.
    pub fn coordinates(
        &self,
        domain: ScalarDomain,
        dimension: DimExponents,
    ) -> Result<ValueType, InvalidValueType> {
        ValueType::coordinates(self.basis(), domain, dimension)
    }
    /// Nonnegative count type using this declaration's exact cardinality.
    pub fn counts(&self) -> Result<ValueType, InvalidValueType> {
        if self.factors.is_some() {
            return Err(InvalidValueType::FiniteSpaceType);
        }
        ValueType::counts(self.id, self.labels.len() as u32)
    }
}

impl From<FiniteSpaceDef> for super::KernelNode {
    fn from(value: FiniteSpaceDef) -> Self {
        Self::FiniteSpace(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basis_identity_order_and_cardinality_are_distinct() {
        let id = Id::new();
        let a = FiniteSpaceDef::new(id, ["A".into(), "B".into()]).unwrap();
        let b = FiniteSpaceDef::new(Id::new(), ["A".into(), "B".into()]).unwrap();
        let reversed = FiniteSpaceDef::new(id, ["B".into(), "A".into()]).unwrap();
        assert_ne!(a, reversed);
        assert_ne!(a.counts().unwrap(), b.counts().unwrap());
        assert_ne!(
            a.counts().unwrap(),
            a.coordinates(
                eqiora_core::ScalarDomain::Integer,
                eqiora_core::DimExponents::DIMENSIONLESS
            )
            .expect("integer coordinates")
        );
        assert_eq!(a.counts().unwrap().shape().extents()[0].get(), 2);
        assert_eq!(a.counts().unwrap().array_rank(), 0);
        assert!(FiniteSpaceDef::new(id, []).is_err());
        assert!(FiniteSpaceDef::new(id, ["A".into(), "A".into()]).is_err());
        assert!(FiniteSpaceDef::new(id, [" ".into()]).is_err());
    }
}
