//! Nominal heterogeneous products over typed values and field symbols.
use eqiora_core::{Diagnostic, Id, ValueType, diagnostic::codes, entity::kinds};
use std::collections::BTreeSet;

/// A closed record declaration with ordered member types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordDef {
    id: Id<kinds::Record>,
    members: Vec<(String, ValueType)>,
}
impl RecordDef {
    /// Maximum number of members in one flat record.
    pub const MAX_MEMBERS: usize = 65_536;

    /// Construct a non-empty record declaration, checking all members.
    pub fn new(
        id: Id<kinds::Record>,
        members: Vec<(String, ValueType)>,
    ) -> Result<Self, Diagnostic> {
        if members.is_empty() || members.len() > Self::MAX_MEMBERS {
            return Err(invalid("record requires between 1 and 65536 members"));
        }
        let mut seen = BTreeSet::new();
        for (name, _) in &members {
            let mut bytes = name.bytes();
            if name == "_"
                || !matches!(bytes.next(), Some(first) if first.is_ascii_alphabetic() || first == b'_')
                || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
                || !seen.insert(name)
            {
                return Err(invalid(
                    "record requires unique non-wildcard identifier members",
                ));
            }
        }
        Ok(Self { id, members })
    }
    /// Declaration identity, independent of member display names.
    pub const fn id(&self) -> Id<kinds::Record> {
        self.id
    }
    /// Member names and their mathematical types, in order.
    pub fn members(&self) -> &[(String, ValueType)] {
        &self.members
    }
}

/// One record instance: typed member expressions in declaration order.
/// Static expressions keep their Parameter dependence; bus roots name Fields.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordInstanceDef {
    id: Id<kinds::RecordInstance>,
    definition: Id<kinds::Record>,
    expression: super::ExprDag,
}
impl RecordInstanceDef {
    /// Construct an instance from an expression DAG with one root per declared member.
    /// Whole-Model admission checks the member types and the static or bus timing rules.
    pub fn new(
        id: Id<kinds::RecordInstance>,
        definition: Id<kinds::Record>,
        expression: super::ExprDag,
    ) -> Result<Self, Diagnostic> {
        if expression.roots().is_empty() || expression.roots().len() > RecordDef::MAX_MEMBERS {
            return Err(invalid(
                "record instance requires between 1 and 65536 member roots",
            ));
        }
        Ok(Self {
            id,
            definition,
            expression,
        })
    }
    /// Instance identity.
    pub const fn id(&self) -> Id<kinds::RecordInstance> {
        self.id
    }
    /// Record declaration this instance belongs to.
    pub const fn definition(&self) -> Id<kinds::Record> {
        self.definition
    }
    /// Shared expression DAG whose roots follow declaration member order.
    pub const fn expression(&self) -> &super::ExprDag {
        &self.expression
    }
}

fn invalid(message: &str) -> Diagnostic {
    Diagnostic::error(codes::INVALID_KERNEL_DEFINITION, message)
}
impl From<RecordDef> for super::KernelNode {
    fn from(value: RecordDef) -> Self {
        Self::Record(value)
    }
}
impl From<RecordInstanceDef> for super::KernelNode {
    fn from(value: RecordInstanceDef) -> Self {
        Self::RecordInstance(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, ScalarDomain};

    #[test]
    fn declarations_keep_nominal_identity_order_and_heterogeneous_types() {
        let volts = ValueType::scalar(
            ScalarDomain::Real,
            DimExponents::from_integers([1, 2, -3, -1, 0, 0, 0]).unwrap(),
        )
        .unwrap();
        let mode = super::super::EnumDef::new(Id::new(), ["Off".into(), "On".into()]).unwrap();
        let members = vec![
            ("voltage".into(), volts.clone()),
            ("mode".into(), mode.value_type()),
            ("valid".into(), ValueType::boolean()),
        ];
        let record = RecordDef::new(Id::new(), members.clone()).unwrap();
        assert_eq!(record.members()[0].1, volts);
        assert_eq!(record.members()[1].1, mode.value_type());
        assert_eq!(record.members()[2].1, ValueType::boolean());
        assert_ne!(
            record.id(),
            RecordDef::new(Id::new(), members).unwrap().id()
        );
        assert!(RecordDef::new(Id::new(), vec![]).is_err());
        assert!(
            RecordDef::new(
                Id::new(),
                vec![
                    ("x".into(), ValueType::boolean()),
                    ("x".into(), ValueType::boolean())
                ]
            )
            .is_err()
        );
        assert!(RecordDef::new(Id::new(), vec![("_".into(), ValueType::boolean())]).is_err());
    }
}
