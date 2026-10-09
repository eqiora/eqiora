use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id, ValueLiteral, ValueType};

/// Closed, ordered alternatives with declaration identity shared across occurrences.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumDef {
    id: Id<kinds::Enum>,
    members: Vec<String>,
}
impl EnumDef {
    /// Reject empty, duplicate or invalid member names before publishing a declaration.
    pub fn new(
        id: Id<kinds::Enum>,
        members: impl IntoIterator<Item = String>,
    ) -> Result<Self, Diagnostic> {
        let mut checked = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for member in members {
            if !member_name(&member)
                || !seen.insert(member.clone())
                || checked.len() >= ValueType::MAX_ENUM_MEMBERS as usize
            {
                return Err(invalid(
                    "enum requires unique identifier members within the tag bound",
                ));
            }
            checked
                .try_reserve(1)
                .map_err(|_| invalid("enum member allocation failed"))?;
            checked.push(member);
        }
        if checked.is_empty() {
            return Err(invalid("enum requires at least one member"));
        }
        Ok(Self {
            id,
            members: checked,
        })
    }
    pub const fn id(&self) -> Id<kinds::Enum> {
        self.id
    }
    pub fn members(&self) -> &[String] {
        &self.members
    }
    pub fn value_type(&self) -> ValueType {
        ValueType::enumeration(self.id, self.members.len() as u32).expect("checked enum")
    }
    pub fn value(&self, tag: u32) -> Result<ValueLiteral, Diagnostic> {
        ValueLiteral::enum_value(self.value_type(), tag)
            .map_err(|_| invalid("enum member tag is outside its declaration"))
    }
}
fn member_name(value: &str) -> bool {
    let mut bytes = value.bytes();
    matches!(bytes.next(), Some(first) if first.is_ascii_alphabetic() || first == b'_')
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn invalid(message: &str) -> Diagnostic {
    Diagnostic::error(codes::INVALID_KERNEL_DEFINITION, message)
}
impl From<EnumDef> for super::KernelNode {
    fn from(value: EnumDef) -> Self {
        Self::Enum(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::typing::{ExpressionType, RootContract, TypedResidual};
    use crate::kernel::{ComparisonOp, ExprDagBuilder};
    #[test]
    fn enum_declaration_and_expression_types_keep_exact_identity() {
        let id = Id::new();
        let definition = EnumDef::new(id, ["Off".into(), "On".into()]).unwrap();
        assert!(EnumDef::new(id, []).is_err());
        assert!(EnumDef::new(id, ["Off".into(), "Off".into()]).is_err());
        assert!(definition.value(2).is_err());
        for invalid in ["", " ", "bad name", "A.B", "1st", "温度"] {
            assert!(EnumDef::new(id, [invalid.into()]).is_err(), "{invalid}");
        }
        assert!(EnumDef::new(id, ["_Off".into(), "On2".into()]).is_ok());

        let ty = ExpressionType::<()>::new(definition.value_type(), None);
        assert!(ty.clone().compare(ComparisonOp::Equal, ty.clone()).is_ok());
        assert!(ty.clone().compare(ComparisonOp::Less, ty.clone()).is_err());
        assert!(
            ty.clone()
                .compare(
                    ComparisonOp::Equal,
                    ExpressionType::new(
                        EnumDef::new(Id::new(), ["Off".into(), "On".into()])
                            .unwrap()
                            .value_type(),
                        None
                    )
                )
                .is_err()
        );
        assert!(crate::kernel::typing::additive(&ty, &ty).is_err());
        let mut builder = ExprDagBuilder::new();
        let a = builder.constant(definition.value(0).unwrap()).unwrap();
        let b = builder.constant(definition.value(1).unwrap()).unwrap();
        let condition = builder.compare(ComparisonOp::Equal, a, b).unwrap();
        let selected = builder.select(condition, a, b).unwrap();
        assert!(
            TypedResidual::infer(
                builder.finish([selected, a]).unwrap(),
                None,
                |_| None,
                RootContract::EquationSides,
                |_| Err::<ExpressionType<()>, ()>(())
            )
            .is_ok()
        );
    }
}
