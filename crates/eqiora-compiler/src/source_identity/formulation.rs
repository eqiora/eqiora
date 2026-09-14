//! Separate canonical identity for authored formulations.

use super::*;

const MAGIC: &[u8; 8] = b"EQIORAFM";
const CANONICAL_FORMULATION_VERSION: u16 = 2;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct AuthoredFormSourceIdentity([u8; 32]);

impl AuthoredFormSourceIdentity {
    pub(crate) fn from_component(component: &ComponentDecl) -> Result<Self, Diagnostic> {
        let limits = LocalSourceIdentityLimits::default();
        let mut budget = Budget::new(limits);
        budget.account_members(component.formulations().len(), "Formulation")?;
        let declarations = component.formulations().collect::<Vec<_>>();
        let formulations = encode_sorted_records(
            &declarations,
            &mut budget,
            |(name, relation, left, right, _), budget| {
                let (test, trial, zero_on) = component
                    .formulation_test(name)
                    .expect("retained form test");
                let mut encoder = Encoder::new(budget.limits.max_canonical_bytes);
                encoder.field(1, |encoder| encoder.u16(1))?;
                encoder.field(2, |encoder| encode_name(encoder, relation, budget))?;
                encoder.field(3, |encoder| encode_expression(encoder, left, budget, 1))?;
                encoder.field(4, |encoder| encode_expression(encoder, right, budget, 1))?;
                encoder.field(5, |encoder| encode_name(encoder, name, budget))?;
                encoder.field(6, |encoder| encode_name(encoder, test, budget))?;
                encoder.field(7, |encoder| encode_name(encoder, trial, budget))?;
                encoder.field(8, |encoder| {
                    encoder.u32(as_u32(zero_on.len(), "test boundaries")?)?;
                    for boundary in zero_on {
                        encode_name(encoder, boundary, budget)?;
                    }
                    Ok(())
                })?;
                encoder.finish()
            },
        )?;
        let mut encoder = Encoder::new(limits.max_canonical_bytes);
        encoder.raw(MAGIC)?;
        encoder.u16(CANONICAL_FORMULATION_VERSION)?;
        encoder.field(1, |encoder| {
            encode_name(encoder, component.name(), &mut budget)
        })?;
        encoder.field(2, |encoder| encoder.records(&formulations))?;
        Ok(Self(Sha256::digest(encoder.finish()?).into()))
    }
}

impl fmt::Debug for AuthoredFormSourceIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "AuthoredFormSourceIdentity({self})")
    }
}

impl fmt::Display for AuthoredFormSourceIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use eqiora_lang::{format, parse};

    use super::*;

    fn document(source: &str) -> Document {
        parse("fixture.eqi", source).into_document().unwrap()
    }

    #[test]
    fn identity_is_separate_from_model_allocation_identity() {
        let without = "component D() { relation balance { 1 = 0; } }";
        let first = "component D() { relation balance { 1 = 0; } form weak for balance { test w: 1 for u zero_on surface; integrate(region, dot(grad(w), grad(u))) = integrate(region, w * f); } }";
        let changed = "component D() { relation balance { 1 = 0; } form weak for balance { test w: 1 for u zero_on surface; integrate(region, dot(grad(w), k * grad(u))) = integrate(region, w * f); } }";
        let model_identity =
            |source: &str| LocalSourceIdentity::from_document(&document(source)).unwrap();
        let form_identity = |source: &str| {
            AuthoredFormSourceIdentity::from_component(&document(source).components()[0]).unwrap()
        };

        assert_eq!(model_identity(without), model_identity(first));
        assert_eq!(model_identity(first), model_identity(changed));
        assert_eq!(
            model_identity(first),
            model_identity(&format(&document(first)))
        );
        assert_ne!(form_identity(without), form_identity(first));
        assert_ne!(form_identity(first), form_identity(changed));
        for changed in [
            first.replace("zero_on surface", "zero_on other"),
            first.replace("form weak", "form renamed"),
            first
                .replace("test w:", "test v:")
                .replace("grad(w)", "grad(v)")
                .replace("w * f", "v * f"),
        ] {
            assert_eq!(model_identity(first), model_identity(&changed));
            assert_ne!(form_identity(first), form_identity(&changed));
        }

        assert_eq!(
            form_identity(first),
            form_identity(&format(&document(first)))
        );
    }
}
