//! Separate canonical identity for authored formulations.

use super::*;

const MAGIC: &[u8; 8] = b"EQIORAFM";
const CANONICAL_FORMULATION_VERSION: u16 = 4;

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
            |(name, relations, equations, _), budget| {
                let binding = component
                    .formulation_binding(name)
                    .expect("retained form binder");
                let mut encoder = Encoder::new(budget.limits.max_canonical_bytes);
                encoder.field(1, |encoder| encoder.u16(1))?;
                budget.account_members(relations.len(), "Formulation Relations")?;
                encoder.field(2, |encoder| {
                    encoder.u32(as_u32(relations.len(), "Formulation Relations")?)?;
                    for relation in *relations {
                        encode_name(encoder, relation, budget)?;
                    }
                    Ok(())
                })?;
                budget.account_members(equations.len(), "Formulation equalities")?;
                encoder.field(3, |encoder| {
                    encoder.u32(as_u32(equations.len(), "Formulation equalities")?)?;
                    for (left, right) in *equations {
                        encode_expression(encoder, left, budget, 1)?;
                        encode_expression(encoder, right, budget, 1)?;
                    }
                    Ok(())
                })?;
                encoder.field(5, |encoder| encode_name(encoder, name, budget))?;
                match binding {
                    eqiora_lang::FormulationBinding::WeakTests { tests } => {
                        budget.account_members(tests.len(), "Formulation tests")?;
                        encoder.field(6, |e| {
                            e.u16(1)?;
                            e.u32(as_u32(tests.len(), "Formulation tests")?)?;
                            for (name, trial, zero_on) in tests {
                                encode_name(e, name, budget)?;
                                encode_name(e, trial, budget)?;
                                budget.account_members(zero_on.len(), "test boundaries")?;
                                e.u32(as_u32(zero_on.len(), "test boundaries")?)?;
                                for name in zero_on {
                                    encode_name(e, name, budget)?;
                                }
                            }
                            Ok(())
                        })?;
                    }
                    eqiora_lang::FormulationBinding::Interval {
                        name,
                        lower,
                        upper,
                        domain,
                    } => {
                        encoder.field(6, |e| {
                            e.u16(2)?;
                            encode_name(e, name, budget)
                        })?;
                        encoder.field(7, |e| {
                            for name in [lower, upper, domain] {
                                encode_name(e, name, budget)?;
                            }
                            Ok(())
                        })?;
                    }
                }
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
    #[test]
    fn plural_identity_binds_each_relation_test_and_equality() {
        let source = "component C() { form weak for momentum,incompressibility { test v:1 for velocity zero_on surface; test q:1 for pressure; integrate(body,v)=integrate(body,force); integrate(body,q)=integrate(body,0); } }";
        let identity = |source: &str| {
            AuthoredFormSourceIdentity::from_component(&document(source).components()[0]).unwrap()
        };
        assert_eq!(identity(source), identity(&format(&document(source))));
        for changed in [
            source.replace("momentum,incompressibility", "momentum,other"),
            source.replace("momentum,incompressibility", "incompressibility,momentum"),
            source.replace("for pressure;", "for other;"),
            source.replace("test q:1", "test r:1"),
            source.replace("for pressure;", "for pressure zero_on surface;"),
            source.replace("integrate(body,0)", "integrate(body,1)"),
            source.replace(" integrate(body,q)=integrate(body,0);", ""),
        ] {
            assert_ne!(identity(source), identity(&changed));
            assert_eq!(
                LocalSourceIdentity::from_document(&document(source)).unwrap(),
                LocalSourceIdentity::from_document(&document(&changed)).unwrap()
            );
        }
    }
}
