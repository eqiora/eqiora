use super::{Context, MAX_DEPTH, MAX_NODES, Math, MathReference, MathRendering, failure};
use crate::ModelDocument;
use eqiora_compiler::{AuthoredFormExpressionV1 as Form, QuantityRole};
use eqiora_core::{Diagnostic, EntityKind, RawId};
use eqiora_lang::{NotationLabel, NotationProfile};

impl ModelDocument {
    /// Render every retained authored Formulation, preserving test/trial and support identities.
    ///
    /// # Errors
    /// Rejects an unavailable semantic reference or excessive presentation size.
    pub fn render_formulations(
        &self,
        profile: NotationProfile,
    ) -> Result<Vec<MathRendering>, Diagnostic> {
        self.authored_formulations()
            .map(|form| {
                let form = form.projection();
                let mut context = Context {
                    document: self,
                    references: Vec::new(),
                    remaining: MAX_NODES,
                };
                let left = context.form(form.left(), 0)?;
                let right = context.form(form.right(), 0)?;
                let mut equation = Math::Binary("=", Box::new(left), Box::new(right));
                if let Some((interval, lower, upper)) = form.interval() {
                    let parent = context.exact(form.domain_ulid(), EntityKind::Domain)?;
                    context.reference(MathReference {
                        graph_id: Some(parent),
                        role: None,
                        declarations: vec![],
                        operator: None,
                    });
                    let mut arguments = [interval, lower, upper, form.domain_ulid()]
                        .into_iter()
                        .map(|name| {
                            NotationLabel::identifier(name)
                                .map(Math::Label)
                                .ok_or_else(|| failure("interval binder exceeds notation limit"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    arguments.push(equation);
                    equation = Math::Function("for_all_ordered_intervals".into(), arguments);
                }
                super::super::output::render(equation, context.references, profile)
            })
            .collect()
    }
}

impl Context<'_> {
    fn exact(&self, ulid: &str, kind: EntityKind) -> Result<RawId, Diagnostic> {
        self.document
            .program()
            .nodes()
            .map(|node| node.id())
            .find(|id| id.kind() == kind && id.ulid().to_string() == ulid)
            .ok_or_else(|| failure("authored form references an unavailable semantic entity"))
    }

    fn form(&mut self, node: &Form, depth: usize) -> Result<Math, Diagnostic> {
        if depth > MAX_DEPTH || self.remaining == 0 {
            return Err(failure("authored form exceeds bounded presentation size"));
        }
        self.remaining -= 1;
        let next = depth + 1;
        Ok(match node {
            Form::EndpointFlux {
                interval,
                endpoint,
                normal,
                flux,
            } => Math::Function(
                "outward_flux".into(),
                vec![
                    Math::Label(
                        NotationLabel::identifier(interval)
                            .ok_or_else(|| failure("invalid interval binder"))?,
                    ),
                    Math::Label(
                        NotationLabel::identifier(endpoint)
                            .ok_or_else(|| failure("invalid endpoint binder"))?,
                    ),
                    Math::Number(normal.to_string()),
                    self.form(flux, next)?,
                ],
            ),
            Form::IntervalIntegral {
                interval,
                integrand,
            } => Math::Integral(
                Box::new(Math::Label(
                    NotationLabel::identifier(interval)
                        .ok_or_else(|| failure("invalid interval binder"))?,
                )),
                Box::new(self.form(integrand, next)?),
            ),
            Form::Number { value } => Math::Number(value.to_string()),
            Form::Field { ulid } => {
                self.quantity(self.exact(ulid, EntityKind::Field)?, QuantityRole::Value)?
            }
            Form::Parameter { ulid } => self.quantity(
                self.exact(ulid, EntityKind::Parameter)?,
                QuantityRole::Value,
            )?,
            Form::Test { field_ulid } => Math::Function(
                "test".into(),
                vec![self.quantity(
                    self.exact(field_ulid, EntityKind::Field)?,
                    QuantityRole::Value,
                )?],
            ),
            Form::Coordinate { axis } => {
                Math::Function("coordinate".into(), vec![Math::Number(axis.to_string())])
            }
            Form::Neg { value } => Math::Negative(Box::new(self.form(value, next)?)),
            Form::Gradient { value } => Math::Gradient(Box::new(self.form(value, next)?)),
            Form::Sin { value } => Math::Function("sin".into(), vec![self.form(value, next)?]),
            Form::Add { left, right }
            | Form::Sub { left, right }
            | Form::Mul { left, right }
            | Form::Div { left, right } => {
                let op = match node {
                    Form::Add { .. } => "+",
                    Form::Sub { .. } => "-",
                    Form::Mul { .. } => "*",
                    _ => "/",
                };
                Math::Binary(
                    op,
                    Box::new(self.form(left, next)?),
                    Box::new(self.form(right, next)?),
                )
            }
            Form::Dot { left, right } => Math::Inner(
                Box::new(self.form(left, next)?),
                Box::new(self.form(right, next)?),
            ),
            Form::Pow { base, exponent } => {
                Math::Power(Box::new(self.form(base, next)?), *exponent)
            }
            Form::Integrate {
                domain_ulid,
                integrand,
            } => {
                let graph_id = self.exact(domain_ulid, EntityKind::Domain)?;
                self.reference(MathReference {
                    graph_id: Some(graph_id),
                    role: None,
                    declarations: vec![],
                    operator: None,
                });
                let domain = NotationLabel::identifier(&graph_id.to_string())
                    .ok_or_else(|| failure("support identity exceeds notation limit"))?;
                Math::Integral(
                    Box::new(Math::Label(domain)),
                    Box::new(self.form(integrand, next)?),
                )
            }
            _ => return Err(failure("unsupported authored mathematical form")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn form_reference_cannot_recover_an_entity_of_another_typed_kind() {
        let document =
            ModelDocument::compile("kind.eqi", "model M(){variable x:1;relation law{x=0;}}")
                .unwrap();
        let field = document.aliases()["x"];
        let context = Context {
            document: &document,
            references: vec![],
            remaining: MAX_NODES,
        };
        assert_eq!(
            context
                .exact(&field.ulid().to_string(), EntityKind::Field)
                .unwrap(),
            field
        );
        assert!(
            context
                .exact(&field.ulid().to_string(), EntityKind::Parameter)
                .is_err()
        );
        assert!(
            context
                .exact(&field.ulid().to_string(), EntityKind::Domain)
                .is_err()
        );
    }
}
