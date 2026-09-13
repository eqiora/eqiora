use super::{MAX_DEPTH, MAX_NODES, Math, MathReference, MathRendering};
use crate::ModelDocument;
use eqiora_compiler::QuantityRole;
use eqiora_core::{Diagnostic, RawId, diagnostic::codes};
use eqiora_lang::{NotationLabel, NotationProfile};
use eqiora_schema::kernel::{
    ComparisonOp, ExprDag, ExprId, ExprNode, KernelNode, SymbolRef, UnaryMathFunction,
};

mod form;
mod literal;

impl ModelDocument {
    /// Render the admitted, ordered left/right equations of one exact Relation.
    ///
    /// # Errors
    /// Rejects a non-Relation ID, a constrained Relation, or an expression exceeding
    /// presentation limits.
    pub fn render_equations(
        &self,
        relation: RawId,
        profile: NotationProfile,
    ) -> Result<Vec<MathRendering>, Diagnostic> {
        let Some(KernelNode::Relation(relation)) = self.program().node(relation) else {
            return Err(failure(
                "mathematical rendering requires an exact Relation ID",
            ));
        };
        if relation.has_constraints() {
            return Err(failure(
                "equation rendering does not admit inequality or complementarity conditions",
            ));
        }
        relation
            .equation_sides()
            .map(|(left, right)| {
                let mut context = Context {
                    document: self,
                    references: Vec::new(),
                    remaining: MAX_NODES,
                };
                let left = context.lower(relation.expression(), left, 0)?;
                let right = context.lower(relation.expression(), right, 0)?;
                super::output::render(
                    Math::Binary("=", Box::new(left), Box::new(right)),
                    context.references,
                    profile,
                )
            })
            .collect()
    }
}

pub(super) fn failure(message: &str) -> Diagnostic {
    Diagnostic::error(codes::LANGUAGE_TYPE_ERROR, message)
}

struct Context<'a> {
    document: &'a ModelDocument,
    references: Vec<MathReference>,
    remaining: usize,
}

impl Context<'_> {
    fn reference(&mut self, reference: MathReference) {
        if !self.references.contains(&reference) {
            self.references.push(reference);
        }
    }

    fn quantity(&mut self, graph_id: RawId, role: QuantityRole) -> Result<Math, Diagnostic> {
        let entries = self
            .document
            .notation()
            .iter()
            .filter(|entry| entry.graph_id() == Some(graph_id) && entry.identity().role() == role)
            .take(MAX_NODES + 1)
            .collect::<Vec<_>>();
        if entries.len() > MAX_NODES {
            return Err(failure(
                "quantity reference inventory exceeds bounded presentation size",
            ));
        }
        let label = if let [entry] = entries.as_slice() {
            entry.label().clone()
        } else {
            NotationLabel::identifier(&format!("{}_{}", graph_id, role.name()))
                .ok_or_else(|| failure("semantic identity exceeds notation presentation limit"))?
        };
        self.reference(MathReference {
            graph_id: Some(graph_id),
            role: Some(role),
            declarations: entries.iter().map(|entry| entry.identity()).collect(),
            operator: None,
        });
        Ok(Math::Label(label))
    }

    fn lower(&mut self, dag: &ExprDag, id: ExprId, depth: usize) -> Result<Math, Diagnostic> {
        if depth > MAX_DEPTH || self.remaining == 0 {
            return Err(failure(
                "mathematical expression exceeds bounded presentation size",
            ));
        }
        self.remaining -= 1;
        let node = dag
            .node(id)
            .ok_or_else(|| failure("mathematical expression references an absent node"))?;
        let next = depth + 1;
        let call = |name: &str, args| Math::Function(name.to_owned(), args);
        Ok(match node {
            ExprNode::Symbol(symbol) => match symbol {
                SymbolRef::Field(id) => self.quantity((*id).into(), QuantityRole::Value)?,
                SymbolRef::Parameter(id) => self.quantity((*id).into(), QuantityRole::Value)?,
                SymbolRef::Port(id) => self.quantity((*id).into(), QuantityRole::Value)?,
                SymbolRef::Across(id) => self.quantity((*id).into(), QuantityRole::Across)?,
                SymbolRef::Through(id) => self.quantity((*id).into(), QuantityRole::Through)?,
                SymbolRef::PortTrace(id) => self.quantity((*id).into(), QuantityRole::Trace)?,
                SymbolRef::PortFlux(id) => self.quantity((*id).into(), QuantityRole::Flux)?,
                SymbolRef::Derivative(id) => {
                    Math::Derivative(Box::new(self.quantity((*id).into(), QuantityRole::Value)?))
                }
                SymbolRef::Pre(id) => call(
                    "pre",
                    vec![self.quantity((*id).into(), QuantityRole::Value)?],
                ),
                SymbolRef::Next(id) => call(
                    "next",
                    vec![self.quantity((*id).into(), QuantityRole::Value)?],
                ),
                SymbolRef::Time => call("time", vec![]),
                _ => return Err(failure("unsupported semantic symbol presentation")),
            },
            ExprNode::Constant(value) => self.literal(value)?,
            ExprNode::Add(a, b)
            | ExprNode::Sub(a, b)
            | ExprNode::Mul(a, b)
            | ExprNode::Div(a, b)
            | ExprNode::And(a, b)
            | ExprNode::Or(a, b) => {
                let operator = match node {
                    ExprNode::Add(..) => "+",
                    ExprNode::Sub(..) => "-",
                    ExprNode::Mul(..) => "*",
                    ExprNode::Div(..) => "/",
                    ExprNode::And(..) => "and",
                    _ => "or",
                };
                Math::Binary(
                    operator,
                    Box::new(self.lower(dag, *a, next)?),
                    Box::new(self.lower(dag, *b, next)?),
                )
            }
            ExprNode::Compare(op, a, b) => {
                let op = match op {
                    ComparisonOp::Equal => "=",
                    ComparisonOp::NotEqual => "!=",
                    ComparisonOp::Less => "<",
                    ComparisonOp::LessEqual => "<=",
                    ComparisonOp::Greater => ">",
                    ComparisonOp::GreaterEqual => ">=",
                };
                Math::Binary(
                    op,
                    Box::new(self.lower(dag, *a, next)?),
                    Box::new(self.lower(dag, *b, next)?),
                )
            }
            ExprNode::PowI(value, exponent) => {
                Math::Power(Box::new(self.lower(dag, *value, next)?), *exponent)
            }
            ExprNode::Index { value, index } => {
                Math::Index(Box::new(self.lower(dag, *value, next)?), *index)
            }
            ExprNode::Array { elements } => Math::Array(
                elements
                    .iter()
                    .map(|id| self.lower(dag, *id, next))
                    .collect::<Result<_, _>>()?,
            ),
            ExprNode::Neg(value) => Math::Negative(Box::new(self.lower(dag, *value, next)?)),
            ExprNode::Gradient(value) => Math::Gradient(Box::new(self.lower(dag, *value, next)?)),
            ExprNode::Not(value)
            | ExprNode::Hold(value)
            | ExprNode::Ordinal(value)
            | ExprNode::ToReal(value)
            | ExprNode::ToInteger(value)
            | ExprNode::Divergence(value)
            | ExprNode::SymmetricPart(value)
            | ExprNode::IsotropicLift(value)
            | ExprNode::Trace(value)
            | ExprNode::NormalComponent(value) => {
                let name = match node {
                    ExprNode::Neg(_) => "negative",
                    ExprNode::Not(_) => "not",
                    ExprNode::Hold(_) => "hold",
                    ExprNode::Ordinal(_) => "ordinal",
                    ExprNode::ToReal(_) => "real",
                    ExprNode::ToInteger(_) => "integer",
                    ExprNode::Gradient(_) => "gradient",
                    ExprNode::Divergence(_) => "divergence",
                    ExprNode::SymmetricPart(_) => "symmetric part",
                    ExprNode::IsotropicLift(_) => "isotropic lift",
                    ExprNode::Trace(_) => "trace",
                    _ => "normal component",
                };
                call(name, vec![self.lower(dag, *value, next)?])
            }
            ExprNode::UnaryMath(function, value) => call(
                match function {
                    UnaryMathFunction::Sin => "sin",
                    UnaryMathFunction::Sqrt => "sqrt",
                    _ => return Err(failure("unsupported typed mathematical function")),
                },
                vec![self.lower(dag, *value, next)?],
            ),
            ExprNode::Require { condition, value } => call(
                "require",
                vec![
                    self.lower(dag, *condition, next)?,
                    self.lower(dag, *value, next)?,
                ],
            ),
            ExprNode::Select {
                condition,
                then_value,
                else_value,
            } => call(
                "select",
                vec![
                    self.lower(dag, *condition, next)?,
                    self.lower(dag, *then_value, next)?,
                    self.lower(dag, *else_value, next)?,
                ],
            ),
            ExprNode::Complex { real, imag } => call(
                "complex",
                vec![self.lower(dag, *real, next)?, self.lower(dag, *imag, next)?],
            ),
            ExprNode::Quotient(a, b) | ExprNode::Remainder(a, b) => call(
                if matches!(node, ExprNode::Quotient(..)) {
                    "quotient"
                } else {
                    "remainder"
                },
                vec![self.lower(dag, *a, next)?, self.lower(dag, *b, next)?],
            ),
            ExprNode::SpatialCoordinate(axis) => {
                call("coordinate", vec![Math::Number(axis.to_string())])
            }
            ExprNode::Sample { value, clock } => {
                self.reference(MathReference {
                    graph_id: Some((*clock).into()),
                    role: None,
                    declarations: vec![],
                    operator: None,
                });
                call(
                    "sample",
                    vec![
                        self.lower(dag, *value, next)?,
                        Math::Label(
                            NotationLabel::identifier(&clock.to_string())
                                .ok_or_else(|| failure("clock identity exceeds notation limit"))?,
                        ),
                    ],
                )
            }
            ExprNode::PureOperatorApplication(application) => {
                self.reference(MathReference {
                    graph_id: None,
                    role: None,
                    declarations: vec![],
                    operator: Some(application.definition()),
                });
                call(
                    &format!("operator {}", application.definition()),
                    application
                        .arguments()
                        .iter()
                        .map(|id| self.lower(dag, *id, next))
                        .collect::<Result<_, _>>()?,
                )
            }
            _ => return Err(failure("unsupported typed expression presentation")),
        })
    }
}
