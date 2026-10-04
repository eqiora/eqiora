//! Retain independently typed nominal members through the ordinary expression lowerer.
use super::*;

pub(in crate::lower) fn lower_record(
    file: &str,
    members: &[LoweringExpression],
    bindings: &BTreeMap<String, Binding>,
) -> Result<ExprDag, Diagnostic> {
    let mut lowerer = ExpressionLowerer {
        file,
        bindings,
        support: None,
        builder: ExprDagBuilder::new(),
        dependencies: BTreeSet::new(),
        ports: BTreeSet::new(),
        cache: HashMap::new(),
        sampling: false,
        allow_discrete_symbols: true,
        allow_observables: false,
        activation: &ActivationSyntax::Continuous,
        initial: true,
    };
    let roots = members
        .iter()
        .map(|member| {
            expression_type(file, member, bindings, None)?;
            // A bus root retains Field ownership; it does not execute a read at
            // a synthesized activation. Whole-Model admission checks its clock.
            if let LoweringExpressionNode::Name(name) = member.node.as_ref()
                && let Some(Binding::Field(id, _)) = bindings.get(name)
            {
                lowerer.builder.symbol(SymbolRef::Field(*id))
            } else {
                lowerer.lower(member).map(|value| value.id)
            }
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    lowerer.builder.finish(roots)
}
