//! Adapt real-symbol residuals without duplicating component arithmetic.
use super::*;
use eqiora_core::ScalarDomain;

impl ComponentScalarization {
    /// Real residuals with complex intermediates use the ordinary component
    /// lowering once, then retain its real-symbol scalar program for exact
    /// coefficient and regularity consumers.
    pub(crate) fn lower_real_scalar<I: Clone + Eq>(
        typed: &TypedResidual<I>,
    ) -> Result<crate::ScalarOperatorIr, Diagnostic> {
        let mut work = 0;
        let mut lowering = ComponentDagLowering::new(
            typed.expression(),
            typed.node_types(),
            &mut work,
            SpatialContext::Pointwise,
        );
        let mut roots = Vec::new();
        for &root in typed.expression().roots() {
            let ty = typed.node_type(root).expect("typed root");
            if ty.value_type.scalar_domain() != ScalarDomain::Real || !ty.shape().is_scalar() {
                return Err(invalid_component_ir(
                    "typed scalar projection requires real scalar roots",
                ));
            }
            roots.push(lowering.lower_part(root, &[], ScalarPart::Real)?);
        }
        let symbol_types = typed
            .expression()
            .nodes()
            .iter()
            .zip(typed.node_types())
            .filter_map(|(node, ty)| match node {
                ExprNode::Symbol(symbol) => Some((*symbol, &ty.value_type)),
                _ => None,
            })
            .collect::<HashMap<_, _>>();
        for source in &lowering.inputs {
            let ty = symbol_types.get(&source.symbol()).ok_or_else(|| {
                invalid_component_ir("scalar projection input has no Semantic type")
            })?;
            if ty.scalar_domain() != ScalarDomain::Real || !ty.shape().is_scalar() {
                return Err(invalid_component_ir(
                    "typed scalar projection requires real scalar symbols",
                ));
            }
        }
        let source_values = typed
            .node_types()
            .iter()
            .enumerate()
            .map(|(index, ty)| {
                if ty.value_type.scalar_domain() == ScalarDomain::Complex || !ty.shape().is_scalar()
                {
                    None
                } else {
                    lowering
                        .cache
                        .components
                        .get(&(index, Box::default(), ScalarPart::Real))
                        .copied()
                }
            })
            .collect();
        lowering
            .builder
            .finish(roots)?
            .into_symbol_operator(source_values)
    }
}
