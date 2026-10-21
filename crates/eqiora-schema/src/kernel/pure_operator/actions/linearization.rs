//! Independent real coordinates reuse retained partials and scoped bound inputs.
use super::*;
use crate::kernel::UnaryMathFunction;

pub(super) struct Coordinate {
    pub(super) formal: u16,
    pub(super) axes: Box<[ComponentIndex]>,
    pub(super) part: Option<UnaryMathFunction>,
    pub(super) value: CalculusNodeId,
}

pub(super) struct Linearization {
    pub(super) root: CalculusNodeId,
    pub(super) coordinates: Vec<Coordinate>,
}

impl Linearization {
    pub(super) fn bind(
        builder: &mut CalculusBuilder,
        root: CalculusNodeId,
        inputs: &[(u16, PureValueClass)],
    ) -> Result<Self, PureOperatorError> {
        let end = definition_index(root, builder.nodes.len())?;
        let source = builder.nodes[..=end].to_vec();
        let reachable = builder.reachable_nodes(root)?;
        let mut mapped = vec![None; source.len()];
        let mut bindings = std::collections::BTreeMap::new();
        let mut coordinates = Vec::new();
        for (index, node) in source.iter().enumerate() {
            if !reachable[index] {
                continue;
            }
            let id = if let CalculusNode::FormalComponent { formal, axes } = node
                && inputs.iter().any(|(selected, _)| selected == formal)
            {
                let key = (*formal, axes.clone());
                if let Some(&id) = bindings.get(&key) {
                    id
                } else {
                    let input = CalculusNodeId(index as u32);
                    let id = bind_component(builder, *formal, axes, input, &mut coordinates)?;
                    bindings.insert(key, id);
                    id
                }
            } else {
                builder.push(
                    node.map_operands(|id| mapped[id.index() as usize].expect("reachable operand")),
                )?
            };
            mapped[index] = Some(id);
        }
        // Even an ignored input differentiates the demanded source, retaining its
        // validity and derivative-domain checks instead of fabricating a zero.
        for (formal, class) in inputs {
            if !coordinates
                .iter()
                .any(|coordinate| coordinate.formal == *formal)
            {
                let axes = vec![ComponentIndex::Fixed(0); class.rank()].into_boxed_slice();
                let input = builder.push(CalculusNode::FormalComponent {
                    formal: *formal,
                    axes: axes.clone(),
                })?;
                bind_component(builder, *formal, &axes, input, &mut coordinates)?;
            }
        }
        Ok(Self {
            root: mapped[end].expect("reachable root"),
            coordinates,
        })
    }
}

fn bind_component(
    builder: &mut CalculusBuilder,
    formal: u16,
    axes: &[ComponentIndex],
    input: CalculusNodeId,
    coordinates: &mut Vec<Coordinate>,
) -> Result<CalculusNodeId, PureOperatorError> {
    let domain = builder.value_type(input)?.scalar_domain();
    let parts: &[Option<UnaryMathFunction>] = match domain {
        ScalarDomain::Real => &[None],
        ScalarDomain::Complex => &[Some(UnaryMathFunction::Real), Some(UnaryMathFunction::Imag)],
        _ => return Err(PureOperatorError::FormalTypeMismatch),
    };
    let mut values = Vec::with_capacity(parts.len());
    for &part in parts {
        let value = match part {
            Some(part) => builder.push(CalculusNode::UnaryMath(part, input))?,
            None => input,
        };
        let value = builder.push(CalculusNode::BoundInput(value))?;
        coordinates.push(Coordinate {
            formal,
            axes: axes.into(),
            part,
            value,
        });
        values.push(value);
    }
    match values.as_slice() {
        [real] => Ok(*real),
        [real, imag] => builder.push(CalculusNode::Complex {
            real: *real,
            imag: *imag,
        }),
        _ => unreachable!("numeric domain has one or two real coordinates"),
    }
}
