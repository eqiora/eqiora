//! Component algebra shares one lowering for pointwise values and retained fluxes.
use super::*;
use crate::form_compiler::bilinear::BasisComponent;

pub(super) fn lower<S: Coefficient>(
    context: &Context<'_, S>,
    id: ExprId,
    coefficient: Data<S>,
    row: &mut Row<S>,
    flux: bool,
    depth: usize,
) -> Result<(), Diagnostic> {
    let components = match context.affine_components(id, depth)? {
        Some(components) => components,
        None => context
            .components(id, depth)?
            .into_iter()
            .map(
                |offset| crate::form_compiler::spatial::data::affine::ComponentRow {
                    offset,
                    terms: Vec::new(),
                },
            )
            .collect(),
    };
    let width = if flux { context.dimension } else { 1 };
    if components.len() != row.forcing.len() * width {
        return Err(invalid(
            "component action differs from its exact residual shape",
        ));
    }
    for (index, component) in components.into_iter().enumerate() {
        let test = if flux {
            BasisComponent::Gradient {
                component: index / width,
                axis: index % width,
            }
        } else {
            BasisComponent::Value(index)
        };
        let offset = coefficient.clone().multiply(component.offset);
        if flux {
            row.flux.push(FluxTerm::Prescribed {
                component: index / width,
                axis: index % width,
                value: offset,
            });
        } else {
            row.forcing[index] = row.forcing[index]
                .clone()
                .add(offset.multiply(Data::constant(
                    context.dimension,
                    <S as From<f64>>::from(-1.),
                )));
        }
        for (trial, component, value) in component.terms {
            let term = Term {
                trial,
                derivative: false,
                pairing: Pairing::Components {
                    test,
                    trial: component,
                },
                coefficient: coefficient.clone().multiply(value),
                positive_diffusion: false,
            };
            if flux {
                row.flux.push(FluxTerm::Trial(term.clone()));
            }
            row.terms.push(term);
        }
    }
    Ok(())
}
