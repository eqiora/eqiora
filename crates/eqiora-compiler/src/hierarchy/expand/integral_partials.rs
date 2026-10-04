//! Leibniz contributions use ordinary integral Observables, partials and exact point evaluation.
use super::*;
use crate::lower::LoweringExpression as E;
use eqiora_lang::BinaryOp;

impl RootExpansion<'_, '_> {
    pub(super) fn expand_integral_partials(&mut self) -> Result<(), Diagnostic> {
        let sources = self
            .items
            .iter()
            .filter_map(|item| match item {
                FlatItemBlueprint::Observable {
                    name,
                    value,
                    reduction: Some(measure),
                    domain,
                    ..
                } if measure.limits.is_some() => Some((
                    name.clone(),
                    (value.clone(), measure.clone(), domain.clone()),
                )),
                _ => None,
            })
            .collect::<BTreeMap<_, _>>();
        let parameters = self
            .items
            .iter()
            .filter_map(|item| match item {
                FlatItemBlueprint::Parameter { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let held_unknowns = self
            .items
            .iter()
            .filter_map(|item| match item {
                FlatItemBlueprint::Field { name, .. }
                | FlatItemBlueprint::Port { name, .. }
                | FlatItemBlueprint::Observable { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let mut generated = Vec::new();
        let mut structure = Vec::new();
        for item in &mut self.items {
            let FlatItemBlueprint::Observable {
                name,
                value,
                value_type,
                reduction: None,
                domain,
                range,
                identity,
            } = item
            else {
                continue;
            };
            let Some((operand, wrt)) = value.partial_operands() else {
                continue;
            };
            let Some(source) = operand.name_value() else {
                continue;
            };
            let source = source.to_owned();
            let wrt = wrt.clone();
            let Some((density, measure, original_output)) = sources.get(&source) else {
                continue;
            };
            let reject = |message| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    &identity.definition.file,
                    *range,
                    message,
                )
            };
            if domain.is_some() || original_output.is_some() || measure.measure.is_some() {
                return Err(reject(
                    "moving-limit partial requires one complete Cartesian coordinate interval",
                ));
            }
            if !wrt
                .name_value()
                .is_some_and(|name| parameters.contains(name))
            {
                return Err(reject(
                    "moving-limit partial requires an independent Parameter; integral coordinates are bound",
                ));
            }
            if density
                .referenced_names()
                .iter()
                .any(|name| held_unknowns.contains(name))
            {
                return Err(reject(
                    "moving-limit differentiation requires explicit polynomial density regularity, not an unknown Field or reduced dependency",
                ));
            }
            self.declaration_count = self
                .declaration_count
                .checked_add(1)
                .filter(|count| *count <= self.elaborator.limits.max_declarations)
                .ok_or_else(|| {
                    reject("integral partial contribution exceeds the declaration expansion limit")
                })?;
            let key = identity.key.integral_partial()?;
            let full = key.full_identity()?;
            let helper = internal_name(full);
            let mut helper_identity = identity.clone();
            helper_identity.key = key;
            helper_identity.full = full;
            generated.push(FlatItemBlueprint::Observable {
                name: helper.clone(),
                value_type: value_type.clone(),
                value: E::partial(density.clone(), wrt.clone(), *range),
                reduction: Some(measure.clone()),
                domain: None,
                range: *range,
                identity: helper_identity,
            });
            let coordinate =
                E::coordinate(measure.domain.clone(), measure.domain.clone(), 0, *range);
            let limits = measure.limits.as_ref().expect("selected moving limits");
            let endpoint = |limit: &E| {
                E::binary(
                    BinaryOp::Mul,
                    E::evaluate_at(
                        density.clone(),
                        vec![(coordinate.clone(), limit.clone())],
                        None,
                        *range,
                    ),
                    E::partial(limit.clone(), wrt.clone(), *range),
                    *range,
                )
            };
            let boundary = E::binary(
                BinaryOp::Sub,
                endpoint(&limits[1]),
                endpoint(&limits[0]),
                *range,
            );
            *value = E::binary(
                BinaryOp::Add,
                E::name(helper.clone(), *range),
                boundary,
                *range,
            );
            let mut dependencies = self
                .structural_dependencies
                .get(&source)
                .cloned()
                .unwrap_or_default();
            dependencies.extend(
                self.structural_dependencies
                    .get(name)
                    .into_iter()
                    .flatten()
                    .cloned(),
            );
            structure.push((name.clone(), dependencies.clone()));
            structure.push((helper, dependencies));
        }
        for (owner, dependencies) in structure {
            self.record_structural(&owner, dependencies)?;
        }
        self.items.extend(generated);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn leibniz_contribution_obeys_the_existing_declaration_limit() {
        use crate::StaticBindingValue;
        use crate::hierarchy::{HierarchyLimits, selected};
        use eqiora_core::{DimExponents, DynQuantity};
        use eqiora_schema::kernel::AxisBounds;
        let source = "model Moving(support line:interval(m)) {
            coordinate x:m on line from line; parameter a:m=1[m];
            variable anchor:1; relation fixed {anchor=1;}
            observable total:m^3=integral(x*x,measure(line),lower=0[m],upper=a);
            observable slope:m^2=partial(total,wrt=a);
        }";
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        let binding = StaticBindingValue::CoordinateInterval(
            AxisBounds::new(
                DynQuantity::new(-4.0, length),
                DynQuantity::new(4.0, length),
            )
            .unwrap(),
        );
        let compile = |limit| {
            selected::local_document(
                "limits.eqi",
                source.len(),
                eqiora_lang::parse("limits.eqi", source)
                    .into_document()
                    .unwrap(),
                Some("Moving"),
                &[("line", binding)],
                HierarchyLimits {
                    max_declarations: limit,
                    ..Default::default()
                },
            )
        };
        // Interval, Parameter, Field, Relation + activation, two outputs, one contribution.
        compile(8).unwrap();
        let errors = compile(7).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("integral partial contribution")),
            "{errors:?}"
        );
    }
}
