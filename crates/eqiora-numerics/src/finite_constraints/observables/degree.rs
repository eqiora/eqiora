//! Conservative per-factor degree admission; no floating-point expression reassociation.
use super::*;

impl Context<'_> {
    pub(super) fn degree_density(
        &mut self,
        id: Id<kinds::Observable>,
        depth: usize,
    ) -> Result<Degree, Diagnostic> {
        if depth > 32 {
            return Err(invalid(
                "finite integral degree dependency depth exceeds 32",
            ));
        }
        let typed = self
            .kernel
            .typed_observable(id)
            .map_err(|errors| errors.into_iter().next().expect("failed typing"))?;
        self.charge(typed.expression().nodes().len())?;
        let components = eqiora_ir::ComponentScalarization::lower(&typed)?;
        let [row] = components.rows() else {
            return Err(invalid("integral density requires one real scalar root"));
        };
        let mut inputs = Vec::with_capacity(row.symbols().len());
        for coordinate in row.symbols() {
            inputs.push(match coordinate.symbol() {
                SymbolRef::Field(_) | SymbolRef::Parameter(_) => Degree::new(),
                SymbolRef::Coordinate { factor, axis, .. } => {
                    Degree::from([((factor.erase(), axis), 1)])
                }
                SymbolRef::Observable(dependency) => {
                    self.degree_observable(dependency, depth + 1)?
                }
                _ => {
                    return Err(invalid(
                        "finite integral degree references an unsupported symbol",
                    ));
                }
            });
        }
        let axes = inputs
            .iter()
            .flat_map(|degree| degree.keys().copied())
            .collect::<std::collections::BTreeSet<_>>();
        let mut result = Degree::new();
        for axis in axes {
            let degrees = inputs
                .iter()
                .map(|degree| Some(usize::from(degree.get(&axis).copied().unwrap_or(0))))
                .collect::<Vec<_>>();
            let degree = row.polynomial_degree(&degrees).ok_or_else(|| invalid(
                "finite integral requires a polynomial in its coordinate factors with coordinate-independent denominators",
            ))?;
            let degree = u16::try_from(degree)
                .map_err(|_| invalid("finite polynomial degree exceeds u16"))?;
            if degree != 0 {
                result.insert(axis, degree);
            }
        }
        // A polynomial value on a restricted domain is insufficient for a
        // regular integral. Reuse the factor-density domain check before
        // choosing quadrature nodes that might miss a pole or invalid guard.
        let varying = row
            .symbols()
            .iter()
            .zip(&inputs)
            .filter(|(_, degree)| !degree.is_empty())
            .map(|(coordinate, _)| coordinate.symbol())
            .collect::<std::collections::HashSet<_>>();
        eqiora_ir::ScalarOperatorIr::lower_typed_scalar(&typed)?
            .require_regular_density(|symbol| varying.contains(&symbol))
            .map_err(|error| invalid(error.message()))?;
        Ok(result)
    }

    fn degree_observable(
        &mut self,
        id: Id<kinds::Observable>,
        depth: usize,
    ) -> Result<Degree, Diagnostic> {
        if let Some(degree) = self.degrees.get(&id.erase()) {
            return Ok(degree.clone());
        }
        let Some(KernelNode::Observable(definition)) = self.kernel.node(id.erase()) else {
            return Err(invalid("polynomial degree references a foreign Observable"));
        };
        if definition.reduction().limits().is_some() {
            return Err(invalid(
                "explicit integral limits require admitted numerical endpoint evaluation",
            ));
        }
        let definition = definition.clone();
        let mut degree = self.degree_density(id, depth)?;
        if let ObservableReduction::SpatialIntegral { domain, .. } = definition.reduction() {
            for (axis, _) in axes(self.kernel, domain)? {
                degree.remove(&axis);
            }
        }
        self.degrees.insert(id.erase(), degree.clone());
        Ok(degree)
    }
}
