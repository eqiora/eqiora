//! First-order actions over retained accepted members, never a new evaluator.

use eqiora_core::{
    Diagnostic, DimExponents, Id, ScalarDomain, ScalarType, ValueType, entity::kinds,
};

use super::{CompleteEvaluationMap, axes, invalid};
use crate::{
    DifferentiableJvp, DifferentiableVjp, DifferentiationEvidence, DifferentiationMode,
    LinearizationState,
};

type Parameter = Id<kinds::Parameter>;

/// Admitted point/seed axes and explicit shared coordinates of a complete map.
///
/// This uses immutable accepted evaluations. A Recompute map explicitly releases
/// numerical state: each action then reaccepts one frozen point and checks its
/// original acceptance evidence before using it for all seeds. It never differentiates solver
/// iterations. First-order implicit
/// JVP/VJP products are exposed; product results have no higher-derivative API.
#[derive(Debug)]
pub struct EvaluationMapProducts<'a> {
    map: &'a CompleteEvaluationMap,
    shared: Vec<usize>,
    mapped: Vec<usize>,
    point_shape: Vec<usize>,
    seed_shape: Vec<usize>,
    extents: Vec<usize>,
    axes: Vec<Option<usize>>,
    shape: Vec<usize>,
    seeds: usize,
    cells: usize,
    estimated_numerical_bytes: usize,
}

impl<'a> EvaluationMapProducts<'a> {
    /// Admit an exact coordinate partition and dense nested point/seed grid.
    ///
    /// `point_shape` flattens to the map's request order. `seed_shape` describes
    /// independent first-order seeds, not derivative order. `point_axes` places
    /// each point axis in the combined grid; remaining positions hold seed axes
    /// in their declared order. For example, `[3]`, `[2]`, `[1]` means seed-major
    /// shape `[2, 3]`; `[0]` means point-major `[3, 2]`. Empty shapes represent a
    /// single scalar occurrence/seed; zero extents represent empty axes.
    ///
    /// Shared IDs retain the supplied order and must occur exactly once in the
    /// Program. All other inputs remain mapped in Program order. Shared values
    /// must match bit-for-bit across points; equal values are never inferred to
    /// be shared. Inputs are coherent-SI real coordinates of that exact Program.
    ///
    /// The byte limit bounds numerical buffers for one complete JVP and one
    /// complete VJP result, including both evidence-point copies, duplicate
    /// primal/full-gradient payloads and shared sums. It excludes the already
    /// retained map, record/other metadata allocation,
    /// caller input buffers and transient derivative-solver scratch. It is not a
    /// process/peak-memory or total execution budget. Rank is bounded to 32;
    /// shape, byte and stride products are checked before product allocation.
    ///
    /// # Errors
    /// Rejects wrong axes/counts, foreign/duplicate IDs, inconsistent sharing,
    /// unsupported scalar/derivative profiles, invalid members or byte limits.
    pub fn new(
        map: &'a CompleteEvaluationMap,
        shared_inputs: &[Parameter],
        point_shape: &[usize],
        seed_shape: &[usize],
        point_axes: &[usize],
        numerical_bytes_limit: usize,
    ) -> Result<Self, Diagnostic> {
        map.validate()?;
        let identity = map.plan().program_identity();
        if identity.scalar_type() != ScalarType::F64 {
            return Err(invalid(
                "mapped products require f64-coordinate first-order implicit evaluations",
            ));
        }
        let rank = point_shape
            .len()
            .checked_add(seed_shape.len())
            .ok_or_else(overflow)?;
        if rank > 32 || point_axes.len() != point_shape.len() {
            return Err(invalid(
                "mapped product axis rank or point-axis inventory is invalid",
            ));
        }
        if axes::product(point_shape).map_err(axis_error)? != map.len() {
            return Err(invalid(
                "point axes must contain every accepted map occurrence",
            ));
        }
        let seeds = axes::product(seed_shape).map_err(axis_error)?;
        let cells = multiply(map.len(), seeds)?;
        addressable::<Option<DifferentiableJvp>>(cells)?;
        addressable::<Option<DifferentiableVjp>>(cells)?;
        let (shared, mapped) = super::partition::coordinates(identity, shared_inputs)?;
        for point in map.plan().points() {
            if let Some(first) = map.plan().points().first()
                && shared.iter().any(|&input| {
                    point.coordinates()[input].to_bits() != first.coordinates()[input].to_bits()
                })
            {
                return Err(invalid(
                    "explicit shared input differs between accepted points",
                ));
            }
        }
        let estimated_numerical_bytes = numerical_bytes(
            cells,
            seeds,
            identity.output_dimension(),
            identity.input_dimension(),
            mapped.len(),
            shared.len(),
        )?;
        if estimated_numerical_bytes > numerical_bytes_limit
            || estimated_numerical_bytes > isize::MAX as usize
        {
            return Err(invalid(
                "mapped products exceed the retained numerical byte limit",
            ));
        }
        let mut positions = point_axes.iter().copied().map(Some).collect::<Vec<_>>();
        if point_axes
            .iter()
            .enumerate()
            .any(|(index, &axis)| axis >= rank || point_axes[..index].contains(&axis))
        {
            return Err(invalid(
                "point-axis positions are duplicate or out of bounds",
            ));
        }
        positions.extend(
            (0..rank)
                .filter(|position| !point_axes.contains(position))
                .map(Some),
        );
        let mut extents = point_shape
            .iter()
            .chain(seed_shape)
            .copied()
            .collect::<Vec<_>>();
        // The axis owner requires a map level. A scalar/scalar product has one
        // private unit level, omitted from its public scalar grid shape.
        if rank == 0 {
            extents.push(1);
            positions.push(Some(0));
        }
        let ty = record_type();
        let layout =
            axes::Layout::new(&ty, &extents, &positions, grid_limits()).map_err(axis_error)?;
        layout
            .admit_buffer(&layout.shape, cells)
            .map_err(axis_error)?;
        let shape = if rank == 0 {
            Vec::new()
        } else {
            layout.shape.clone()
        };
        // Empty record grids must not conceal an unrepresentable logical
        // stride after adding numerical coordinate components.
        for (grid, components) in [
            (shape.as_slice(), identity.output_dimension()),
            (shape.as_slice(), mapped.len()),
            (seed_shape, shared.len()),
        ] {
            axes::product(&with_components(grid, components)).map_err(axis_error)?;
        }
        Ok(Self {
            map,
            shared,
            mapped,
            point_shape: point_shape.to_vec(),
            seed_shape: seed_shape.to_vec(),
            extents,
            axes: positions,
            shape,
            seeds,
            cells,
            estimated_numerical_bytes,
        })
    }

    /// Original complete map; its members remain the primal-map authority.
    #[must_use]
    pub const fn map(&self) -> &'a CompleteEvaluationMap {
        self.map
    }

    /// Point/seed grid shape, excluding numerical coordinate components.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Conservative retained numerical-buffer estimate, not peak memory.
    #[must_use]
    pub const fn estimated_numerical_bytes(&self) -> usize {
        self.estimated_numerical_bytes
    }

    /// Apply shared and occurrence-local tangents at each independent seed.
    ///
    /// Shared input shape is `seed_shape + [shared_input_count]`; mapped input
    /// shape is `shape() + [mapped_input_count]`. All buffers are row-major.
    /// Output records follow that same combined grid, preserving each exact
    /// point's accepted evidence. No partial result is published on failure.
    ///
    /// # Errors
    /// Rejects shape/nonfinite inputs before actions, then preserves derivative
    /// diagnostics and rejects foreign evidence or nonfinite products.
    pub fn jvp(&self, shared: &[f64], mapped: &[f64]) -> Result<EvaluationMapJvp, Diagnostic> {
        admit(shared, multiply(self.seeds, self.shared.len())?)?;
        admit(mapped, multiply(self.cells, self.mapped.len())?)?;
        let mut products = vec![None; self.cells];
        for point in 0..self.map.len() {
            if self.seeds == 0 {
                break;
            }
            let evaluation = self.map.product_member(point)?;
            for seed in 0..self.seeds {
                let position = self.position(point, seed)?;
                let tangent = assemble_tangent(
                    self.map.plan().program_identity().input_dimension(),
                    &self.shared,
                    &self.mapped,
                    row(shared, seed, self.shared.len()),
                    row(mapped, position, self.mapped.len()),
                );
                let product = (|| {
                    let product = evaluation.jvp(&tangent)?;
                    validate_evidence(&evaluation, product.evidence(), DifferentiationMode::Jvp)?;
                    finite(product.tangent())?;
                    Ok(product)
                })().map_err(|error: Diagnostic| error.with_context(format!(
                    "mapped JVP point occurrence {point} {:?}, seed occurrence {seed} {:?}, grid offset {position}",
                    coordinates(point, &self.point_shape), coordinates(seed, &self.seed_shape)
                )))?;
                products[position] = Some(product);
            }
        }
        Ok(EvaluationMapJvp {
            shape: with_components(
                &self.shape,
                self.map.plan().program_identity().output_dimension(),
            ),
            products: products.into_iter().map(Option::unwrap).collect(),
        })
    }

    /// Apply output cotangents and sum shared-input covectors, never average.
    ///
    /// Input shape is `shape() + [output_dimension]`. Each seed's shared sum
    /// uses original request order, irrespective of axis placement. The real
    /// pairing is the existing coordinate pairing. Overflow fails closed;
    /// arbitrary regrouping or different point order need not be bit-identical.
    ///
    /// # Errors
    /// Rejects malformed/nonfinite input before actions. Any member derivative,
    /// foreign-evidence or nonfinite-sum failure rejects the complete product.
    pub fn vjp(&self, cotangents: &[f64]) -> Result<EvaluationMapVjp, Diagnostic> {
        let width = self.map.plan().program_identity().output_dimension();
        admit(cotangents, multiply(self.cells, width)?)?;
        let mut shared = vec![0.0; multiply(self.seeds, self.shared.len())?];
        let mut mapped = vec![0.0; multiply(self.cells, self.mapped.len())?];
        let mut products = vec![None; self.cells];
        for point in 0..self.map.len() {
            if self.seeds == 0 {
                break;
            }
            let evaluation = self.map.product_member(point)?;
            for seed in 0..self.seeds {
                let position = self.position(point, seed)?;
                let product = (|| {
                    let product = evaluation.vjp(row(cotangents, position, width))?;
                    validate_evidence(&evaluation, product.evidence(), DifferentiationMode::Vjp)?;
                    finite(product.input_cotangent())?;
                    accumulate(
                        product.input_cotangent(), &self.shared, &self.mapped,
                        row_mut(&mut shared, seed, self.shared.len()),
                        row_mut(&mut mapped, position, self.mapped.len()),
                    )?;
                    Ok(product)
                })().map_err(|error: Diagnostic| error.with_context(format!(
                    "mapped VJP point occurrence {point} {:?}, seed occurrence {seed} {:?}, grid offset {position}",
                    coordinates(point, &self.point_shape), coordinates(seed, &self.seed_shape)
                )))?;
                products[position] = Some(product);
            }
        }
        let inputs = self.map.plan().program_identity().inputs();
        Ok(EvaluationMapVjp {
            shared_shape: with_components(&self.seed_shape, self.shared.len()),
            mapped_shape: with_components(&self.shape, self.mapped.len()),
            shared_inputs: self
                .shared
                .iter()
                .filter_map(|&index| {
                    crate::differentiation::signature::layout(inputs)
                        .find(|(_, range)| range.start == index)
                        .map(|(input, _)| input.parameter())
                })
                .collect(),
            mapped_inputs: self
                .mapped
                .iter()
                .filter_map(|&index| {
                    crate::differentiation::signature::layout(inputs)
                        .find(|(_, range)| range.start == index)
                        .map(|(input, _)| input.parameter())
                })
                .collect(),
            shared,
            mapped,
            products: products.into_iter().map(Option::unwrap).collect(),
        })
    }

    fn position(&self, point: usize, seed: usize) -> Result<usize, Diagnostic> {
        let mut occurrence = coordinates(point, &self.point_shape);
        occurrence.extend(coordinates(seed, &self.seed_shape));
        if occurrence.is_empty() {
            occurrence.push(0);
        }
        let ty = record_type();
        axes::Layout::new(&ty, &self.extents, &self.axes, grid_limits())
            .map_err(axis_error)?
            .offset(&occurrence, &[])
            .map_err(axis_error)
    }
}

/// Complete first-order JVP records in the admitted point/seed grid.
///
/// More seeds do not establish higher derivatives:
///
/// ```compile_fail
/// use eqiora_api::EvaluationMapJvp;
/// fn second_derivative(first: EvaluationMapJvp) {
///     first.jvp(&[]);
/// }
/// ```
#[derive(Debug)]
pub struct EvaluationMapJvp {
    shape: Vec<usize>,
    products: Vec<DifferentiableJvp>,
}

impl EvaluationMapJvp {
    /// Combined grid followed by complete-Field output components.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// One existing accepted action per grid position, including repetitions.
    #[must_use]
    pub fn products(&self) -> &[DifferentiableJvp] {
        &self.products
    }
}

/// Complete reverse actions with separate shared sums and mapped covectors.
#[derive(Debug)]
pub struct EvaluationMapVjp {
    shared_shape: Vec<usize>,
    mapped_shape: Vec<usize>,
    shared_inputs: Vec<Parameter>,
    mapped_inputs: Vec<Parameter>,
    shared: Vec<f64>,
    mapped: Vec<f64>,
    products: Vec<DifferentiableVjp>,
}

impl EvaluationMapVjp {
    /// Seed axes followed by shared coordinates in the explicitly selected order.
    #[must_use]
    pub fn shared_shape(&self) -> &[usize] {
        &self.shared_shape
    }
    /// Combined point/seed axes followed by mapped coordinates in Program order.
    #[must_use]
    pub fn mapped_shape(&self) -> &[usize] {
        &self.mapped_shape
    }
    /// Exact Program coordinates owning the shared sums.
    #[must_use]
    pub fn shared_inputs(&self) -> &[Parameter] {
        &self.shared_inputs
    }
    /// Exact Program coordinates owning the per-occurrence covectors.
    #[must_use]
    pub fn mapped_inputs(&self) -> &[Parameter] {
        &self.mapped_inputs
    }
    /// Shared covectors, summed over all requested point occurrences.
    #[must_use]
    pub fn shared_cotangents(&self) -> &[f64] {
        &self.shared
    }
    /// Mapped covectors without reducing the point axes.
    #[must_use]
    pub fn mapped_cotangents(&self) -> &[f64] {
        &self.mapped
    }
    /// Unmodified accepted per-position actions supplying all sums and evidence.
    #[must_use]
    pub fn products(&self) -> &[DifferentiableVjp] {
        &self.products
    }
}

fn assemble_tangent(
    width: usize,
    shared: &[usize],
    mapped: &[usize],
    ds: &[f64],
    dx: &[f64],
) -> Vec<f64> {
    let mut tangent = vec![0.0; width];
    for (&index, &value) in shared.iter().zip(ds).chain(mapped.iter().zip(dx)) {
        tangent[index] = value;
    }
    tangent
}

fn accumulate(
    gradient: &[f64],
    shared: &[usize],
    mapped: &[usize],
    ds: &mut [f64],
    dx: &mut [f64],
) -> Result<(), Diagnostic> {
    for (sum, &index) in ds.iter_mut().zip(shared) {
        *sum += gradient[index];
        if !sum.is_finite() {
            return Err(invalid("shared VJP accumulation is nonfinite"));
        }
    }
    for (value, &index) in dx.iter_mut().zip(mapped) {
        *value = gradient[index];
    }
    Ok(())
}

fn validate_evidence(
    evaluation: &crate::DifferentiableEvaluation,
    evidence: &DifferentiationEvidence,
    mode: DifferentiationMode,
) -> Result<(), Diagnostic> {
    let primal = evaluation.primal_evidence();
    if evidence.identity() != evaluation.identity()
        || evidence.point().signatures() != evaluation.point().signatures()
        || !super::exact_values(
            evidence.point().coordinates(),
            evaluation.point().coordinates(),
        )
        || evidence.mode() != mode
        || evidence.linearization_state() != LinearizationState::Reused
        || evidence.state_system() != primal.state_system()
        || evidence.receipt() != primal.receipt()
        || evidence.accepted_state() != primal.accepted_state()
        || evidence.accepted_output() != primal.accepted_output()
    {
        return Err(invalid(
            "mapped product contains a foreign or unpaired linearization",
        ));
    }
    Ok(())
}

fn numerical_bytes(
    cells: usize,
    seeds: usize,
    outputs: usize,
    inputs: usize,
    mapped: usize,
    shared: usize,
) -> Result<usize, Diagnostic> {
    let per_cell = multiply(outputs, 3)?
        .checked_add(multiply(inputs, 3)?)
        .and_then(|n| n.checked_add(mapped))
        .ok_or_else(overflow)?;
    multiply(
        multiply(cells, per_cell)?
            .checked_add(multiply(seeds, shared)?)
            .ok_or_else(overflow)?,
        size_of::<f64>(),
    )
}
fn multiply(a: usize, b: usize) -> Result<usize, Diagnostic> {
    a.checked_mul(b).ok_or_else(overflow)
}
fn addressable<T>(count: usize) -> Result<(), Diagnostic> {
    if multiply(count, size_of::<T>())? > isize::MAX as usize {
        return Err(overflow());
    }
    Ok(())
}
fn overflow() -> Diagnostic {
    invalid("mapped product dimensions or numerical bytes overflow usize")
}
fn axis_error(error: axes::Error) -> Diagnostic {
    invalid(format!("invalid mapped product axes: {error:?}"))
}
fn admit(values: &[f64], expected: usize) -> Result<(), Diagnostic> {
    if values.len() != expected {
        return Err(invalid(
            "mapped product direction shape differs from admitted axes/coordinates",
        ));
    }
    finite(values)
}
fn finite(values: &[f64]) -> Result<(), Diagnostic> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("mapped product input or output is nonfinite"));
    }
    Ok(())
}
fn row(values: &[f64], row: usize, width: usize) -> &[f64] {
    &values[row * width..(row + 1) * width]
}
fn row_mut(values: &mut [f64], row: usize, width: usize) -> &mut [f64] {
    &mut values[row * width..(row + 1) * width]
}
fn with_components(shape: &[usize], components: usize) -> Vec<usize> {
    shape.iter().copied().chain([components]).collect()
}
fn coordinates(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coordinates = vec![0; shape.len()];
    for (coordinate, &extent) in coordinates.iter_mut().zip(shape).rev() {
        *coordinate = index % extent;
        index /= extent;
    }
    coordinates
}
// This scalar describes record slots, not physical derivative component units.
fn record_type() -> ValueType {
    ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).expect("valid scalar type")
}
fn grid_limits() -> axes::Limits {
    axes::Limits {
        rank: 32,
        values: usize::MAX,
        bytes: usize::MAX,
    }
}

#[cfg(test)]
mod tests;
