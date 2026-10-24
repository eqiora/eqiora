//! First-order actions over retained accepted members, never a new evaluator.

use eqiora_core::{Diagnostic, ScalarType};

use super::{CompleteEvaluationMap, axes, invalid};
use crate::{
    DifferentiableCotangent, DifferentiableResult, DifferentiableTangent,
    DifferentiableValueSignature, DifferentiationEvidence, DifferentiationMode, LinearizationState,
};
use std::{borrow::Cow, sync::Arc};

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
    /// The Plan owns the explicit shared inputs and their declared order.
    /// All other inputs remain mapped in Program order. A derivative view cannot
    /// reinterpret equal point values as sharing. Inputs bind that exact Program's ordered tangent/cotangent base spaces.
    ///
    /// The byte limit bounds numerical buffers for one complete JVP and one
    /// complete VJP result, including both evidence-point copies, duplicate
    /// primal/full-gradient payloads, shared sums and typed aggregate records
    /// (even for empty base spaces). It excludes the already retained map,
    /// per-action records, signature metadata and allocator overhead,
    /// caller input buffers and transient derivative-solver scratch. It is not a
    /// process/peak-memory or total execution budget. Rank is bounded to 32;
    /// shape, byte and stride products are checked before product allocation.
    ///
    /// # Errors
    /// Rejects wrong axes/counts,
    /// unsupported scalar/derivative profiles, invalid members or byte limits.
    pub fn new(
        map: &'a CompleteEvaluationMap,
        point_shape: &[usize],
        seed_shape: &[usize],
        point_axes: &[usize],
        numerical_bytes_limit: usize,
    ) -> Result<Self, Diagnostic> {
        map.validate()?;
        let identity = map.plan().program_identity()?;
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
        addressable::<Option<DifferentiableResult<DifferentiableTangent>>>(cells)?;
        addressable::<Option<DifferentiableResult<DifferentiableCotangent>>>(cells)?;
        let (shared, mapped) =
            super::partition::coordinates(identity, map.plan().shared_inputs()?)?;
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
        let layout =
            axes::Layout::new(&[], &extents, &positions, grid_limits()).map_err(axis_error)?;
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
    /// One shared tangent product is required per seed and one mapped product
    /// per combined grid occurrence, including empty base-space products. Each
    /// product must bind the exact ordered shared/mapped input signatures.
    /// Output records follow that same combined grid, preserving each exact
    /// point's accepted evidence. No partial result is published on failure.
    ///
    /// # Errors
    /// Rejects product counts or foreign base spaces before actions, then preserves derivative
    /// diagnostics and rejects foreign evidence or nonfinite products.
    pub fn jvp(
        &self,
        shared: &[DifferentiableTangent],
        mapped: &[DifferentiableTangent],
    ) -> Result<EvaluationMapResult<DifferentiableTangent>, Diagnostic> {
        let identity = self.map.plan().program_identity()?;
        let (shared_leaves, mapped_leaves) =
            super::partition::leaf_indices(identity, self.map.plan().shared_inputs()?)?;
        admit_base_spaces(
            shared.iter().map(DifferentiableTangent::base_signatures),
            self.seeds,
            &shared_leaves
                .iter()
                .map(|&i| &identity.inputs()[i])
                .collect::<Vec<_>>(),
        )?;
        admit_base_spaces(
            mapped.iter().map(DifferentiableTangent::base_signatures),
            self.cells,
            &mapped_leaves
                .iter()
                .map(|&i| &identity.inputs()[i])
                .collect::<Vec<_>>(),
        )?;
        let mut products = vec![None; self.cells];
        for point in 0..self.map.len() {
            if self.seeds == 0 {
                break;
            }
            let evaluation = self.map.accepted_member(point)?;
            for (seed, shared_tangent) in shared.iter().enumerate() {
                let position = self.position(point, seed)?;
                let tangent = super::partition::assemble_coordinates(
                    self.map.plan().program_identity()?.input_dimension(),
                    &self.shared,
                    &self.mapped,
                    shared_tangent.coordinates(),
                    mapped[position].coordinates(),
                );
                let product = (|| {
                    let product = evaluation.jvp(&evaluation.identity().tangent_from_coordinates(&tangent)?)?;
                    validate_evidence(&evaluation, product.evidence(), DifferentiationMode::Jvp)?;
                    Ok(product)
                })().map_err(|error: Diagnostic| error.with_context(format!(
                    "mapped JVP point occurrence {point} {:?}, seed occurrence {seed} {:?}, grid offset {position}",
                    coordinates(point, &self.point_shape), coordinates(seed, &self.seed_shape)
                )))?;
                products[position] = Some(product);
            }
        }
        Ok(EvaluationMapResult {
            reverse: None,
            shape: self.shape.clone(),
            products: products.into_iter().map(Option::unwrap).collect(),
        })
    }

    /// Apply output cotangents and sum shared-input covectors, never average.
    ///
    /// One cotangent product over the exact ordered outputs is required per
    /// combined grid occurrence. Each seed's shared sum
    /// uses original request order, irrespective of axis placement. The real
    /// pairing is the existing coordinate pairing. Overflow fails closed;
    /// arbitrary regrouping or different point order need not be bit-identical.
    ///
    /// # Errors
    /// Rejects product counts or foreign base spaces before actions. Any member derivative,
    /// foreign-evidence or nonfinite-sum failure rejects the complete product.
    pub fn vjp(
        &self,
        cotangents: &[DifferentiableCotangent],
    ) -> Result<EvaluationMapResult<DifferentiableCotangent>, Diagnostic> {
        admit_base_spaces(
            cotangents
                .iter()
                .map(DifferentiableCotangent::base_signatures),
            self.cells,
            &self
                .map
                .plan()
                .program_identity()?
                .outputs()
                .iter()
                .collect::<Vec<_>>(),
        )?;
        self.vjp_with(|position, _| Ok(Cow::Borrowed(&cotangents[position])))
    }

    // Both explicit seeds and finite-statistic chain-rule seeds consume this
    // accepted occurrence once, including under Recompute retention.
    pub(crate) fn vjp_with<'s>(
        &self,
        mut seed_for: impl FnMut(
            usize,
            &crate::DifferentiableEvaluation,
        ) -> Result<Cow<'s, DifferentiableCotangent>, Diagnostic>,
    ) -> Result<EvaluationMapResult<DifferentiableCotangent>, Diagnostic> {
        let mut shared = vec![0.0; multiply(self.seeds, self.shared.len())?];
        let mut mapped = vec![0.0; multiply(self.cells, self.mapped.len())?];
        let mut products = vec![None; self.cells];
        for point in 0..self.map.len() {
            if self.seeds == 0 {
                break;
            }
            let evaluation = self.map.accepted_member(point)?;
            for seed in 0..self.seeds {
                let position = self.position(point, seed)?;
                let product = (|| {
                    let cotangent = seed_for(position, &evaluation)?;
                    let product = evaluation.vjp(&cotangent)?;
                    validate_evidence(&evaluation, product.evidence(), DifferentiationMode::Vjp)?;
                    accumulate(
                        product.derivative().coordinates(), &self.shared, &self.mapped,
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
        let identity = self.map.plan().program_identity()?;
        let (shared_leaves, mapped_leaves) =
            super::partition::leaf_indices(identity, self.map.plan().shared_inputs()?)?;
        Ok(EvaluationMapResult {
            shape: self.shape.clone(),
            reverse: Some(ReverseProducts {
                shared_shape: self.seed_shape.clone(),
                shared: Covectors::new(identity.inputs(), &shared_leaves, &shared, self.seeds)?,
                mapped: Covectors::new(identity.inputs(), &mapped_leaves, &mapped, self.cells)?,
            }),
            products: products.into_iter().map(Option::unwrap).collect(),
        })
    }

    fn position(&self, point: usize, seed: usize) -> Result<usize, Diagnostic> {
        let mut occurrence = coordinates(point, &self.point_shape);
        occurrence.extend(coordinates(seed, &self.seed_shape));
        if occurrence.is_empty() {
            occurrence.push(0);
        }
        axes::Layout::new(&[], &self.extents, &self.axes, grid_limits())
            .map_err(axis_error)?
            .offset(&occurrence, &[])
            .map_err(axis_error)
    }
}

/// Complete first-order actions in the admitted point/seed grid.
///
/// The tangent/cotangent type preserves the action's mathematical role, as in
/// `DifferentiableResult<T>`. Reverse results additionally retain shared sums and
/// occurrence-local covectors. Each action keeps its original accepted evidence.
/// More seeds do not establish higher derivatives:
///
/// ```compile_fail
/// use eqiora_api::{EvaluationMapResult, DifferentiableTangent};
/// fn second_derivative(first: EvaluationMapResult<DifferentiableTangent>) {
///     first.jvp(&[]);
/// }
/// ```
#[derive(Debug)]
pub struct EvaluationMapResult<T> {
    shape: Vec<usize>,
    products: Vec<DifferentiableResult<T>>,
    reverse: Option<ReverseProducts>,
}

impl<T> EvaluationMapResult<T> {
    /// Combined point/seed grid; each result retains its intrinsic leaves.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// One existing accepted action per grid position, including repetitions.
    #[must_use]
    pub fn products(&self) -> &[DifferentiableResult<T>] {
        &self.products
    }
}

#[derive(Debug)]
struct ReverseProducts {
    shared_shape: Vec<usize>,
    shared: Covectors,
    mapped: Covectors,
}

impl EvaluationMapResult<DifferentiableCotangent> {
    fn reverse(&self) -> &ReverseProducts {
        self.reverse
            .as_ref()
            .expect("reverse result owns its covector partition")
    }
    /// Seed axes only; each covector retains its ordered intrinsic base spaces.
    #[must_use]
    pub fn shared_shape(&self) -> &[usize] {
        &self.reverse().shared_shape
    }
    /// Combined point/seed axes only; intrinsic base spaces belong to each covector.
    #[must_use]
    pub fn mapped_shape(&self) -> &[usize] {
        &self.shape
    }
    /// Exact ordered base spaces of the shared sums, including empty seed grids.
    #[must_use]
    pub fn shared_signatures(&self) -> &[DifferentiableValueSignature] {
        &self.reverse().shared.signatures
    }
    /// Exact ordered base spaces of the mapped covectors, including empty point grids.
    #[must_use]
    pub fn mapped_signatures(&self) -> &[DifferentiableValueSignature] {
        &self.reverse().mapped.signatures
    }
    /// One typed covector per seed, summed over requested points in request order.
    #[must_use]
    pub fn shared_cotangents(&self) -> &[DifferentiableCotangent] {
        &self.reverse().shared.values
    }
    /// One typed covector per combined grid occurrence, without reducing point axes.
    #[must_use]
    pub fn mapped_cotangents(&self) -> &[DifferentiableCotangent] {
        &self.reverse().mapped.values
    }
}

/// The signature inventory remains meaningful even when the occurrence grid is empty.
#[derive(Debug)]
struct Covectors {
    signatures: Arc<[DifferentiableValueSignature]>,
    values: Vec<DifferentiableCotangent>,
}

impl Covectors {
    fn new(
        inputs: &[DifferentiableValueSignature],
        selected: &[usize],
        values: &[f64],
        count: usize,
    ) -> Result<Self, Diagnostic> {
        let signatures: Arc<[_]> = selected
            .iter()
            .map(|&index| inputs[index].clone())
            .collect();
        // Storage is only a transport into the independently selected base spaces.
        let width = crate::differentiation::signature::dimension(&signatures)?;
        admit(values, multiply(count, width)?)?;
        let values = (0..count)
            .map(|index| {
                DifferentiableCotangent::from_coordinates(
                    signatures.clone(),
                    row(values, index, width).to_vec(),
                )
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { signatures, values })
    }
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
        || !evidence.same_primal(&primal)
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
    let numerical = multiply(
        multiply(cells, per_cell)?
            .checked_add(multiply(seeds, shared)?)
            .ok_or_else(overflow)?,
        size_of::<f64>(),
    )?;
    // Empty products still retain a typed record for every seed/grid occurrence.
    let records = seeds.checked_add(cells).ok_or_else(overflow)?;
    numerical
        .checked_add(multiply(records, size_of::<DifferentiableCotangent>())?)
        .ok_or_else(overflow)
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
fn admit_base_spaces<'a>(
    products: impl ExactSizeIterator<Item = &'a [DifferentiableValueSignature]>,
    count: usize,
    expected: &[&DifferentiableValueSignature],
) -> Result<(), Diagnostic> {
    if products.len() != count {
        return Err(invalid(
            "mapped derivative product count differs from the declared grid",
        ));
    }
    for product in products {
        if !product.iter().eq(expected.iter().copied()) {
            return Err(invalid(
                "mapped derivative base spaces differ from the exact ordered signatures",
            ));
        }
    }
    Ok(())
}

fn admit(values: &[f64], expected: usize) -> Result<(), Diagnostic> {
    if values.len() != expected {
        return Err(invalid(
            "mapped product direction shape differs from admitted axes/coordinates",
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("mapped product direction is nonfinite"));
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
fn grid_limits() -> axes::Limits {
    axes::Limits {
        rank: 32,
        values: usize::MAX,
        bytes: usize::MAX,
    }
}

#[cfg(test)]
pub(super) mod tests;
