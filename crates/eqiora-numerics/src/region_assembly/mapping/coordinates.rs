//! Real numerical coordinates over the existing typed topology and constraints.
use super::assembly::MappedRegionAssembly;
use super::*;
use crate::form_compiler::region::coordinates::{self, Real};
use crate::region_assembly::{PreparedRegionAssembly, RegionAssemblyCell};
use eqiora_assembly::{
    AssemblyPacketSetIdentityV1, AssemblyPlan, AssemblyTarget, TargetAssemblyMap,
};
use std::ops::Range;

/// The physical map remains the owner of Field identity, quotienting, orientation
/// and scaling. Only scalar storage changes; physical components are never doubled.
pub(crate) struct RegionRealCoordinates<'a, S: Coefficient> {
    mapping: &'a RegionDofMap<S>,
    ranges: Vec<Range<usize>>,
    constraints: ConstrainedDofLayout<f64>,
}

impl<'a, S: Coefficient + Send + Sync> RegionRealCoordinates<'a, S> {
    pub(crate) fn new(mapping: &'a RegionDofMap<S>) -> Result<Self, Diagnostic> {
        let mut widths = vec![None; mapping.full_count];
        for (field, global) in mapping
            .globals
            .iter()
            .map(|(key, index)| (key.field, *index))
            .chain(mapping.gauges.iter().map(|(field, index)| (*field, *index)))
        {
            let width = mapping.fields[&field].1.real_scalar_coordinates()?;
            if widths[global].is_some_and(|previous| previous != width) {
                return Err(invalid(
                    "quotiented coefficients have different scalar domains",
                ));
            }
            widths[global] = Some(width);
        }
        let prescribed = mapping
            .constraints
            .lift(&vec![S::zero(); mapping.free_count()])?;
        let mut ranges = Vec::with_capacity(widths.len());
        let mut fixed = Vec::new();
        for (index, width) in widths.into_iter().enumerate() {
            let width =
                width.ok_or_else(|| invalid("scalar coordinate has no exact Field owner"))?;
            let start = fixed.len();
            let end = start
                .checked_add(width)
                .ok_or_else(|| invalid("real coordinate count overflows usize"))?;
            let value = prescribed[index];
            if width == 1 && value.im() != 0. {
                return Err(invalid(
                    "a prescribed real Field has an imaginary coefficient",
                ));
            }
            let free = mapping.constraints.is_free(index)?;
            for part in [value.re(), value.im()].into_iter().take(width) {
                fixed.push((!free).then_some(part));
            }
            ranges.push(start..end);
        }
        Ok(Self {
            mapping,
            ranges,
            constraints: ConstrainedDofLayout::new(fixed)?,
        })
    }

    pub(crate) fn full_count(&self) -> usize {
        self.ranges.last().map_or(0, |range| range.end)
    }

    /// Canonical representative of every full real coordinate after quotienting.
    pub(crate) fn full_keys(&self) -> Vec<(FieldDof, bool)> {
        let mut keys = vec![None; self.full_count()];
        for key in self.mapping.keys() {
            for (part, global) in self.global_coordinates(key).enumerate() {
                keys[global].get_or_insert((key, part == 1));
            }
        }
        keys.into_iter()
            .map(|key| key.expect("mapped full coordinate"))
            .collect()
    }

    pub(crate) fn free_count(&self) -> usize {
        self.constraints.free_count()
    }

    /// Independent rates use analytic prescribed trace rates, never the fixed
    /// State values retained in the coordinate map.
    pub(crate) fn lift_rates(
        &self,
        reduced: &[f64],
        prescribed: &[f64],
    ) -> Result<Vec<f64>, Diagnostic> {
        if prescribed.len() != self.full_count()
            || prescribed.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                "prescribed rates require exact finite full coordinates",
            ));
        }
        let mut full = self.constraints.lift_direction(reduced)?;
        for (global, value) in full.iter_mut().enumerate() {
            if self.free_coordinate(global).is_none() {
                *value = prescribed[global];
            }
        }
        Ok(full)
    }

    pub(crate) fn global_coordinates(&self, key: FieldDof) -> Range<usize> {
        self.ranges[self.mapping.global_dof(key).expect("mapped Field DOF")].clone()
    }

    pub(crate) fn free_coordinate(&self, global: usize) -> Option<usize> {
        self.constraints
            .free_index(global)
            .map(|index| index.index())
    }

    pub(crate) fn restrict_action(&self, values: &[f64]) -> Result<Vec<f64>, Diagnostic> {
        self.constraints.restrict_action(values)
    }

    pub(crate) fn encode_full(&self, values: &[S]) -> Result<Vec<f64>, Diagnostic> {
        if values.len() != self.ranges.len() || values.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "full physical coordinates have invalid size or values",
            ));
        }
        let mut result = Vec::with_capacity(self.full_count());
        for (value, range) in values.iter().zip(&self.ranges) {
            if range.len() == 1 && value.im() != 0. {
                return Err(invalid("a real Field has an imaginary coefficient"));
            }
            result.extend([value.re(), value.im()].into_iter().take(range.len()));
        }
        Ok(result)
    }

    /// Expand only storage indices. Both parts inherit the physical functional's
    /// orientation, including edge and face moments.
    fn cell_coordinates(&self, cell: usize) -> Result<(Vec<usize>, Vec<i8>), Diagnostic> {
        let globals = self
            .mapping
            .cells
            .get(cell)
            .ok_or_else(|| invalid("real coordinate cell is outside its exact mapping"))?;
        let mut coordinates = Vec::new();
        let mut signs = Vec::new();
        for (&global, &sign) in globals.iter().zip(self.mapping.cell_signs(cell)?) {
            let range = self.ranges[global].clone();
            signs.extend(std::iter::repeat_n(sign, range.len()));
            coordinates.extend(range);
        }
        Ok((coordinates, signs))
    }

    /// Validate physical values through their existing owner, then encode only
    /// independent coordinates. A real Field cannot hide an imaginary part.
    pub(crate) fn validated_algebraic_point(
        &self,
        fields: &BTreeMap<RawId, RecoveredRegionField<S>>,
    ) -> Result<Vec<f64>, Diagnostic> {
        self.encode(&self.mapping.validated_algebraic_point(fields)?)
    }

    pub(crate) fn encode(&self, values: &[S]) -> Result<Vec<f64>, Diagnostic> {
        if values.len() != self.mapping.free_count() || values.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "physical reduced coordinates have invalid size or values",
            ));
        }
        let mut coordinates = Vec::with_capacity(self.constraints.free_count());
        for (global, value) in self
            .mapping
            .constraints
            .free_globals()
            .into_iter()
            .zip(values.iter().copied())
        {
            let width = self.ranges[global].len();
            if width == 1 && value.im() != 0. {
                return Err(invalid("a real Field has an imaginary coefficient"));
            }
            coordinates.extend([value.re(), value.im()].into_iter().take(width));
        }
        Ok(coordinates)
    }

    /// Restore physical scalar values before invoking the single typed Field
    /// recovery owner. Its scales, essential values and trace quotients apply once.
    pub(crate) fn recover(
        &self,
        reduced: &[f64],
    ) -> Result<BTreeMap<RawId, RecoveredRegionField<S>>, Diagnostic> {
        self.mapping.recover(
            &self.decode(reduced)?,
            &self.mapping.fields.keys().copied().collect::<Vec<_>>(),
        )
    }

    pub(crate) fn decode(&self, reduced: &[f64]) -> Result<Vec<S>, Diagnostic> {
        let full = self.constraints.lift(reduced)?;
        self.mapping
            .constraints
            .free_globals()
            .into_iter()
            .map(|global| {
                let range = &self.ranges[global];
                S::from_parts(
                    full[range.start],
                    if range.len() == 2 {
                        full[range.start + 1]
                    } else {
                        0.
                    },
                )
                .ok_or_else(|| {
                    invalid("physical scalar cannot represent its exact Field coordinates")
                })
            })
            .collect()
    }

    pub(super) fn require_multiplier_closure(
        &self,
        matrix: &eqiora_assembly::CsrMatrix<f64>,
        multipliers: &BTreeSet<RawId>,
    ) -> Result<(), Diagnostic> {
        let globals = self.mapping.constraints.free_globals();
        let mut projected = Vec::new();
        for mode in self.mapping.constant_multiplier_modes(multipliers)? {
            // Every connected mode belongs to one Field; both scalar directions
            // of a complex constant must have independent nonzero images.
            let width = self.ranges[globals[mode[0]]].len();
            for part in 0..width {
                projected.push(
                    mode.iter()
                        .map(|&index| {
                            let global = self.ranges[globals[index]].start + part;
                            self.constraints
                                .free_index(global)
                                .expect("free multiplier mode")
                                .index()
                        })
                        .collect(),
                );
            }
        }
        super::multiplier_closure::require_independent_actions(matrix, &projected)
    }

    /// Project static local actions before constraint elimination and scatter.
    /// No complex-linear CSR conversion can represent a conjugated trial action.
    pub(crate) fn assemble_static<'mesh>(
        &self,
        mesh: &'mesh impl eqiora_meshing::MeshGeometry<Map<'mesh> = eqiora_meshing::AffineGeometryMap>,
        input: &RegionSolveInput<S>,
    ) -> Result<
        (
            eqiora_assembly::LinearSystem<f64>,
            eqiora_assembly::LinearSystem<f64>,
            eqiora_assembly::AssemblyReport,
        ),
        Diagnostic,
    > {
        self.with_static_work(mesh, input, |plan, work| {
            use eqiora_assembly::AssemblyBackend;
            let (systems, report) = eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND
                .assemble(plan, work)?
                .into_parts();
            let mut systems = systems.into_iter();
            let reduced = systems.next().expect("validated reduced target");
            let full = systems.next().expect("validated full target");
            debug_assert!(systems.next().is_none());
            Ok((reduced, full, report))
        })
    }

    pub(crate) fn with_static_work<'mesh, R>(
        &self,
        mesh: &'mesh impl eqiora_meshing::MeshGeometry<Map<'mesh> = eqiora_meshing::AffineGeometryMap>,
        input: &RegionSolveInput<S>,
        consume: impl FnOnce(
            &eqiora_assembly::AssemblyPlan,
            &dyn eqiora_assembly::AssemblyWork<f64>,
        ) -> Result<R, Diagnostic>,
    ) -> Result<R, Diagnostic> {
        if input.previous.is_some()
            || input.geometry_action.is_some()
            || input
                .forms
                .iter()
                .any(|(form, _)| form.time_binding().is_some())
        {
            return Err(invalid(
                "real-coordinate assembly has no temporal action binding yet",
            ));
        }
        let prepared = self.prepare_assembly(mesh, input)?;
        consume(&prepared.plan, &prepared.work)
    }

    /// Keep physical topology, history and geometry with their existing map;
    /// expand only scalar storage before the shared prepared assembly owner.
    pub(crate) fn prepare_assembly<'mesh>(
        &self,
        mesh: &'mesh impl eqiora_meshing::MeshGeometry<Map<'mesh> = eqiora_meshing::AffineGeometryMap>,
        input: &RegionSolveInput<S>,
    ) -> Result<MappedRegionAssembly<S, Real>, Diagnostic> {
        self.mapping.validate_solve_input(mesh, input)?;
        if !self.mapping.gauges.is_empty() {
            return Err(invalid(
                "real-coordinate assembly has no integral-constraint packets yet",
            ));
        }
        let physical_plan = self.mapping.assembly_plan()?;
        let physical_cells = self.mapping.assembly_cells(
            mesh,
            &input.forms,
            input.previous.as_ref(),
            input.geometry_action.as_deref(),
            &physical_plan,
        )?;
        let plan = AssemblyPlan::new(vec![
            AssemblyTarget::new(self.constraints.free_count())?,
            AssemblyTarget::new(self.ranges.last().expect("nonempty mapped Fields").end)?,
        ])?;
        let forms = input
            .forms
            .iter()
            .map(|(form, _)| (form.domain(), form))
            .collect::<BTreeMap<_, _>>();
        let mut cells = Vec::with_capacity(physical_cells.len());
        for cell in physical_cells {
            let (globals, orientation) = self.cell_coordinates(cell.index)?;
            let form = forms[&self.mapping.cell_domains[cell.index]];
            let previous = cell
                .previous
                .into_iter()
                .map(|(id, values)| {
                    let field = &form.previous_fields()[&id].value_type;
                    Ok((id, coordinates::values::<S, Real>(field, &values)?))
                })
                .collect::<Result<_, Diagnostic>>()?;
            cells.push(RegionAssemblyCell {
                index: cell.index,
                geometry: cell.geometry,
                previous_geometry: cell.previous_geometry,
                orientation,
                previous,
                mappings: vec![
                    TargetAssemblyMap::new(
                        plan.target_id(0).expect("reduced target"),
                        self.constraints.reduced_map(&globals)?,
                    ),
                    TargetAssemblyMap::new(
                        plan.target_id(1).expect("full target"),
                        self.constraints.full_map(&globals)?,
                    ),
                ]
                .into(),
            });
        }
        let natural = input
            .natural
            .iter()
            .map(|(index, facet)| Ok((*index, facet.clone().project::<Real>()?)))
            .collect::<Result<_, Diagnostic>>()?;
        let mut work = PreparedRegionAssembly::<S, Real>::new(
            AssemblyPacketSetIdentityV1::Unbound,
            &plan,
            input.forms.clone(),
            &self.mapping.cell_domains,
            cells,
            natural,
        )?;
        if let Some(faces) = &input.conservative {
            work.append_conservative(&plan, &input.forms, faces, input.operator_properties)?;
        }
        let packet_domains = work.packet_domains();
        Ok(MappedRegionAssembly {
            plan,
            work,
            packet_domains,
        })
    }
}
