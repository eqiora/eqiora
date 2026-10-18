//! Real numerical coordinates over the existing typed topology and constraints.
use super::*;
use num_complex::Complex64 as C;
use std::ops::Range;

/// The physical map remains the owner of Field identity, quotienting, orientation
/// and scaling. Only scalar storage changes; physical components are never doubled.
pub(crate) struct RegionRealCoordinates<'a> {
    mapping: &'a RegionDofMap<C>,
    ranges: Vec<Range<usize>>,
    constraints: ConstrainedDofLayout<f64>,
}

impl<'a> RegionRealCoordinates<'a> {
    pub(crate) fn new(mapping: &'a RegionDofMap<C>) -> Result<Self, Diagnostic> {
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
            .lift(&vec![C::new(0., 0.); mapping.free_count()])?;
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
            if width == 1 && value.im != 0. {
                return Err(invalid(
                    "a prescribed real Field has an imaginary coefficient",
                ));
            }
            let free = mapping.constraints.is_free(index)?;
            for part in [value.re, value.im].into_iter().take(width) {
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
        fields: &BTreeMap<RawId, RecoveredRegionField<C>>,
    ) -> Result<Vec<f64>, Diagnostic> {
        let values = self.mapping.validated_algebraic_point(fields)?;
        let mut coordinates = Vec::with_capacity(self.constraints.free_count());
        for (global, value) in self
            .mapping
            .constraints
            .free_globals()
            .into_iter()
            .zip(values)
        {
            let width = self.ranges[global].len();
            if width == 1 && value.im != 0. {
                return Err(invalid("a real Field has an imaginary coefficient"));
            }
            coordinates.extend([value.re, value.im].into_iter().take(width));
        }
        Ok(coordinates)
    }

    /// Restore physical scalar values before invoking the single typed Field
    /// recovery owner. Its scales, essential values and trace quotients apply once.
    pub(crate) fn recover(
        &self,
        reduced: &[f64],
    ) -> Result<BTreeMap<RawId, RecoveredRegionField<C>>, Diagnostic> {
        let full = self.constraints.lift(reduced)?;
        let physical_reduced = self
            .mapping
            .constraints
            .free_globals()
            .into_iter()
            .map(|global| {
                let range = &self.ranges[global];
                C::new(
                    full[range.start],
                    if range.len() == 2 {
                        full[range.start + 1]
                    } else {
                        0.
                    },
                )
            })
            .collect::<Vec<_>>();
        self.mapping.recover(
            &physical_reduced,
            &self.mapping.fields.keys().copied().collect::<Vec<_>>(),
        )
    }

    /// Project static local actions before constraint elimination and scatter.
    /// No complex-linear CSR conversion can represent a conjugated trial action.
    pub(crate) fn assemble_static<'mesh>(
        &self,
        mesh: &'mesh impl eqiora_meshing::MeshGeometry<Map<'mesh> = eqiora_meshing::AffineGeometryMap>,
        input: &RegionSolveInput<C>,
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
        input: &RegionSolveInput<C>,
        consume: impl FnOnce(
            &eqiora_assembly::AssemblyPlan,
            &dyn eqiora_assembly::AssemblyWork<f64>,
        ) -> Result<R, Diagnostic>,
    ) -> Result<R, Diagnostic> {
        self.mapping.validate_solve_input(mesh, input)?;
        if !self.mapping.gauges.is_empty() {
            return Err(invalid(
                "real-coordinate assembly has no integral-constraint packets yet",
            ));
        }
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
        let plan = self.mapping.assembly_plan()?;
        let cells = self
            .mapping
            .assembly_cells(mesh, &input.forms, None, None, &plan)?;
        let forms = input
            .forms
            .iter()
            .map(|(form, rule)| (form.domain(), (form, rule)))
            .collect::<BTreeMap<_, _>>();
        let mut packets = Vec::new();
        for cell in &cells {
            let (form, rule) = forms[&self.mapping.cell_domains[cell.index]];
            let (local, _) = form.prepare_real_affine_actions(&cell.geometry, None, rule)?;
            let (globals, signs) = self.cell_coordinates(cell.index)?;
            packets.push((local.reoriented(&signs, &signs)?, globals));
        }
        for (index, facet) in &input.natural {
            let cell = cells
                .get(*index)
                .ok_or_else(|| invalid("natural packet has a foreign cell"))?;
            let (form, _) = forms[&self.mapping.cell_domains[*index]];
            facet.validate_parent(form, &cell.geometry)?;
            let local = facet.real_coordinate_load()?;
            let (globals, signs) = self.cell_coordinates(*index)?;
            packets.push((local.reoriented(&signs, &signs)?, globals));
        }
        self.constraints
            .with_work(packets.len(), |index| Ok(packets[index].clone()), consume)
    }
}
