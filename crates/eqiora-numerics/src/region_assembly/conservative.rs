//! Native conservative faces join the same mapped volume and rate actions.
use super::*;
use crate::canonical_boundary::PhysicalBoundaryQuantity;
use crate::cartesian_elliptic::CartesianBoundaryValue;
use crate::cartesian_elliptic::{FvmFlux, cartesian_flux_contributions};
use crate::form_compiler::region::RegionBoundaryLaw;
use eqiora_assembly::{AssemblyMap, LocalContribution};
use eqiora_meshing::{CartesianMesh, MeshTopology};
use eqiora_schema::kernel::BoundarySide;
use eqiora_solver::LinearOperatorProperties;

/// Geometry and prescribed face data; equation coefficients remain in the bound form.
pub(crate) struct ConservativeFacets<S: Coefficient> {
    pub(crate) mesh: CartesianMesh,
    pub(crate) boundaries: BTreeMap<(RawId, usize, BoundarySide), RegionBoundaryLaw<S>>,
    pub(crate) primal: Option<Box<ConservativePrimal<S>>>,
}

/// A flux derivative needs both the original constitutive law and its direction.
pub(crate) struct ConservativePrimal<S: Coefficient> {
    form: BoundRegionForm<S>,
    boundaries: BTreeMap<(RawId, usize, BoundarySide), RegionBoundaryLaw<S>>,
}

impl<S: Coefficient> ConservativeFacets<S> {
    pub(crate) fn retain_primal(
        &mut self,
        form: &BoundRegionForm<S>,
        faces: &Self,
    ) -> Result<(), Diagnostic> {
        if self.mesh != faces.mesh || self.primal.is_some() || faces.primal.is_some() {
            return Err(invalid(
                "conservative input partial requires its unchanged primal mesh",
            ));
        }
        self.primal = Some(Box::new(ConservativePrimal {
            form: form.clone(),
            boundaries: faces.boundaries.clone(),
        }));
        Ok(())
    }

    fn boundary(
        &self,
        boundaries: &BTreeMap<(RawId, usize, BoundarySide), RegionBoundaryLaw<S>>,
        field: RawId,
        component: usize,
        (axis, side, point): (usize, BoundarySide, &[f64]),
    ) -> Result<CartesianBoundaryValue<S>, Diagnostic> {
        let law = boundaries
            .get(&(field, axis, side))
            .ok_or_else(|| invalid("conservative facet lacks its exact boundary law"))?;
        let mut normal = vec![0.; self.mesh.topological_dimension()];
        normal[axis] = match side {
            BoundarySide::Lower => -1.,
            BoundarySide::Upper => 1.,
        };
        let values = law.evaluate(point, &normal)?;
        let value = *values
            .get(component)
            .ok_or_else(|| invalid("conservative boundary omits its exact Field component"))?;
        match law.quantity {
            PhysicalBoundaryQuantity::Trace if law.trace_field == Some(field) => {
                Ok(CartesianBoundaryValue::Essential(value))
            }
            PhysicalBoundaryQuantity::Flux if law.trace_field.is_none() => {
                Ok(CartesianBoundaryValue::Natural(value))
            }
            _ => Err(invalid(
                "conservative boundary differs from its exact tested Field",
            )),
        }
    }
}

impl<S: Coefficient + Send + Sync, P: Projection<S>> PreparedRegionAssembly<S, P>
where
    P::Scalar: Send + Sync,
{
    pub(crate) fn append_conservative(
        &mut self,
        plan: &AssemblyPlan,
        forms: &[(BoundRegionForm<S>, QuadratureRule)],
        faces: &ConservativeFacets<S>,
        properties: LinearOperatorProperties,
    ) -> Result<(), Diagnostic> {
        let [(form, _)] = forms else {
            return Err(invalid(
                "conservative face binding requires its exact single Region",
            ));
        };
        let fields = form.fields();
        if fields
            .iter()
            .any(|field| field.space != eqiora_realization::Space::cell_constant())
        {
            return Err(invalid(
                "conservative faces require cell-constant Field coordinates",
            ));
        }
        let primal = faces.primal.as_deref();
        let row = primal
            .map_or(form, |primal| &primal.form)
            .conservative_flux()?
            .project::<P>()?;
        let partial = primal
            .map(|_| form.conservative_flux()?.project::<P>())
            .transpose()?;
        if !row.has_flux() {
            return Ok(());
        }
        let ranges = P::ranges(fields)?;
        let coordinate = |component| -> Result<(usize, f64, RawId), Diagnostic> {
            let (field, component, part) = row.coordinate(component)?;
            let index = fields
                .iter()
                .position(|layout| layout.field == field)
                .ok_or_else(|| {
                    invalid("conservative coordinate is absent from its bound Field layout")
                })?;
            Ok((
                ranges[index].start + component * P::width(&fields[index].value_type)? + part,
                fields[index].scale,
                field,
            ))
        };
        let coefficient =
            |test, trial, axis, point: &[f64]| row.diffusion(test, trial, axis, point);
        let prescribed = |test, axis, point: &[f64]| row.prescribed_flux(test, axis, point);
        let boundary_value = |boundaries: &BTreeMap<_, _>, component, axis, side, point: &[f64]| {
            let (field, component, part) = row.coordinate(component)?;
            Ok(
                match faces.boundary(boundaries, field, component, (axis, side, point))? {
                    CartesianBoundaryValue::Essential(value) => {
                        CartesianBoundaryValue::Essential(P::component(value, part))
                    }
                    CartesianBoundaryValue::Natural(value) => {
                        CartesianBoundaryValue::Natural(P::component(value, part))
                    }
                },
            )
        };
        let boundary = |component, axis, side, point: &[f64]| {
            boundary_value(
                primal.map_or(&faces.boundaries, |primal| &primal.boundaries),
                component,
                axis,
                side,
                point,
            )
        };
        let coefficient_partial = |test, trial, axis, point: &[f64]| {
            partial
                .as_ref()
                .expect("selected partial")
                .diffusion(test, trial, axis, point)
        };
        let prescribed_partial = |test, axis, point: &[f64]| {
            partial
                .as_ref()
                .expect("selected partial")
                .prescribed_flux(test, axis, point)
        };
        let boundary_partial = |component, axis, side, point: &[f64]| {
            boundary_value(&faces.boundaries, component, axis, side, point)
        };
        let dimension = faces.mesh.topological_dimension();
        let rule = if dimension == 1 {
            QuadratureRule::point()
        } else {
            QuadratureRule::tensor_product_gauss_legendre(dimension - 1, 1)?
        };
        let mut flux = FvmFlux::new(&coefficient, &prescribed, &boundary);
        if partial.is_some() {
            flux =
                flux.with_direction(&coefficient_partial, &prescribed_partial, &boundary_partial);
        }
        let contributions =
            cartesian_flux_contributions(&faces.mesh, row.components(), flux, &rule, properties)?;
        for face in contributions {
            let (test, _, tested) = coordinate(face.test)?;
            let (trial, trial_scale, _) = coordinate(face.trial)?;
            let row_scale = form.physical_dual_scale(tested)?.value().recip();
            let cells = face
                .cells
                .iter()
                .map(|&cell| self.packet(cell))
                .collect::<Result<Vec<_>, _>>()?;
            if cells.iter().any(|cell| {
                cell.domain != form.domain()
                    || cell.orientation.get(test) != Some(&1)
                    || cell.orientation.get(trial) != Some(&1)
            }) {
                return Err(invalid(
                    "conservative face differs from its exact cell coordinate maps",
                ));
            }
            let mappings = cells[0]
                .mappings
                .iter()
                .map(|target| {
                    let maps = cells
                        .iter()
                        .map(|cell| {
                            cell.mappings
                                .iter()
                                .find(|mapping| mapping.target() == target.target())
                                .map(|mapping| mapping.map())
                                .ok_or_else(|| {
                                    invalid("conservative cells disagree on assembly targets")
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(TargetAssemblyMap::new(
                        target.target(),
                        AssemblyMap::new(
                            maps.iter().map(|map| map.equations()[test]).collect(),
                            maps.iter().map(|map| map.unknowns()[trial]).collect(),
                        )?,
                    ))
                })
                .collect::<Result<Vec<_>, Diagnostic>>()?;
            let count = face.cells.len();
            validate_maps(plan, count, &mappings)?;
            let local = LocalContribution::new(
                count,
                count,
                face.local
                    .matrix()
                    .iter()
                    .map(|value| *value * <P::Scalar as From<f64>>::from(row_scale * trial_scale))
                    .collect(),
                face.local
                    .rhs()
                    .iter()
                    .map(|value| *value * <P::Scalar as From<f64>>::from(row_scale))
                    .collect(),
            )?;
            self.packets.push(PreparedRegionPacket {
                domain: form.domain(),
                orientation: vec![1; count],
                mappings: mappings.into(),
                action: RegionPacketAction::Affine(local),
            });
        }
        Ok(())
    }
}
