use eqiora_assembly::{AssemblyMap, DofId, LocalUnknown};
use eqiora_core::Diagnostic;
use eqiora_meshing::{MeshEntity, MeshTopology, SimplicialMesh};

use super::{
    CELL_LOCAL_DOF_COUNT, COMPONENTS, CONSTRAINT_LOCAL_DOF_COUNT, FACET_LOCAL_DOF_COUNT,
    P1_BASIS_COUNT, invalid,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GaugeLayout {
    reduced: usize,
    full: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MixedLayout {
    pub(crate) vertex_count: usize,
    cell_count: usize,
    reduced_vertex_velocity: Vec<[Option<DofId>; COMPONENTS]>,
    reduced_bubble_offset: usize,
    reduced_pressure_offset: usize,
    gauge: Option<GaugeLayout>,
    pub(crate) reduced_size: usize,
    full_bubble_offset: usize,
    pub(crate) full_pressure_offset: usize,
    pub(crate) full_size: usize,
}

impl MixedLayout {
    pub(crate) fn new(
        mesh: &SimplicialMesh,
        fixed_velocity: &[Option<[f64; COMPONENTS]>],
        with_zero_integral_constraint: bool,
    ) -> Result<Self, Diagnostic> {
        let vertex_count = mesh.vertices().len();
        if fixed_velocity.len() != vertex_count {
            return Err(invalid(
                "MINI Stokes fixed-velocity mask does not match the mesh vertices",
            ));
        }
        let cell_count = mesh
            .entity_count(super::DIMENSION)
            .expect("2D simplex mesh owns cells");
        let mut reduced_vertex_velocity = vec![[None; COMPONENTS]; vertex_count];
        let mut next = 0_usize;
        for (vertex, components) in reduced_vertex_velocity.iter_mut().enumerate() {
            if fixed_velocity[vertex].is_none() {
                for component in components {
                    *component = Some(DofId::new(next));
                    next = checked_add(next, 1, "free velocity count")?;
                }
            }
        }
        let reduced_bubble_offset = next;
        next = checked_add(
            next,
            checked_mul(cell_count, COMPONENTS, "bubble velocity count")?,
            "mixed reduced width",
        )?;
        let reduced_pressure_offset = next;
        next = checked_add(next, vertex_count, "mixed pressure width")?;
        let reduced_gauge = with_zero_integral_constraint.then_some(next);
        let reduced_size = checked_add(
            next,
            usize::from(with_zero_integral_constraint),
            "mixed gauge width",
        )?;

        let full_bubble_offset = checked_mul(vertex_count, COMPONENTS, "full vertex velocity")?;
        let full_pressure_offset = checked_add(
            full_bubble_offset,
            checked_mul(cell_count, COMPONENTS, "full bubble velocity")?,
            "full pressure offset",
        )?;
        let full_without_gauge = checked_add(full_pressure_offset, vertex_count, "full width")?;
        let full_gauge = with_zero_integral_constraint.then_some(full_without_gauge);
        let full_size = checked_add(
            full_without_gauge,
            usize::from(with_zero_integral_constraint),
            "full gauge width",
        )?;
        let gauge = reduced_gauge
            .zip(full_gauge)
            .map(|(reduced, full)| GaugeLayout { reduced, full });
        Ok(Self {
            vertex_count,
            cell_count,
            reduced_vertex_velocity,
            reduced_bubble_offset,
            reduced_pressure_offset,
            gauge,
            reduced_size,
            full_bubble_offset,
            full_pressure_offset,
            full_size,
        })
    }

    pub(crate) fn reduced_cell_map(
        &self,
        cell: usize,
        vertices: &[MeshEntity],
        fixed_velocity: &[Option<[f64; COMPONENTS]>],
    ) -> Result<AssemblyMap<f64>, Diagnostic> {
        let mut equations = Vec::with_capacity(CELL_LOCAL_DOF_COUNT);
        let mut unknowns = Vec::with_capacity(CELL_LOCAL_DOF_COUNT);
        self.append_reduced_vertex_velocity(
            vertices.iter().take(P1_BASIS_COUNT),
            fixed_velocity,
            &mut equations,
            &mut unknowns,
        );
        for component in 0..COMPONENTS {
            let dof = DofId::new(self.reduced_bubble_offset + cell * COMPONENTS + component);
            equations.push(Some(dof));
            unknowns.push(LocalUnknown::Free(dof));
        }
        for vertex in vertices {
            let dof = DofId::new(self.reduced_pressure_offset + vertex.index());
            equations.push(Some(dof));
            unknowns.push(LocalUnknown::Free(dof));
        }
        AssemblyMap::new(equations, unknowns)
    }

    pub(crate) fn full_cell_map(
        &self,
        cell: usize,
        vertices: &[MeshEntity],
    ) -> Result<AssemblyMap<f64>, Diagnostic> {
        let mut dofs = Vec::with_capacity(CELL_LOCAL_DOF_COUNT);
        self.append_full_vertex_velocity(vertices, &mut dofs);
        for component in 0..COMPONENTS {
            dofs.push(self.full_bubble_offset + cell * COMPONENTS + component);
        }
        for vertex in vertices {
            dofs.push(self.full_pressure_offset + vertex.index());
        }
        identity_map(&dofs)
    }

    pub(crate) fn reduced_constraint_map(
        &self,
        vertices: &[MeshEntity],
    ) -> Result<AssemblyMap<f64>, Diagnostic> {
        let gauge = self
            .gauge
            .expect("constraint map exists only with a zero-integral gauge")
            .reduced;
        let mut dofs = vertices
            .iter()
            .map(|vertex| self.reduced_pressure_offset + vertex.index())
            .collect::<Vec<_>>();
        dofs.push(gauge);
        debug_assert_eq!(dofs.len(), CONSTRAINT_LOCAL_DOF_COUNT);
        identity_map(&dofs)
    }

    pub(crate) fn full_constraint_map(
        &self,
        vertices: &[MeshEntity],
    ) -> Result<AssemblyMap<f64>, Diagnostic> {
        let gauge = self
            .gauge
            .expect("constraint map exists only with a zero-integral gauge")
            .full;
        let mut dofs = vertices
            .iter()
            .map(|vertex| self.full_pressure_offset + vertex.index())
            .collect::<Vec<_>>();
        dofs.push(gauge);
        debug_assert_eq!(dofs.len(), CONSTRAINT_LOCAL_DOF_COUNT);
        identity_map(&dofs)
    }

    pub(crate) fn reduced_facet_map(
        &self,
        vertices: &[MeshEntity],
        fixed_velocity: &[Option<[f64; COMPONENTS]>],
    ) -> Result<AssemblyMap<f64>, Diagnostic> {
        let mut equations = Vec::with_capacity(FACET_LOCAL_DOF_COUNT);
        let mut unknowns = Vec::with_capacity(FACET_LOCAL_DOF_COUNT);
        self.append_reduced_vertex_velocity(
            vertices.iter(),
            fixed_velocity,
            &mut equations,
            &mut unknowns,
        );
        AssemblyMap::new(equations, unknowns)
    }

    pub(crate) fn full_facet_map(
        &self,
        vertices: &[MeshEntity],
    ) -> Result<AssemblyMap<f64>, Diagnostic> {
        let mut dofs = Vec::with_capacity(FACET_LOCAL_DOF_COUNT);
        self.append_full_vertex_velocity(vertices, &mut dofs);
        identity_map(&dofs)
    }

    fn append_reduced_vertex_velocity<'a>(
        &self,
        vertices: impl Iterator<Item = &'a MeshEntity>,
        fixed_velocity: &[Option<[f64; COMPONENTS]>],
        equations: &mut Vec<Option<DofId>>,
        unknowns: &mut Vec<LocalUnknown<f64>>,
    ) {
        for vertex in vertices {
            let vertex = vertex.index();
            for component in 0..COMPONENTS {
                let dof = self.reduced_vertex_velocity[vertex][component];
                equations.push(dof);
                unknowns.push(dof.map_or_else(
                    || {
                        LocalUnknown::Fixed(
                            fixed_velocity[vertex].expect("constrained vertex owns a fixed value")
                                [component],
                        )
                    },
                    LocalUnknown::Free,
                ));
            }
        }
    }

    fn append_full_vertex_velocity(&self, vertices: &[MeshEntity], dofs: &mut Vec<usize>) {
        for vertex in vertices {
            for component in 0..COMPONENTS {
                dofs.push(self.full_vertex_velocity(vertex.index(), component));
            }
        }
    }

    pub(crate) const fn full_vertex_velocity(&self, vertex: usize, component: usize) -> usize {
        vertex * COMPONENTS + component
    }

    pub(crate) const fn full_gauge(&self) -> Option<usize> {
        match self.gauge {
            Some(gauge) => Some(gauge.full),
            None => None,
        }
    }

    pub(crate) const fn reduced_velocity_end(&self) -> usize {
        self.reduced_pressure_offset
    }

    pub(crate) fn reconstruct(
        &self,
        values: &[f64],
        fixed_velocity: &[Option<[f64; COMPONENTS]>],
    ) -> Result<ReconstructedFields, Diagnostic> {
        if values.len() != self.reduced_size {
            return Err(invalid(
                "MINI Stokes solution width does not match its layout",
            ));
        }
        let mut vertex_values = vec![[0.0; COMPONENTS]; self.vertex_count];
        for (vertex, component_values) in vertex_values.iter_mut().enumerate() {
            for (component, output) in component_values.iter_mut().enumerate() {
                *output = self.reduced_vertex_velocity[vertex][component].map_or_else(
                    || {
                        fixed_velocity[vertex].expect("constrained vertex owns fixed velocity")
                            [component]
                    },
                    |dof| values[dof.index()],
                );
            }
        }
        let cell_bubble_values = (0..self.cell_count)
            .map(|cell| {
                std::array::from_fn(|component| {
                    values[self.reduced_bubble_offset + cell * COMPONENTS + component]
                })
            })
            .collect();
        let pressure_values = values
            [self.reduced_pressure_offset..self.reduced_pressure_offset + self.vertex_count]
            .to_vec();
        let gauge_multiplier = self.gauge.map(|gauge| values[gauge.reduced]);
        Ok((
            vertex_values,
            cell_bubble_values,
            pressure_values,
            gauge_multiplier,
        ))
    }

    pub(crate) fn initial_point(
        &self,
        fixed_velocity: &[Option<[f64; COMPONENTS]>],
        vertex_values: &[[f64; COMPONENTS]],
        cell_bubble_values: &[[f64; COMPONENTS]],
        pressure_values: &[f64],
        gauge_multiplier: Option<f64>,
    ) -> Result<Vec<f64>, Diagnostic> {
        if fixed_velocity.len() != self.vertex_count
            || vertex_values.len() != self.vertex_count
            || cell_bubble_values.len() != self.cell_count
            || pressure_values.len() != self.vertex_count
            || gauge_multiplier.is_some() != self.gauge.is_some()
        {
            return Err(invalid(
                "MINI initial fields do not match the selected mixed layout",
            ));
        }
        let mut point = vec![0.0; self.reduced_size];
        for (vertex, values) in vertex_values.iter().enumerate() {
            for (component, value) in values.iter().copied().enumerate() {
                if let Some(dof) = self.reduced_vertex_velocity[vertex][component] {
                    point[dof.index()] = value;
                } else {
                    let prescribed = fixed_velocity[vertex]
                        .expect("a constrained layout vertex owns prescribed values")[component];
                    let tolerance = 4096.0 * f64::EPSILON * (1.0 + prescribed.abs());
                    if (value - prescribed).abs() > tolerance {
                        return Err(invalid(
                            "MINI initial velocity violates an essential boundary value",
                        ));
                    }
                }
            }
        }
        for (cell, values) in cell_bubble_values.iter().enumerate() {
            for (component, value) in values.iter().copied().enumerate() {
                point[self.reduced_bubble_offset + cell * COMPONENTS + component] = value;
            }
        }
        point[self.reduced_pressure_offset..self.reduced_pressure_offset + self.vertex_count]
            .copy_from_slice(pressure_values);
        if let (Some(gauge), Some(multiplier)) = (self.gauge, gauge_multiplier) {
            point[gauge.reduced] = multiplier;
        }
        Ok(point)
    }

    pub(crate) fn fill_full_values(
        &self,
        output: &mut [f64],
        vertex_values: &[[f64; COMPONENTS]],
        cell_bubble_values: &[[f64; COMPONENTS]],
        pressure_values: &[f64],
        gauge_multiplier: Option<f64>,
    ) {
        for (vertex, values) in vertex_values.iter().enumerate() {
            for component in 0..COMPONENTS {
                output[self.full_vertex_velocity(vertex, component)] = values[component];
            }
        }
        for (cell, values) in cell_bubble_values.iter().enumerate() {
            for component in 0..COMPONENTS {
                output[self.full_bubble_offset + cell * COMPONENTS + component] = values[component];
            }
        }
        output[self.full_pressure_offset..self.full_pressure_offset + self.vertex_count]
            .copy_from_slice(pressure_values);
        match (self.gauge, gauge_multiplier) {
            (Some(gauge), Some(multiplier)) => output[gauge.full] = multiplier,
            (None, None) => {}
            _ => unreachable!("layout and reconstructed gauge evidence agree"),
        }
    }
}

type ReconstructedFields = (
    Vec<[f64; COMPONENTS]>,
    Vec<[f64; COMPONENTS]>,
    Vec<f64>,
    Option<f64>,
);

fn identity_map(dofs: &[usize]) -> Result<AssemblyMap<f64>, Diagnostic> {
    AssemblyMap::new(
        dofs.iter().map(|dof| Some(DofId::new(*dof))).collect(),
        dofs.iter()
            .map(|dof| LocalUnknown::Free(DofId::new(*dof)))
            .collect(),
    )
}

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, Diagnostic> {
    left.checked_add(right)
        .ok_or_else(|| invalid(format!("{name} overflows usize")))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, Diagnostic> {
    left.checked_mul(right)
        .ok_or_else(|| invalid(format!("{name} overflows usize")))
}
