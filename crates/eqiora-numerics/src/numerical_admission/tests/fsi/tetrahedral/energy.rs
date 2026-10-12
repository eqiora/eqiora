//! Analytic barycentric moments, independent of bound forms and basis tabulation.
use super::*;
use eqiora_meshing::MeshGeometry;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn factorial(n: usize) -> f64 {
    (1..=n).map(|i| i as f64).product()
}
// Volume-normalized Dirichlet integral: average(prod lambda_i^a_i)=3! prod(a_i!)/(3+sum a_i)!.
fn average(powers: [usize; 4]) -> f64 {
    6. * powers.iter().map(|&p| factorial(p)).product::<f64>()
        / factorial(3 + powers.iter().sum::<usize>())
}
fn sym_squared(g: &[[f64; 3]; 3]) -> f64 {
    (0..3)
        .flat_map(|i| (0..3).map(move |j| 0.25 * (g[i][j] + g[j][i]).powi(2)))
        .sum()
}

fn energy(mesh: &SimplicialMesh, fields: Fields, values: &Snapshot) -> [f64; 3] {
    let mut kinetic = 0.;
    let mut elastic = 0.;
    let mut viscous = 0.;
    for (cell, vertices) in mesh.cells().iter().enumerate() {
        let fluid = vertices
            .iter()
            .all(|&v| values[&fields.fluid_velocity.erase()].contains_key(&MeshEntity::new(0, v)));
        let field = if fluid {
            fields.fluid_velocity
        } else {
            fields.solid_velocity
        };
        let velocity = vertices
            .iter()
            .map(|&v| &values[&field.erase()][&MeshEntity::new(0, v)])
            .collect::<Vec<_>>();
        let geometry = mesh.geometry_map(MeshEntity::new(3, cell)).unwrap();
        let volume = geometry.measure_scale() / 6.;
        let inverse = geometry.inverse_jacobian().unwrap();
        let grad_lambda: [[f64; 3]; 4] = std::array::from_fn(|i| {
            std::array::from_fn(|a| {
                if i == 0 {
                    -(0..3).map(|j| inverse[3 * j + a]).sum::<f64>()
                } else {
                    inverse[3 * (i - 1) + a]
                }
            })
        });
        let gradient =
            |field: eqiora_core::Id<eqiora_core::entity::kinds::Field>| -> [[f64; 3]; 3] {
                std::array::from_fn(|component| {
                    std::array::from_fn(|axis| {
                        vertices
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| {
                                values[&field.erase()][&MeshEntity::new(0, v)][component]
                                    * grad_lambda[i][axis]
                            })
                            .sum()
                    })
                })
            };
        let mut speed_squared = 0.;
        for i in 0..4 {
            for j in 0..4 {
                let mut powers = [0; 4];
                powers[i] += 1;
                powers[j] += 1;
                speed_squared += dot(velocity[i], velocity[j]) * average(powers);
            }
        }
        if fluid {
            let bubble = &values[&field.erase()][&MeshEntity::new(3, cell)];
            for (i, nodal) in velocity.iter().enumerate() {
                let mut powers = [1; 4];
                powers[i] += 1;
                speed_squared += 2. * 256. * dot(nodal, bubble) * average(powers);
            }
            speed_squared += 256_f64.powi(2) * dot(bubble, bubble) * average([2; 4]);
            let mut bubble_gradient = [[0.; 3]; 3];
            for i in 0..4 {
                for j in 0..4 {
                    let mut powers = [2; 4];
                    powers[i] -= 1;
                    powers[j] -= 1;
                    let weight = 256_f64.powi(2) * average(powers);
                    for a in 0..3 {
                        for b in 0..3 {
                            bubble_gradient[a][b] += weight * grad_lambda[i][a] * grad_lambda[j][b];
                        }
                    }
                }
            }
            // Integral grad(b)=0 since b vanishes on all faces: no P1/bubble cross term.
            let trace = (0..3).map(|a| bubble_gradient[a][a]).sum::<f64>();
            let directional = (0..3)
                .flat_map(|a| (0..3).map(move |b| bubble[a] * bubble_gradient[a][b] * bubble[b]))
                .sum::<f64>();
            viscous += 0.5
                * volume
                * (2. * sym_squared(&gradient(field)) + dot(bubble, bubble) * trace + directional);
        } else {
            let g = gradient(fields.displacement);
            let divergence = (0..3).map(|i| g[i][i]).sum::<f64>();
            elastic += volume * (4. * sym_squared(&g) + 0.5 * 2. * divergence.powi(2));
        }
        kinetic += 0.5 * (if fluid { 2. } else { 3. }) * volume * speed_squared;
    }
    [kinetic, elastic, viscous]
}

pub(super) fn balance(
    mesh: &SimplicialMesh,
    fields: Fields,
    old: &Snapshot,
    new: &Snapshot,
    dt: f64,
) -> f64 {
    let before = energy(mesh, fields, old);
    let after = energy(mesh, fields, new);
    let increment = energy(mesh, fields, &difference(new, old));
    // Zero exterior velocity and zero body force leave only BE dissipation.
    after[0] + after[1] - before[0] - before[1] + increment[0] + increment[1] + dt * after[2]
}
