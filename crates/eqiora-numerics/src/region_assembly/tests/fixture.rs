use std::num::NonZeroU16;

use eqiora_assembly::{AssemblyTarget, TargetAssemblyMap};
use eqiora_compiler::compile;
use eqiora_core::{DimExponents, DynQuantity};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::ReferenceCell;
use eqiora_realization::Space;
use eqiora_schema::kernel::KernelNode;
use eqiora_sem::KernelProgram;

use crate::constrained_dofs::ConstrainedDofLayout;
use crate::form_compiler::region::{CompiledRegionForm, RegionFieldBinding, RegionTimeBinding};

use super::*;

pub(super) struct Fixture {
    pub packet_set: AssemblyPacketSetIdentityV1,
    pub plan: AssemblyPlan,
    pub forms: Vec<(BoundRegionForm<f64>, QuadratureRule)>,
    pub domains: Vec<RawId>,
    pub cells: Vec<RegionAssemblyCell<f64>>,
    pub boundary_packets: Vec<AssemblyPacket<f64>>,
    pub matrix: Vec<f64>,
    pub rhs: Vec<f64>,
    pub fixed: Vec<Option<f64>>,
}

impl Fixture {
    pub fn new(regions: usize, reversed: bool) -> Self {
        let (program, field_indices) = program(regions, reversed);
        let n = 3 * regions * (regions + 1) / 2;
        let mut fixed = vec![None; n];
        fixed[0] = Some(1.25);
        fixed[n - 1] = Some(-0.5);
        let constraints = ConstrainedDofLayout::new(fixed.clone()).unwrap();
        let plan = AssemblyPlan::new(vec![
            AssemblyTarget::new(n - 2).unwrap(),
            AssemblyTarget::new(n).unwrap(),
        ])
        .unwrap();
        let reference = ReferenceCell::hypercube(1).unwrap();
        let quadrature = QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap();
        let mut selected = BTreeMap::new();
        for node in program.nodes() {
            if let KernelNode::Domain(domain) = node {
                let region = program.resolved_cartesian_bounds(domain.id()).unwrap()[0]
                    .lower()
                    .value() as usize;
                selected.insert(region, domain.id().erase());
            }
        }
        let mut forms = Vec::new();
        let mut domains = Vec::new();
        let mut cells = Vec::new();
        for (region, domain) in selected {
            let form = CompiledRegionForm::<f64>::derive(&program, domain, 1).unwrap();
            let bindings = form
                .fields()
                .map(|(field, value_type)| RegionFieldBinding {
                    field,
                    space: Space::continuous_lagrange(NonZeroU16::new(1).unwrap()),
                    scale: DynQuantity::new(1.0, value_type.dimension()),
                })
                .collect::<Vec<_>>();
            let multipliers = form
                .rows()
                .map(|(relation, _, _)| {
                    (relation, DynQuantity::new(1.0, dim([0, -1, 1, 0, 0, 0, 0])))
                })
                .collect();
            let time = RegionTimeBinding {
                step: DynQuantity::new(0.5, dim([0, 0, 1, 0, 0, 0, 0])),
                states: Vec::new(),
            };
            let bound = form
                .bind(reference, &bindings, &multipliers, Some(&time))
                .unwrap();
            let offset = 3 * region * (region + 1) / 2;
            for half in 0..2 {
                let mut globals = Vec::new();
                let mut previous = BTreeMap::new();
                for layout in bound.fields() {
                    // Exact source identities select explicitly provided physical history.
                    // History values do not become fresh Model initialization equations.
                    let authored = field_indices[&layout.field];
                    let initial = (authored + 1) as f64;
                    globals.extend([
                        offset + 3 * authored + half,
                        offset + 3 * authored + half + 1,
                    ]);
                    previous.insert(layout.field, vec![initial; 2]);
                }
                cells.push(RegionAssemblyCell {
                    orientation: vec![1; globals.len()],
                    index: 2 * region + half,
                    geometry: AffineGeometryMap::new(
                        reference,
                        1,
                        vec![region as f64 + 0.25 + 0.5 * half as f64],
                        vec![0.25],
                    )
                    .unwrap(),
                    mappings: vec![
                        TargetAssemblyMap::new(
                            plan.target_id(1).unwrap(),
                            constraints.full_map(&globals).unwrap(),
                        ),
                        TargetAssemblyMap::new(
                            plan.target_id(0).unwrap(),
                            constraints.reduced_map(&globals).unwrap(),
                        ),
                    ],
                    previous,
                });
                domains.push(domain);
            }
            forms.push((bound, quadrature.clone()));
        }
        // Neither form nor supplied-cell container order determines packet order.
        if reversed {
            forms.reverse();
            cells.reverse();
        }
        let (matrix, rhs) = expected(regions, n);
        Self {
            packet_set: AssemblyPacketSetIdentityV1::from_sha256([91; 32]),
            plan,
            forms,
            domains,
            cells,
            boundary_packets: Vec::new(),
            matrix,
            rhs,
            fixed,
        }
    }

    pub fn prepare(&self) -> Result<PreparedRegionAssembly<f64>, Diagnostic> {
        PreparedRegionAssembly::new(
            self.packet_set,
            &self.plan,
            self.forms.clone(),
            &self.domains,
            self.cells.clone(),
            self.boundary_packets.clone(),
        )
    }
}

fn dim(exponents: [i32; 7]) -> DimExponents {
    DimExponents::from_integers(exponents).unwrap()
}

fn program(regions: usize, reversed: bool) -> (KernelProgram, BTreeMap<RawId, usize>) {
    let mut source = String::from("model Regions() {\n");
    let order = if reversed {
        (0..regions).rev().collect::<Vec<_>>()
    } else {
        (0..regions).collect()
    };
    for region in order {
        source += &format!(
            "domain body{region} = box({region}, {}); parameter diffusion{region}: m ^ 2 / s = {}; parameter rate{region}: 1 / s = {}; parameter load{region}: 1 / s = {};\n",
            region + 1,
            region + 2,
            region + 3,
            region + 1
        );
        let fields = if reversed {
            (0..=region).rev().collect::<Vec<_>>()
        } else {
            (0..=region).collect()
        };
        for field in fields {
            source += &format!("state value{region}_{field}: 1 on body{region};\n");
        }
        for field in 0..=region {
            source += &format!(
                "relation row{region}_{field} on body{region} {{ derivative(value{region}_{field}) - div(diffusion{region} * grad(value{region}_{field})) + rate{region} * value{region}_{field}"
            );
            for trial in 0..=region {
                if trial != field {
                    source += &format!(
                        " + {} * rate{region} * value{region}_{trial}",
                        (field + 1) * (trial + 2)
                    );
                }
            }
            source += &format!(" = load{region}; }}\n");
        }
    }
    source += "}";
    if reversed {
        source = source
            .replace("value", "renamed")
            .replace("row", "equation");
    }
    let (transaction, model, symbols) = compile("regions.eqi", &source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let mut indices = BTreeMap::new();
    let prefix = if reversed { "renamed" } else { "value" };
    for region in 0..regions {
        for field in 0..=region {
            indices.insert(
                symbols.get(&format!("{prefix}{region}_{field}")).unwrap(),
                field,
            );
        }
    }
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        indices,
    )
}

fn expected(regions: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut matrix = vec![0.0; n * n];
    let mut rhs = vec![0.0; n];
    let h = 0.5;
    for region in 0..regions {
        let offset = 3 * region * (region + 1) / 2;
        for half in 0..2 {
            for field in 0..=region {
                for i in 0..2 {
                    let row = offset + 3 * field + half + i;
                    rhs[row] += (2 * (field + 1) + region + 1) as f64 * h / 2.0;
                    for trial in 0..=region {
                        for j in 0..2 {
                            let column = offset + 3 * trial + half + j;
                            let reaction = if trial == field {
                                region + 5
                            } else {
                                (field + 1) * (trial + 2) * (region + 3)
                            };
                            let diffusion = if trial == field {
                                (region + 2) as f64 / h * if i == j { 1.0 } else { -1.0 }
                            } else {
                                0.0
                            };
                            matrix[row * n + column] += diffusion
                                + reaction as f64 * h / 6.0 * if i == j { 2.0 } else { 1.0 };
                        }
                    }
                }
            }
        }
    }
    (matrix, rhs)
}
