//! Ordinary equations for consecutive Regions, sharing the fixture's Port types.
#[derive(Clone, Copy)]
pub(super) enum Material {
    Fluid { density: f64, viscosity: f64 },
    Solid { density: f64, mu: f64, lambda: f64 },
}

pub(super) fn network(declarations: &str, regions: &[(&str, Material)]) -> String {
    let declarations = declarations.split_once("model Main()").unwrap().0;
    let mut source =
        format!("{declarations}model Main() {{\nparameter zero_pressure: kg / (m * s ^ 2) = 0;\n");
    for (index, &(name, material)) in regions.iter().enumerate() {
        source.push_str(&format!(
            "domain {name} = box({index}, {}, 0, 1);\n",
            index + 1
        ));
        for (axis, coordinate) in [(0, "x"), (1, "y")] {
            for side in ["lower", "upper"] {
                source.push_str(&format!("domain {name}_{coordinate}_{side} = boundary({name}, axis = {axis}, side = {side});\n"));
            }
        }
        source.push_str(&format!(
            r#"
variable {name}_load_potential: kg / (m * s ^ 2) on {name};
relation {name}_load_definition on {name} {{ {name}_load_potential - zero_pressure = 0; }}
"#
        ));
        match material {
            Material::Fluid { density, viscosity } => source.push_str(&format!(
                r#"
state {name}_velocity: vector<m / s, 2> on {name} in smooth;
variable {name}_pressure: kg / (m * s ^ 2) on {name} in h1;
parameter {name}_density: kg / m ^ 3 = {density};
parameter {name}_viscosity: kg / (m * s) = {viscosity};
relation {name}_momentum on {name} {{
  {name}_density * derivative({name}_velocity)
    - div(2 * {name}_viscosity * symmetric_part(grad({name}_velocity))
      - isotropic_lift({name}_pressure)) - grad({name}_load_potential) = 0;
}}
relation {name}_incompressibility on {name} {{ div({name}_velocity) = 0; }}
instance {name}_boundary: NewtonianMechanicalInterface2d(
  body = {name},
  exterior = boundaries({name}_x_lower, {name}_x_upper, {name}_y_lower, {name}_y_upper),
  velocity = {name}_velocity, pressure = {name}_pressure, dynamic_viscosity = {name}_viscosity
);
"#
            )),
            Material::Solid {
                density,
                mu,
                lambda,
            } => source.push_str(&format!(
                r#"
state {name}_displacement: vector<m, 2> on {name} in smooth;
state {name}_velocity: vector<m / s, 2> on {name} in h1;
parameter {name}_density: kg / m ^ 3 = {density};
parameter {name}_mu: kg / (m * s ^ 2) = {mu};
parameter {name}_lambda: kg / (m * s ^ 2) = {lambda};
relation {name}_kinematics on {name} {{ derivative({name}_displacement) - {name}_velocity = 0; }}
relation {name}_momentum on {name} {{
  {name}_density * derivative({name}_velocity)
    - div(2 * {name}_mu * symmetric_part(grad({name}_displacement))
      + {name}_lambda * isotropic_lift(div({name}_displacement)))
    - grad({name}_load_potential) = 0;
}}
instance {name}_boundary: ElastodynamicMechanicalInterface2d(
  body = {name},
  exterior = boundaries({name}_x_lower, {name}_x_upper, {name}_y_lower, {name}_y_upper),
  displacement = {name}_displacement, velocity = {name}_velocity,
  mu = {name}_mu, lambda = {name}_lambda
);
"#
            )),
        }
        let mut walls = vec!["y_lower", "y_upper"];
        if index == 0 {
            walls.push("x_lower");
        }
        if index + 1 == regions.len() {
            walls.push("x_upper");
        }
        for wall in walls {
            source.push_str(&format!(
                r#"
instance {name}_{wall}_zero: ZeroVelocity2d(body = {name}, face = {name}_{wall});
connect {name}_boundary.mechanical[boundary = {name}_{wall}], {name}_{wall}_zero.mechanical;
"#
            ));
        }
    }
    for pair in regions.windows(2) {
        let (left, right) = (pair[0].0, pair[1].0);
        source.push_str(&format!("connect {left}_boundary.mechanical[boundary = {left}_x_upper], {right}_boundary.mechanical[boundary = {right}_x_lower];\n"));
    }
    source.push_str("}\n");
    source
}
