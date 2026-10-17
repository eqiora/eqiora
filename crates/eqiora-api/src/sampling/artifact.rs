//! Frozen samples are data; reopening never calls a generator draw.
use base64::{Engine, engine::general_purpose::STANDARD};
use eqiora_artifact::JsonLimits;
use serde::{Deserialize, Serialize};

use super::*;

const SCHEMA: &str = "eqiora.sampled-parameter-point/v1";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSample {
    schema: String,
    program: String,
    generator: String,
    master: [u8; 32],
    identity: SamplingIdentity,
    stream: [u8; 32],
    fixed_parameters: Vec<String>,
    fixed_values: Vec<f64>,
    bounds: Vec<[f64; 2]>,
    point: Vec<f64>,
}

impl SampledParameterPoint {
    /// Encode frozen values and complete sampling lineage against the static Program.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        let bytes = eqiora_artifact::JsonLimits::EXECUTION
            .encode(&WireSample {
                schema: SCHEMA.to_owned(),
                program: STANDARD.encode(self.program.to_bytes()?),
                generator: self.generator.to_owned(),
                master: self.master,
                identity: self.identity.clone(),
                stream: self.stream,
                fixed_parameters: self
                    .fixed
                    .signatures()
                    .iter()
                    .map(|s| s.entity().to_string())
                    .collect(),
                fixed_values: self.fixed.coordinates().to_vec(),
                bounds: self.bounds.clone(),
                point: self.point.coordinates().to_vec(),
            })
            .map_err(|_| invalid("cannot encode frozen sample"))?;
        JsonLimits::default().check(&bytes)?;
        Ok(bytes)
    }

    /// Reopen frozen input, request bounds and stream addressing without drawing.
    ///
    /// Checks exact signatures, fixed values, intervals and stream lineage. This
    /// retains recorded draw provenance, not proof of the producer's execution.
    pub fn from_bytes(
        bytes: &[u8],
        program: &DifferentiableProgramIdentity,
    ) -> Result<Self, Diagnostic> {
        JsonLimits::default().check(bytes)?;
        let wire: WireSample =
            serde_json::from_slice(bytes).map_err(|_| invalid("invalid frozen sample JSON"))?;
        if wire.schema != SCHEMA
            || wire.program != STANDARD.encode(program.to_bytes()?)
            || wire.generator != ParameterSampler::GENERATOR
            || wire.stream
                != wire
                    .identity
                    .stream_digest(&wire.master, ParameterSampler::GENERATOR)
        {
            return Err(invalid(
                "frozen sample differs from its Program or stream lineage",
            ));
        }
        let fixed_signatures = wire
            .fixed_parameters
            .iter()
            .map(|id| {
                program
                    .inputs()
                    .iter()
                    .find(|s| s.entity().to_string() == *id)
                    .cloned()
                    .ok_or_else(|| {
                        invalid("frozen sample fixed Parameter is absent from its Program")
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let fixed =
            DifferentiableValues::from_coordinates(fixed_signatures.into(), wire.fixed_values)?;
        // One admission owner for fresh requests and recorded request metadata.
        let request = ParameterSampler::admit(program.clone(), wire.master, &fixed, &wire.bounds)?;
        let point =
            DifferentiableValues::from_coordinates(program.inputs().to_vec().into(), wire.point)?;
        if request
            .fixed_coordinates
            .iter()
            .zip(fixed.coordinates())
            .any(|(&i, value)| point.coordinates()[i].to_bits() != value.to_bits())
            || request
                .sampled_coordinates
                .iter()
                .zip(&wire.bounds)
                .any(|(&i, &[lower, upper])| {
                    let value = point.coordinates()[i];
                    value < lower || value >= upper
                })
        {
            return Err(invalid(
                "frozen sample values differ from its fixed inputs or intervals",
            ));
        }
        let sample = Self {
            fixed: request.fixed,
            generator: ParameterSampler::GENERATOR,
            bounds: request.bounds,
            master: wire.master,
            identity: wire.identity,
            stream: wire.stream,
            program: program.clone(),
            point,
        };
        if sample.to_bytes()? != bytes {
            return Err(invalid("frozen sample bytes are not canonical"));
        }
        Ok(sample)
    }
}
