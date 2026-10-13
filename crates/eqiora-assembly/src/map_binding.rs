//! Exact immutable map binding shared by preparation and packet admission.
use crate::{DofId, LocalUnknown};
use eqiora_core::{Diagnostic, Scalar, ScalarDomain};
use num_complex::ComplexFloat;
use num_traits::ToPrimitive;

#[cfg(test)]
thread_local! {
    static ENCODINGS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[cfg(test)]
pub(crate) fn encoding_count() -> usize {
    ENCODINGS.with(std::cell::Cell::get)
}

pub(crate) fn encode<S: Scalar + ComplexFloat>(
    equations: &[Option<DofId>],
    unknowns: &[LocalUnknown<S>],
) -> Result<Vec<u8>, Diagnostic> {
    #[cfg(test)]
    ENCODINGS.with(|count| count.set(count.get() + 1));
    let mut out = Vec::new();
    push_usize(&mut out, equations.len())?;
    for equation in equations {
        match equation {
            Some(dof) => {
                out.push(1);
                push_usize(&mut out, dof.index())?;
            }
            None => out.push(0),
        }
    }
    push_usize(&mut out, unknowns.len())?;
    for unknown in unknowns {
        match unknown {
            LocalUnknown::Free(dof) => {
                out.push(0);
                push_usize(&mut out, dof.index())?;
            }
            LocalUnknown::Fixed(value) => {
                if !value.is_finite() {
                    return Err(invalid("fixed assembly values must be finite"));
                }
                out.push(1);
                // Binary32 components are exactly representable in binary64.
                // The packet scalar tag independently binds their precision.
                let real = value
                    .re()
                    .to_f64()
                    .ok_or_else(|| invalid("fixed real component is not representable"))?;
                out.extend_from_slice(&real.to_bits().to_be_bytes());
                if S::DOMAIN == ScalarDomain::Complex {
                    let imaginary = value
                        .im()
                        .to_f64()
                        .ok_or_else(|| invalid("fixed imaginary component is not representable"))?;
                    out.extend_from_slice(&imaginary.to_bits().to_be_bytes());
                }
            }
        }
    }
    Ok(out)
}

pub(crate) fn push_usize(out: &mut Vec<u8>, value: usize) -> Result<(), Diagnostic> {
    let value =
        u64::try_from(value).map_err(|_| invalid("assembly structure exceeds portable u64"))?;
    out.extend_from_slice(&value.to_be_bytes());
    Ok(())
}

fn invalid(message: &str) -> Diagnostic {
    Diagnostic::error(eqiora_core::diagnostic::codes::ASSEMBLY_FAILED, message)
}
