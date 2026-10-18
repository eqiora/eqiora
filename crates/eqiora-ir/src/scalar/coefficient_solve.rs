//! Nilpotent coefficient solve B Q = A for scalar and square matrix values.
//! Coefficients are row-major blocks; the block index is a direction bitmask.
use super::*;

pub(super) fn admit(length: usize, extent: usize) -> Result<usize, Diagnostic> {
    let width = extent
        .checked_mul(extent)
        .filter(|&n| n > 0)
        .ok_or_else(|| ir_builder_error("invalid coefficient solve extent"))?;
    let count = length / width;
    let work = count
        .is_power_of_two()
        .then(|| 3usize.checked_pow(count.trailing_zeros()))
        .flatten()
        // A matrix solve also factors B0 and estimates its conditioning with
        // the existing inverse owner. Scalar division needs neither operation.
        .and_then(|work| work.checked_add(if extent == 1 { 0 } else { 2 }))
        .and_then(|work| extent.checked_pow(3).and_then(|n| work.checked_mul(n)));
    if !length.is_multiple_of(width) || work.is_none_or(|work| work > MAX_COMPONENT_PRODUCTS) {
        return Err(ir_builder_error(
            "coefficient solve work exceeds scalar product budget",
        ));
    }
    Ok(count)
}

pub(super) fn coefficients(
    a: &[Option<Scaled>],
    b: &[Option<Scaled>],
    extent: usize,
) -> Result<Vec<Scaled>, Diagnostic> {
    let count = admit(a.len(), extent)?;
    require_length(b, a.len(), "coefficient solve denominator")?;
    let width = extent * extent;
    if b[..width].iter().all(Option::is_none) {
        return Err(ir_builder_error(
            "coefficient solve requires a base denominator",
        ));
    }
    if a.iter().chain(b).flatten().any(|value| !value.is_finite()) {
        return Ok(vec![Scaled::new(f64::NAN); a.len()]);
    }
    let base = b[..width]
        .iter()
        .map(|v| v.unwrap_or(Scaled::ZERO))
        .collect::<Vec<_>>();
    if extent == 1 && base[0].is_zero() {
        return Ok(vec![Scaled::new(f64::NAN); a.len()]);
    }
    let mut factor = if extent == 1 {
        None
    } else {
        Some(map_evaluation::MapEvaluation::new(&base, extent)?)
    };
    let mut q = Vec::with_capacity(a.len());
    for mask in 0..count {
        let mut rhs = a[mask * width..(mask + 1) * width]
            .iter()
            .map(|v| v.unwrap_or(Scaled::ZERO))
            .collect::<Vec<_>>();
        let mut subset = mask;
        while subset != 0 {
            for row in 0..extent {
                for column in 0..extent {
                    for k in 0..extent {
                        rhs[row * extent + column] = rhs[row * extent + column]
                            - b[subset * width + row * extent + k].unwrap_or(Scaled::ZERO)
                                * q[(mask ^ subset) * width + k * extent + column];
                    }
                }
            }
            subset = (subset - 1) & mask;
        }
        if let Some(factor) = &mut factor {
            q.extend(factor.solve(&rhs)?);
        } else {
            q.push(rhs[0] / base[0]);
        }
    }
    Ok(q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_coefficients_match_closed_form_noncommuting_inverse_through_order_four() {
        // A=s[[1,x],[y,1]], A^-1=[[1,-x],[-y,1]]/(s*(1-xy)).
        // Two independent x directions and two y directions test repeated-axis
        // factorials as well as noncommuting products. Swapping A's rows tests
        // pivoting and swaps the corresponding columns of its inverse.
        for exponent in [-700, -400, 400, 700] {
            let scale = 2f64.powi(exponent);
            for swap_rows in [false, true] {
                let mut a = vec![None; 16 * 4];
                a[0] = Some(Scaled::new(1.));
                a[3] = Some(Scaled::new(1.));
                let mut b = vec![None; a.len()];
                let entry = |row, column| (row ^ usize::from(swap_rows)) * 2 + column;
                b[entry(0, 0)] = Some(Scaled::new(scale));
                b[entry(1, 1)] = Some(Scaled::new(scale));
                for bit in [1, 2] {
                    b[bit * 4 + entry(0, 1)] = Some(Scaled::new(scale));
                }
                for bit in [4, 8] {
                    b[bit * 4 + entry(1, 0)] = Some(Scaled::new(scale));
                }
                let actual = coefficients(&a, &b, 2).unwrap();
                for mask in 0usize..16 {
                    let x = (mask & 3).count_ones();
                    let y = (mask & 12).count_ones();
                    let factorial = |n| if n == 2 { 2. } else { 1. };
                    for row in 0..2 {
                        for column in 0..2 {
                            let original_column = column ^ usize::from(swap_rows);
                            let sign = if row == original_column && x == y {
                                1.
                            } else if row == 0 && original_column == 1 && x == y + 1
                                || row == 1 && original_column == 0 && y == x + 1
                            {
                                -1.
                            } else {
                                0.
                            };
                            let expected = sign * factorial(x) * factorial(y);
                            assert_eq!(
                                actual[mask * 4 + row * 2 + column].value() * scale,
                                expected,
                                "exponent={exponent}, swap={swap_rows}, mask={mask}, row={row}, column={column}"
                            );
                        }
                    }
                }
            }
        }
    }
}
