"""Evaluate the frozen external CSR operator without PETSc in the Eqiora process."""
import numpy as np
from shared_measurement import ComparisonError


def csr_product(indptr, indices, values, vector):
    indptr, indices, values, vector = map(np.asarray, (indptr, indices, values, vector))
    if indptr.ndim != 1:
        raise ComparisonError('external CSR indptr must be a rank-one array')
    count = len(indptr)-1
    if (indptr.ndim != 1 or not np.issubdtype(indptr.dtype, np.integer)
            or indices.ndim != 1 or values.ndim != 1 or values.dtype != np.dtype('float64')
            or not np.issubdtype(indices.dtype, np.integer) or vector.ndim != 1 or len(vector) != count
            or not np.isfinite(values).all() or not np.isfinite(vector).all()
            or count <= 0 or indptr[0] != 0 or np.any(indptr[1:] < indptr[:-1])
            or np.any(indptr < 0) or np.any(indptr > len(indices))
            or indptr[-1] != len(indices) or indices.shape != values.shape
            or np.any(indices < 0) or np.any(indices >= len(vector))):
        raise ComparisonError('invalid frozen external CSR operator')
    rows = np.repeat(np.arange(count), np.diff(indptr))
    return np.bincount(rows, weights=values*vector[indices], minlength=count)


def residual_inf(operator, canonical):
    unknown = np.asarray(canonical)[operator['free']]/operator['scales']
    action = csr_product(operator['indptr'], operator['indices'], operator['values'], unknown)
    residual = action-operator['rhs']
    if not np.isfinite(residual).all():
        raise ComparisonError('nonfinite frozen external-operator residual')
    return float(np.max(np.abs(residual)))


def validate_operator(operator, *, canonical_size, expected_free, expected_scales):
    if set(operator) != {'indptr', 'indices', 'values', 'rhs', 'free', 'scales'}:
        raise ComparisonError('frozen external operator arrays differ from the declared inventory')
    free, scales = np.asarray(operator['free']), np.asarray(operator['scales'])
    if free.ndim != 1:
        raise ComparisonError('external free indices must be a rank-one array')
    count = len(free)
    if (free.ndim != 1 or not np.issubdtype(free.dtype, np.integer)
            or np.any(free < 0) or np.any(free >= canonical_size) or len(set(map(int, free))) != count
            or not np.array_equal(free, expected_free)):
        raise ComparisonError('frozen free indices do not bijectively identify the declared common basis')
    if scales.shape != (count,) or not np.isfinite(scales).all() or np.any(scales <= 0) or not np.array_equal(scales, expected_scales):
        raise ComparisonError('frozen variable scaling differs from U/P input scales')
    rhs = np.asarray(operator['rhs'])
    pointer = np.asarray(operator['indptr'])
    if pointer.ndim != 1:
        raise ComparisonError('external CSR indptr must be a rank-one array')
    if rhs.shape != (count,) or not np.isfinite(rhs).all() or len(pointer) != count+1:
        raise ComparisonError('frozen operator/RHS dimensions differ from the common free space')
    # Validation checks every CSR dimension/index/value before numerical use.
    csr_product(operator['indptr'], operator['indices'], operator['values'], np.zeros(count))


def load_operator(path, *, canonical_size, expected_free, expected_scales):
    try:
        with np.load(path, allow_pickle=False) as archive:
            operator = {name: archive[name] for name in archive.files}
    except (ValueError, OSError) as error:
        raise ComparisonError(f'invalid frozen reference archive: {error}') from error
    validate_operator(operator, canonical_size=canonical_size, expected_free=expected_free, expected_scales=expected_scales)
    return operator
