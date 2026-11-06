"""Read the shared runner's original float64-hex field artifact without native tools."""
import json
import numpy as np
from topology import ComparisonError
from shared_measurement import digest


def decode(values):
    array = np.asarray(values)
    return np.fromiter((float.fromhex(item) for item in array.flat), dtype=np.float64).reshape(array.shape)


def read_candidate(path, expected_sha256):
    if digest(path) != expected_sha256:
        raise ComparisonError('original candidate cold field digest mismatch')
    candidate = json.loads(path.read_text())
    coordinates, primary = decode(candidate['coordinates_float64_hex']), decode(candidate['values_float64_hex'])
    associated = {name: {'association': field['association'], 'values': decode(field['values_float64_hex'])}
                  for name, field in candidate['associated_fields'].items()}
    return coordinates, primary, associated
