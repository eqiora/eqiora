"""Input-declared mesh controls; no solution or observed cell count selects a level."""
from dataclasses import asdict, dataclass
from types import MappingProxyType
from topology import ComparisonError


@dataclass(frozen=True)
class MeshLevel:
    maximum_target_size: float
    maximum_boundary_error: float
    maximum_boundary_facets: int
    minimum_mean_ratio: float = 1e-5

    def controls(self):
        return asdict(self)


LEVELS = MappingProxyType({
    'coarse': MeshLevel(.05, 4e-4, 25),
    'standard': MeshLevel(.025, 1e-4, 50),
    'fine': MeshLevel(.0125, 2.5e-5, 100),
})


def mesh_level(name):
    try:
        return LEVELS[name]
    except KeyError as error:
        raise ComparisonError(f'undeclared cylinder mesh level {name!r}') from error
