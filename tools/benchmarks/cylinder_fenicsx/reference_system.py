"""UFL assembly and explicit common hierarchical basis; solves require authorisation."""
from dataclasses import dataclass
import basix
from dolfinx import fem
from dolfinx.fem import petsc
import numpy as np
from petsc4py import PETSc
from accuracy import H, MU, U, P, check_rule, duffy_rule
from fenicsx_adapter import weak_forms, cylinder_virtual_work, export_fields
from shared_measurement import ComparisonError


@dataclass
class System:
    matrix: object
    rhs: object
    transformation: object
    free: np.ndarray
    scales: np.ndarray
    prescribed: np.ndarray
    raw_nodal_matrix: object
    prepared: object


def quadrature(name):
    if name == 'duffy-3':
        points, weights = duffy_rule(3)
        degree = 4
    elif name == 'basix-degree-6':
        points, weights = basix.make_quadrature(basix.CellType.triangle, 6,
                                               rule=basix.QuadratureType.gauss_jacobi)
        degree = 6
    else:
        raise ComparisonError('undeclared reference assembly rule')
    check_rule(points, weights, degree)
    return {'quadrature_rule': 'custom', 'quadrature_points': points, 'quadrature_weights': weights}


def canonical_fields(fields):
    return np.concatenate((np.asarray(fields['velocity_vertex']).reshape(-1),
                           np.asarray(fields['velocity_cell_bubble']).reshape(-1),
                           np.asarray(fields['pressure_vertex']).reshape(-1)))


def basis_transformation(prepared):
    nv, nc = len(prepared.coordinates), len(prepared.cells)
    count = 3*nv+2*nc
    transformation = PETSc.Mat().createAIJ(size=(count, count), nnz=4, comm=prepared.domain.comm)
    vm, pm = prepared.velocity_to_mixed, prepared.pressure_to_mixed
    for vertex in range(nv):
        for component in range(2):
            row = int(vm[2*prepared.vertex_velocity_dofs[vertex]+component])
            transformation.setValue(row, 2*vertex+component, 1.)
        row = int(pm[prepared.vertex_pressure_dofs[vertex]])
        transformation.setValue(row, 2*(nv+nc)+vertex, 1.)
    for cell, vertices in enumerate(prepared.cells):
        for component in range(2):
            row = int(vm[2*prepared.cell_centroid_dofs[cell]+component])
            transformation.setValue(row, 2*nv+2*cell+component, 1.)
            for vertex in vertices:
                transformation.setValue(row, 2*int(vertex)+component, 1/3)
    transformation.assemble()
    return transformation


def build_system(prepared, data, *, rule, jit_options):
    stiffness, _ = weak_forms(prepared, quadrature_metadata=quadrature(rule))
    form = fem.form(stiffness, jit_options=jit_options)
    raw_nodal_matrix = petsc.assemble_matrix(form); raw_nodal_matrix.assemble()
    transformation = basis_transformation(prepared)
    full = raw_nodal_matrix.PtAP(transformation)
    nv, nc = len(prepared.coordinates), len(prepared.cells)
    count = 3*nv+2*nc
    essential_vertices = np.unique(np.concatenate([np.asarray(data['boundaries'][name]).reshape(-1)
                                                   for name in ('inlet', 'walls', 'cylinder')]))
    constrained = (2*essential_vertices[:, None]+np.arange(2)).reshape(-1).astype(PETSc.IntType)
    free = np.setdiff1d(np.arange(count, dtype=PETSc.IntType), constrained)
    if len(free) != count-2*len(essential_vertices):
        raise ComparisonError('free MINI inventory differs from full count minus distinct essential vertices')
    prescribed = np.zeros(count)
    inlet_vertices = np.unique(np.asarray(data['boundaries']['inlet']).reshape(-1))
    y = prepared.coordinates[inlet_vertices, 1]
    prescribed[2*inlet_vertices] = 4*U*y*(H-y)/(H*H)
    # Wall/cylinder values take precedence at shared inlet corners.
    zero_vertices = np.unique(np.concatenate([np.asarray(data['boundaries'][name]).reshape(-1)
                                              for name in ('walls', 'cylinder')]))
    prescribed[2*zero_vertices] = 0.
    original = full.createVecRight(); original.array[:] = prescribed
    lifted = full.createVecLeft(); full.mult(original, lifted); lifted.scale(-1.)
    free_is = PETSc.IS().createGeneral(free, comm=prepared.domain.comm)
    matrix = full.createSubMatrix(free_is, free_is)
    rhs = matrix.createVecRight()
    sub = lifted.getSubVector(free_is); rhs.array[:] = sub.array; lifted.restoreSubVector(free_is, sub)
    scales = np.concatenate((np.full(2*(nv+nc), U), np.full(nv, P)))[free]
    for item in (full, original, lifted, free_is):
        item.destroy()
    return System(matrix, rhs, transformation, free, scales, prescribed, raw_nodal_matrix, prepared)


def scaled_operator(system):
    matrix, rhs = system.matrix.copy(), system.rhs.copy()
    scale = rhs.duplicate(); scale.array[:] = system.scales
    matrix.diagonalScale(scale, scale); matrix.scale(1/(MU*U*U))
    rhs.pointwiseMult(rhs, scale); rhs.scale(1/(MU*U*U)); scale.destroy()
    return matrix, rhs


def solve(system, implementation):
    if implementation not in ('mumps', 'superlu') or not PETSc.Sys.hasExternalPackage(implementation):
        raise ComparisonError('declared distinct external factorization unavailable')
    solution = system.matrix.createVecRight(); solution.set(0.)
    ksp = PETSc.KSP().create(system.prepared.domain.comm)
    ksp.setOperators(system.matrix); ksp.setType('preonly'); ksp.getPC().setType('lu')
    ksp.getPC().setFactorSolverType(implementation)
    ksp.setInitialGuessNonzero(False)
    ksp.solve(system.rhs, solution)
    actual = ksp.getPC().getFactorSolverType()
    reason = int(ksp.getConvergedReason())
    if actual != implementation or reason <= 0 or not np.isfinite(solution.array).all():
        raise ComparisonError('external factorization failed or differs from declared implementation')
    canonical = system.prescribed.copy(); canonical[system.free] = solution.array
    nodal = system.transformation.createVecLeft()
    complete = system.transformation.createVecRight(); complete.array[:] = canonical
    system.transformation.mult(complete, nodal)
    function = fem.Function(system.prepared.mixed_space); function.x.array[:] = nodal.array
    function.x.scatter_forward()
    fields = export_fields(system.prepared, function)
    summary = {'implementation': actual, 'ksp_type': ksp.getType(), 'pc_type': ksp.getPC().getType(),
               'convergence_reason': reason, 'completed_iterations': ksp.getIterationNumber(),
               'precision': 'float64', 'initial_guess_nonzero': ksp.getInitialGuessNonzero(),
               'petsc_tolerances_diagnostic_only': list(ksp.getTolerances()),
               'direct_solver_nominal_tolerances_are_not_forward_error_bounds': True}
    for item in (solution, ksp, nodal, complete):
        item.destroy()
    return function, fields, canonical, summary


def scaled_residual(matrix, rhs, canonical, system):
    unknown = matrix.createVecRight(); unknown.array[:] = canonical[system.free]/system.scales
    residual = rhs.duplicate(); matrix.mult(unknown, residual); residual.axpy(-1., rhs)
    result = float(residual.norm(PETSc.NormType.NORM_INFINITY))
    unknown.destroy(); residual.destroy()
    return result


def observations(system, function, fields, canonical, *, jit_options):
    velocity, pressure = function.sub(0), function.sub(1)
    metadata = quadrature('basix-degree-6')
    expressions = cylinder_virtual_work(system.prepared, velocity, pressure, quadrature_metadata=metadata)
    reaction = np.array([fem.assemble_scalar(fem.form(expr, jit_options=jit_options)) for expr in expressions])
    # Supplemental uneliminated full-matrix check in the original nodal basis.
    complete = system.transformation.createVecRight(); complete.array[:] = canonical
    nodal = system.transformation.createVecLeft(); system.transformation.mult(complete, nodal)
    imbalance = nodal.duplicate(); system.raw_nodal_matrix.mult(nodal, imbalance)
    indices = system.prepared.velocity_to_mixed.reshape(-1, 2)[system.prepared.vertex_velocity_dofs[system.prepared.cylinder_vertices]]
    # Nodal centroid basis has nonzero vertex gradients; the P1 virtual lift
    # includes centroid values. Evaluate its dot product, not only vertex rows.
    from fenicsx_adapter import cylinder_indicator
    matrix_reaction = []
    for component in range(2):
        lift = cylinder_indicator(system.prepared, component)
        vector = imbalance.duplicate(); vector.set(0.)
        vector.array[system.prepared.velocity_to_mixed] = lift.x.array
        matrix_reaction.append(vector.dot(imbalance)); vector.destroy()
    for item in (complete, nodal, imbalance):
        item.destroy()
    return {'fields': fields, 'force_on_domain': reaction.tolist(), 'force_on_selection': (-reaction).tolist(),
            'matrix_virtual_work_on_domain': matrix_reaction}
