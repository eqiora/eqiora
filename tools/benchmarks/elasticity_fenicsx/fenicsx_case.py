"""Ordinary blocked-vector Q1 FEniCSx elasticity; shared runner owns timing."""
import time
IMPORT_START = time.perf_counter_ns()
import os
import sys
import sysconfig
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import basix
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem import petsc
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl
from shared_measurement import clock, environment, finish, memory
from shared_accuracy import check_identity_row, q1_vertex_dof_coordinates
from accuracy import ComparisonError, check_accuracy, check_equilibrium
from worker import arguments
IMPORT_SECONDS = (time.perf_counter_ns()-IMPORT_START)/1e9
PROTOCOL_ROOT = Path(__file__).resolve().parent


def prepare(args):
    if dolfinx.__version__ != '0.11.0' or basix.__version__ != '0.11.0':
        raise ComparisonError('requires locked DOLFINx/Basix0.11.0')
    if PETSc.ScalarType != np.float64 or MPI.COMM_WORLD.size != 1:
        raise ComparisonError('requires real binary64 PETSc and one MPI rank')
    env = environment('fenicsx', IMPORT_SECONDS, protocol_root=PROTOCOL_ROOT)
    env.update(mpi_library=MPI.Get_library_version(), petsc_version=PETSc.Sys.getVersionInfo(),
               petsc_scalar_type=str(PETSc.ScalarType), petsc_integer_type=str(PETSc.IntType),
               compiler_environment={key: os.environ.get(key) for key in ('CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS')},
               python_compiler_config={key: sysconfig.get_config_var(key) for key in ('CC', 'CFLAGS', 'LDSHARED')},
               blas=np.show_config(mode='dicts'))
    baseline = memory()
    domain, mesh_time = clock(lambda: mesh.create_unit_square(
        MPI.COMM_SELF, args.cells, args.cells, cell_type=mesh.CellType.quadrilateral))
    points, weights = basix.make_quadrature(basix.CellType.quadrilateral, 3, rule=basix.QuadratureType.gauss_jacobi)
    one_d = (1+np.array((-1., 1.))/np.sqrt(3))/2
    np.testing.assert_array_max_ulp(np.array(sorted(map(tuple, points))),
                                   np.array(sorted((s, t) for s in one_d for t in one_d)), maxulp=8)
    np.testing.assert_array_max_ulp(weights, np.full(4, .25), maxulp=8)
    env['solve_quadrature'] = {'points': points.tolist(), 'weights': weights.tolist(), 'degree': 3, 'rule': 'gauss_jacobi'}
    cache = args.output.parent/'fresh-jit-cache'
    if cache.exists():
        raise ComparisonError('cold FEniCSx JIT directory must not already exist')
    cache.mkdir(parents=True)
    count = (args.cells+1)**2

    def forms():
        space = fem.functionspace(domain, ('Lagrange', 1, (2,)))
        if space.dofmap.index_map.size_global != count or space.dofmap.index_map_bs != 2:
            raise ComparisonError('requires two-component blocked Q1 space on full vertex grid')
        domain.topology.create_connectivity(1, 2)
        facets = mesh.exterior_facet_indices(domain.topology)
        boundary = np.unique(fem.locate_dofs_topological(space, 1, facets))
        if len(boundary) != 4*args.cells:
            raise ComparisonError('incomplete external essential boundary')
        bc = fem.dirichletbc(np.zeros(2, dtype=PETSc.ScalarType), boundary, space)
        trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
        x = ufl.SpatialCoordinate(domain)
        measure = ufl.Measure('dx', domain=domain, metadata={'quadrature_degree': 3, 'quadrature_rule': 'gauss_jacobi'})
        sx, sy = ufl.sin(np.pi*x[0]), ufl.sin(np.pi*x[1])
        potential = sx**2+sy**2-4*sx**2*sy**2
        sigma = 6*ufl.sym(ufl.grad(trial))+2*ufl.div(trial)*ufl.Identity(2)
        jit = {'cache_dir': str(cache), 'cffi_extra_compile_args': ['-O3']}
        stiffness = fem.form(ufl.inner(sigma, ufl.sym(ufl.grad(test)))*measure, jit_options=jit)
        load = fem.form(ufl.inner(ufl.grad(potential), test)*measure, jit_options=jit)
        return space, boundary, bc, stiffness, load, test, measure, jit

    (space, boundary, bc, stiffness, load, test, measure, jit), preparation = clock(forms)
    domain.topology.create_connectivity(0, 2)
    bindings = []
    for vertex in range(domain.topology.index_map(0).size_local):
        entity = np.array([vertex], dtype=np.int32)
        dof = fem.locate_dofs_topological(space, 0, entity)
        geometry_index = mesh.entities_to_geometry(domain, 0, entity, permute=False)
        if geometry_index.shape != (1, 1):
            raise ComparisonError('Q1 vertex does not identify exactly one Geometry vertex')
        bindings.append((vertex, dof.tolist(), domain.geometry.x[geometry_index[0, 0], :2]))
    coordinates = q1_vertex_dof_coordinates(bindings, count)
    env['coordinate_export'] = 'exact Geometry vertices through topological Q1 vertex/block-DOF correspondence'
    boundary_scalar = (2*boundary[:, None]+np.arange(2)).reshape(-1)
    free = np.setdiff1d(np.arange(2*count, dtype=PETSc.IntType), boundary_scalar)

    def assemble(rhs_form):
        matrix = petsc.assemble_matrix(stiffness, bcs=[bc]); matrix.assemble()
        rhs = petsc.assemble_vector(rhs_form)
        petsc.apply_lifting(rhs, [stiffness], bcs=[[bc]])
        rhs.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(rhs, [bc])
        return matrix, rhs

    def solve(matrix, rhs):
        solution = matrix.createVecRight(); solution.set(0.)
        ksp = PETSc.KSP().create(MPI.COMM_SELF)
        ksp.setOperators(matrix); ksp.setType('cg'); ksp.getPC().setType('none')
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED); ksp.setInitialGuessNonzero(False)
        ksp.setTolerances(rtol=1e-12, atol=1e-14, max_it=10000)
        ksp.solve(rhs, solution)
        residual = rhs.duplicate(); matrix.mult(solution, residual); residual.aypx(-1., rhs)
        true_norm, initial_norm = residual.norm(), rhs.norm()
        target = max(1e-14, 1e-12*initial_norm)
        reason = int(ksp.getConvergedReason())
        rtol, atol, divergence, maxits = ksp.getTolerances()
        if (ksp.getType() != 'cg' or ksp.getPC().getType() != 'none' or ksp.getInitialGuessNonzero()
                or ksp.getNormType() != PETSc.KSP.NormType.UNPRECONDITIONED
                or (rtol, atol, maxits) != (1e-12, 1e-14, 10000)):
            raise ComparisonError('PETSc effective policy differs from protocol')
        if reason <= 0 or not np.isfinite(true_norm) or true_norm > target:
            raise ComparisonError(f'PETSc true residual rejected: reason={reason}, true={true_norm}, target={target}')
        summary = {'algorithm': 'cg', 'preconditioner': 'identity', 'completed_iterations': ksp.getIterationNumber(),
                   'initial_residual_norm': initial_norm, 'reported_residual_norm': ksp.getResidualNorm(),
                   'true_residual_norm': true_norm, 'residual_target': target, 'convergence_reason': reason,
                   'relative_tolerance': rtol, 'absolute_tolerance': atol, 'maximum_iterations': maxits,
                   'divergence_tolerance': divergence, 'norm_type': 'unpreconditioned', 'initial_guess_nonzero': False}
        residual.destroy(); ksp.destroy()
        return solution, summary

    def execute(*, phases=False, rhs_form=load):
        if phases:
            (matrix, rhs), assembly_time = clock(lambda: assemble(rhs_form))
            (solution, summary), solve_time = clock(lambda: solve(matrix, rhs))
        else:
            matrix, rhs = assemble(rhs_form); solution, summary = solve(matrix, rhs)
            assembly_time = solve_time = None
        function = fem.Function(space)
        function.x.array[:] = solution.getArray(readonly=True); function.x.scatter_forward()
        solution.destroy()
        return function, summary, matrix, rhs, assembly_time, solve_time

    def extract(result):
        function, summary, matrix, rhs, _, _ = result
        for row in boundary_scalar:
            columns, data = matrix.getRow(int(row)); check_identity_row(int(row), columns, data)
        free_is = PETSc.IS().createGeneral(free, comm=MPI.COMM_SELF)
        bc_is = PETSc.IS().createGeneral(boundary_scalar, comm=MPI.COMM_SELF)
        cross = matrix.createSubMatrix(free_is, bc_is)
        if np.any(cross.getValuesCSR()[2] != 0.):
            raise ComparisonError('essential columns were not eliminated')
        cross.destroy(); free_is.destroy(); bc_is.destroy()
        values = function.x.array.copy().reshape(-1, 2)
        matrix.destroy(); rhs.destroy()
        return coordinates, values, summary

    report = {'environment': env, 'memory_after_import': baseline, 'memory_after_preparation': memory(),
              'mesh_measurement': mesh_time, 'preparation_measurement': preparation,
              'dofs': {'total': 2*count, 'free': 2*(args.cells-1)**2, 'stored_algebraic': 2*count},
              'jit': {'cache': 'fresh directory', 'cffi_extra_compile_args': ['-O3']}}

    def diagnostic():
        result = execute(phases=True)
        assembly_time, solve_time = result[-2:]
        extract(result)
        # A second diagnostic-only solve supplies the nonzero equilibrium falsifier.
        affine_load = fem.form(ufl.inner(ufl.as_vector((1., 0.)), test)*measure, jit_options=jit)
        probe = execute(rhs_form=affine_load)
        function, summary = probe[:2]
        raw_matrix = petsc.assemble_matrix(stiffness); raw_matrix.assemble()
        raw_rhs = petsc.assemble_vector(affine_load)
        raw_rhs.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solution = raw_matrix.createVecRight(); solution.array[:] = function.x.array
        imbalance = raw_rhs.duplicate(); raw_matrix.mult(solution, imbalance); imbalance.axpy(-1., raw_rhs)
        reaction = imbalance.getArray(readonly=True).reshape(-1, 2)[boundary].sum(axis=0)
        body_force = raw_rhs.getArray(readonly=True).reshape(-1, 2).sum(axis=0)
        check_equilibrium(reaction, body_force)
        report['equilibrium_probe'] = {'scope': 'separate untimed affine-potential solve after cold execution',
                                       'reaction': reaction.tolist(), 'body_force': body_force.tolist(), 'solve': summary}
        extract(probe)
        for item in (raw_matrix, raw_rhs, solution, imbalance):
            item.destroy()
        if args.mode == 'correctness-only':
            return {'required_phases_available': True, 'phase_inventory': ['assembly', 'linear_solve']}
        return {'required_phases_available': True, 'assembly': assembly_time, 'linear_solve': solve_time,
                'assembly_scope': 'matrix/vector allocation and assembly, lifting, ghost accumulation, essential BC',
                'linear_solve_scope': 'KSP allocation/configuration, zero start, solve, true residual and summary',
                'execution_extra': 'field recovery included; protocol diagnostics and destruction excluded'}
    return report, execute, extract, diagnostic


def main(args):
    report, execute, extract, diagnostic = prepare(args)
    finish(args, 'fenicsx', report, execute, extract, diagnostic, accuracy=check_accuracy)


if __name__ == '__main__':
    main(arguments())
