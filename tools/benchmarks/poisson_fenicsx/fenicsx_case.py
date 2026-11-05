"""Standard DOLFINx assembly and PETSc solve, with declared operation boundaries."""
import time
IMPORT_START = time.perf_counter_ns()
import json
import os
import sysconfig
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import basix
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem import petsc
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl
from accuracy import check_identity_row, q1_vertex_dof_coordinates
from common import ComparisonError, clock, environment, finish, memory, worker_arguments
IMPORT_SECONDS = (time.perf_counter_ns()-IMPORT_START)/1e9


def main(args):
    if dolfinx.__version__ != "0.11.0" or basix.__version__ != "0.11.0":
        raise ComparisonError("this locked comparison requires DOLFINx and Basix 0.11.0")
    if PETSc.ScalarType != np.float64 or MPI.COMM_WORLD.size != 1:
        raise ComparisonError("comparison requires real binary64 PETSc and one MPI rank")
    env = environment("fenicsx", IMPORT_SECONDS)
    env.update(mpi_library=MPI.Get_library_version(), petsc_version=PETSc.Sys.getVersionInfo(),
               petsc_scalar_type=str(PETSc.ScalarType), petsc_integer_type=str(PETSc.IntType),
               compiler_environment={key: os.environ.get(key) for key in
                                     ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS")},
               python_compiler_config={key: sysconfig.get_config_var(key) for key in ("CC", "CFLAGS", "LDSHARED")},
               blas=np.show_config(mode="dicts"))
    baseline = memory()
    domain, mesh_time = clock(lambda: mesh.create_unit_square(
        MPI.COMM_SELF, args.cells, args.cells, cell_type=mesh.CellType.quadrilateral))
    if domain.topology.cell_type != mesh.CellType.quadrilateral:
        raise ComparisonError("external mesh is not quadrilateral")
    points, weights = basix.make_quadrature(basix.CellType.quadrilateral, 3,
                                           rule=basix.QuadratureType.gauss_jacobi)
    expected_1d = (1.0 + np.array([-1.0, 1.0])/np.sqrt(3.0))/2.0
    expected = np.array(sorted((x, y) for x in expected_1d for y in expected_1d))
    actual = np.array(sorted(map(tuple, points)))
    if actual.shape != (4, 2) or weights.shape != (4,):
        raise ComparisonError("Basix did not provide four tensor Gauss points")
    # This conformance allowance covers only evaluation of the closed-form
    # quadrature coordinates, not PDE accuracy or a solver tolerance.
    np.testing.assert_array_max_ulp(actual, expected, maxulp=8)
    np.testing.assert_array_max_ulp(weights, np.full(4, 0.25), maxulp=8)
    env["solve_quadrature"] = {"points": points.tolist(), "weights": weights.tolist(),
                               "rule": "gauss_jacobi", "degree": 3}
    jit_directory = args.output.parent / "fresh-jit-cache"
    if jit_directory.exists():
        raise ComparisonError("cold FEniCSx JIT directory must not already exist")
    jit_directory.mkdir(parents=True)

    def prepare():
        space = fem.functionspace(domain, ("Lagrange", 1))
        if space.dofmap.index_map.size_global != (args.cells+1)**2 or space.dofmap.index_map_bs != 1:
            raise ComparisonError("external Q1 dof inventory differs from the declared grid")
        domain.topology.create_connectivity(1, 2)
        facets = mesh.exterior_facet_indices(domain.topology)
        boundary = np.unique(fem.locate_dofs_topological(space, 1, facets))
        if len(boundary) != 4*args.cells:
            raise ComparisonError("external Dirichlet boundary inventory is incomplete")
        bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary, space)
        trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
        x = ufl.SpatialCoordinate(domain)
        measure = ufl.Measure("dx", domain=domain, metadata={
            "quadrature_degree": 3, "quadrature_rule": "gauss_jacobi"})
        source = 2*np.pi**2*ufl.sin(np.pi*x[0])*ufl.sin(np.pi*x[1])
        jit = {"cache_dir": str(jit_directory), "cffi_extra_compile_args": ["-O3"]}
        a = fem.form(ufl.inner(ufl.grad(trial), ufl.grad(test))*measure, jit_options=jit)
        load = fem.form(source*test*measure, jit_options=jit)
        return space, bc, boundary, a, load

    (space, bc, boundary, a, load), preparation = clock(prepare)
    # Q1 dofs belong to vertices topologically. tabulate_dof_coordinates
    # evaluates a floating geometry map and can round even a dyadic vertex.
    # Export the exact mesh Geometry vertex by its actual dof/entity binding;
    # never snap or match approximate coordinates by proximity.
    domain.topology.create_connectivity(0, 2)
    vertex_bindings = []
    for vertex in range(domain.topology.index_map(0).size_local):
        entity = np.array([vertex], dtype=np.int32)
        dof = fem.locate_dofs_topological(space, 0, entity)
        geometry_index = mesh.entities_to_geometry(domain, 0, entity, permute=False)
        if geometry_index.shape != (1, 1):
            raise ComparisonError("Q1 vertex does not identify exactly one Geometry vertex")
        vertex_bindings.append((vertex, dof.tolist(), domain.geometry.x[geometry_index[0, 0], :2]))
    coordinates = q1_vertex_dof_coordinates(vertex_bindings, (args.cells+1)**2)
    env["coordinate_export"] = "exact mesh Geometry vertices through topological Q1 vertex/dof correspondence"
    free = np.setdiff1d(np.arange((args.cells+1)**2, dtype=PETSc.IntType), boundary)

    def verify_boundary(matrix):
        # Untimed protocol check on the actual returned assembly, including
        # columns. Never inflate FEniCSx execution time with this diagnostic.
        for row in boundary:
            columns, data = matrix.getRow(int(row))
            check_identity_row(int(row), columns, data)
        free_is = PETSc.IS().createGeneral(free, comm=MPI.COMM_SELF)
        boundary_is = PETSc.IS().createGeneral(boundary, comm=MPI.COMM_SELF)
        cross = matrix.createSubMatrix(free_is, boundary_is)
        if np.any(cross.getValuesCSR()[2] != 0.0):
            raise ComparisonError("external essential boundary columns were not eliminated")
        cross.destroy(); free_is.destroy(); boundary_is.destroy()


    def execute():
        def assemble():
            matrix = petsc.assemble_matrix(a, bcs=[bc])
            matrix.assemble()
            rhs = petsc.assemble_vector(load)
            petsc.apply_lifting(rhs, [a], bcs=[[bc]])
            rhs.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(rhs, [bc])
            return matrix, rhs

        (matrix, rhs), assembly_time = clock(assemble)

        def solve():
            solution = matrix.createVecRight()
            solution.set(0.0)
            ksp = PETSc.KSP().create(MPI.COMM_SELF)
            ksp.setOperators(matrix)
            ksp.setType("bcgs"); ksp.getPC().setType("none")
            ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
            ksp.setInitialGuessNonzero(False)
            ksp.setTolerances(rtol=1e-12, atol=1e-14, max_it=10000)
            ksp.solve(rhs, solution)
            rtol, atol, divergence, maxits = ksp.getTolerances()
            if (ksp.getType() != "bcgs" or ksp.getPC().getType() != "none"
                    or ksp.getNormType() != PETSc.KSP.NormType.UNPRECONDITIONED
                    or ksp.getInitialGuessNonzero() or (rtol, atol, maxits) != (1e-12, 1e-14, 10000)):
                raise ComparisonError("PETSc effective solve policy differs from the declared exact tuple")
            residual = rhs.duplicate()
            matrix.mult(solution, residual)
            residual.aypx(-1.0, rhs)
            true_norm = residual.norm()
            rhs_norm = rhs.norm()
            target = max(1e-14, 1e-12*rhs_norm)
            reason = int(ksp.getConvergedReason())
            if reason <= 0 or not np.isfinite(true_norm) or true_norm > target:
                raise ComparisonError(f"PETSc solve did not meet the matched true residual: reason={reason}, true={true_norm}, target={target}")
            summary = {"algorithm": "bicgstab", "preconditioner": "identity",
                       "completed_iterations": ksp.getIterationNumber(), "initial_residual_norm": rhs_norm,
                       "reported_residual_norm": ksp.getResidualNorm(), "true_residual_norm": true_norm,
                       "residual_target": target, "convergence_reason": reason,
                       "initial_guess_nonzero": ksp.getInitialGuessNonzero(), "norm_type": "unpreconditioned",
                       "relative_tolerance": rtol, "absolute_tolerance": atol, "maximum_iterations": maxits,
                       "divergence_tolerance": divergence}
            residual.destroy(); ksp.destroy()
            return solution, summary

        (solution, summary), solve_time = clock(solve)
        function = fem.Function(space)
        function.x.array[:] = solution.getArray(readonly=True)
        function.x.scatter_forward()
        solution.destroy()
        return function, summary, assembly_time, solve_time, matrix, rhs

    def extract(result):
        function, summary, assembly_time, solve_time, matrix, rhs = result
        verify_boundary(matrix)
        values = function.x.array.copy()
        matrix.destroy(); rhs.destroy()
        return coordinates, values, summary

    report = {"environment": env, "memory_after_import": baseline, "memory_after_preparation": memory(),
              "mesh_measurement": mesh_time, "preparation_measurement": preparation,
              "dofs": {"total": (args.cells+1)**2, "free": (args.cells-1)**2, "stored_algebraic": (args.cells+1)**2},
              "jit": {"cache": "fresh directory", "cffi_extra_compile_args": ["-O3"]}}
    # A separate diagnostic execution retains native phase brackets, matching
    # the separate Eqiora profile execution rather than partitioning medians.
    def diagnostic():
        result = execute()
        _, _, assembly_time, solve_time, _, _ = result
        extract(result)
        if args.mode == "correctness-only":
            return {"required_phases_available": True, "phase_inventory": ["assembly", "linear_solve"]}
        return {"required_phases_available": True, "assembly": assembly_time, "linear_solve": solve_time,
                "assembly_scope": "matrix/vector allocation+assembly, lifting, ghost accumulation and BC application",
                "linear_solve_scope": "KSP allocation/configuration+zero guess+KSPSolve+independent true-residual acceptance",
                "execution_extra": "field recovery included; protocol diagnostics/output inspection/result destruction excluded"}

    finish(args, "fenicsx", report, execute, extract, diagnostic)


if __name__ == "__main__":
    args = worker_arguments()
    try:
        main(args)
    except (ComparisonError, ImportError) as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        raise SystemExit(2)
