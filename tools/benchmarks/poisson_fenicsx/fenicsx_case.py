"""Standard DOLFINx assembly and PETSc solve, with declared operation boundaries."""
import time
IMPORT_START = time.perf_counter_ns()
import json
import os
import sysconfig
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import basix
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem import petsc
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl
from shared_accuracy import check_identity_row, q1_vertex_dof_coordinates, check_tensor_gauss
from shared_measurement import ComparisonError, clock, environment, finish, memory
from worker_arguments import worker_arguments
from accuracy import check_accuracy
IMPORT_SECONDS = (time.perf_counter_ns()-IMPORT_START)/1e9


def main(args):
    if dolfinx.__version__ != "0.11.0" or basix.__version__ != "0.11.0":
        raise ComparisonError("this locked comparison requires DOLFINx and Basix 0.11.0")
    if PETSc.ScalarType != np.float64 or MPI.COMM_WORLD.size != 1:
        raise ComparisonError("comparison requires real binary64 PETSc and one MPI rank")
    env = environment("fenicsx", IMPORT_SECONDS, protocol_root=Path(__file__).parent)
    env.update(mpi_library=MPI.Get_library_version(), petsc_version=PETSc.Sys.getVersionInfo(),
               petsc_scalar_type=str(PETSc.ScalarType), petsc_integer_type=str(PETSc.IntType),
               compiler_environment={key: os.environ.get(key) for key in
                                     ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS")},
               python_compiler_config={key: sysconfig.get_config_var(key) for key in ("CC", "CFLAGS", "LDSHARED")},
               blas=np.show_config(mode="dicts"))
    baseline = memory()
    dimension = args.dimension
    cell_type = mesh.CellType.quadrilateral if dimension == 2 else mesh.CellType.hexahedron
    create = mesh.create_unit_square if dimension == 2 else mesh.create_unit_cube
    domain, mesh_time = clock(lambda: create(MPI.COMM_SELF, *([args.cells]*dimension), cell_type=cell_type))
    if domain.topology.cell_type != cell_type or domain.topology.dim != dimension:
        raise ComparisonError("external mesh has the wrong cell type or dimension")
    basix_cell = basix.CellType.quadrilateral if dimension == 2 else basix.CellType.hexahedron
    points, weights = basix.make_quadrature(basix_cell, 3, rule=basix.QuadratureType.gauss_jacobi)
    check_tensor_gauss(points, weights, dimension)
    env["solve_quadrature"] = {"points": points.tolist(), "weights": weights.tolist(),
                               "rule": "gauss_jacobi", "degree": 3}
    jit_directory = args.output.parent / "fresh-jit-cache"
    if jit_directory.exists():
        raise ComparisonError("cold FEniCSx JIT directory must not already exist")
    jit_directory.mkdir(parents=True)

    def prepare():
        space = fem.functionspace(domain, ("Lagrange", 1))
        if space.dofmap.index_map.size_global != (args.cells+1)**dimension or space.dofmap.index_map_bs != 1:
            raise ComparisonError("external Q1 dof inventory differs from the declared grid")
        domain.topology.create_connectivity(dimension-1, dimension)
        facets = mesh.exterior_facet_indices(domain.topology)
        boundary = np.unique(fem.locate_dofs_topological(space, dimension-1, facets))
        if len(boundary) != (args.cells+1)**dimension - (args.cells-1)**dimension:
            raise ComparisonError("external Dirichlet boundary inventory is incomplete")
        bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary, space)
        trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
        x = ufl.SpatialCoordinate(domain)
        measure = ufl.Measure("dx", domain=domain, metadata={
            "quadrature_degree": 3, "quadrature_rule": "gauss_jacobi"})
        source = dimension*np.pi**2
        for axis in range(dimension):
            source *= ufl.sin(np.pi*x[axis])
        jit = {"cache_dir": str(jit_directory), "cffi_extra_compile_args": ["-O3"]}
        a = fem.form(ufl.inner(ufl.grad(trial), ufl.grad(test))*measure, jit_options=jit)
        load = fem.form(source*test*measure, jit_options=jit)
        return space, bc, boundary, a, load

    (space, bc, boundary, a, load), preparation = clock(prepare)
    # Q1 dofs belong to vertices topologically. tabulate_dof_coordinates
    # evaluates a floating geometry map and can round even a dyadic vertex.
    # Export the exact mesh Geometry vertex by its actual dof/entity binding;
    # never snap or match approximate coordinates by proximity.
    domain.topology.create_connectivity(0, dimension)
    vertex_bindings = []
    for vertex in range(domain.topology.index_map(0).size_local):
        entity = np.array([vertex], dtype=np.int32)
        dof = fem.locate_dofs_topological(space, 0, entity)
        geometry_index = mesh.entities_to_geometry(domain, 0, entity, permute=False)
        if geometry_index.shape != (1, 1):
            raise ComparisonError("Q1 vertex does not identify exactly one Geometry vertex")
        vertex_bindings.append((vertex, dof.tolist(), domain.geometry.x[geometry_index[0, 0], :dimension]))
    coordinates = q1_vertex_dof_coordinates(vertex_bindings, (args.cells+1)**dimension, dimension)
    env["coordinate_export"] = "exact mesh Geometry vertices through topological Q1 vertex/dof correspondence"
    free = np.setdiff1d(np.arange((args.cells+1)**dimension, dtype=PETSc.IntType), boundary)

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


    def assemble():
        matrix = petsc.assemble_matrix(a, bcs=[bc])
        matrix.assemble()
        rhs = petsc.assemble_vector(load)
        petsc.apply_lifting(rhs, [a], bcs=[[bc]])
        rhs.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(rhs, [bc])
        return matrix, rhs


    def solve(matrix, rhs):
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
        actual_policy = {"ksp_type": ksp.getType(), "pc_type": ksp.getPC().getType(),
                         "norm_type": int(ksp.getNormType())}
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
                   "divergence_tolerance": divergence, "actual_policy": actual_policy}
        residual.destroy(); ksp.destroy()
        return solution, summary


    def execute(*, collect_phases=False):
        # Ordinary execution has no nested phase timers. Only the separate
        # diagnostic call enables these Python operation brackets.
        if collect_phases:
            (matrix, rhs), assembly_time = clock(assemble)
            (solution, summary), solve_time = clock(lambda: solve(matrix, rhs))
        else:
            matrix, rhs = assemble()
            solution, summary = solve(matrix, rhs)
            assembly_time = solve_time = None
        function = fem.Function(space)
        function.x.array[:] = solution.getArray(readonly=True)
        function.x.scatter_forward()
        solution.destroy()
        return function, summary, assembly_time, solve_time, matrix, rhs

    def extract(result):
        function, summary, assembly_time, solve_time, matrix, rhs = result
        verify_boundary(matrix)
        policy = summary["actual_policy"]
        if (policy != {"ksp_type": "bcgs", "pc_type": "none", "norm_type": int(PETSc.KSP.NormType.UNPRECONDITIONED)}
                or summary["initial_guess_nonzero"] or
                (summary["relative_tolerance"], summary["absolute_tolerance"], summary["maximum_iterations"]) != (1e-12, 1e-14, 10000)):
            raise ComparisonError(f"PETSc effective policy differs from the declared tuple: {summary}")
        values = function.x.array.copy()
        matrix.destroy(); rhs.destroy()
        return coordinates, values, summary, {}

    report = {"dimension": args.dimension, "cells_per_axis": args.cells, "environment": env, "memory_after_import": baseline, "memory_after_preparation": memory(),
              "mesh_measurement": mesh_time, "preparation_measurement": preparation,
              "dofs": {"total": (args.cells+1)**dimension, "free": (args.cells-1)**dimension, "stored_algebraic": (args.cells+1)**dimension},
              "jit": {"cache": "fresh directory", "cffi_extra_compile_args": ["-O3"]}}
    # A separate diagnostic execution retains native phase brackets, matching
    # the separate Eqiora profile execution rather than partitioning medians.
    def diagnostic():
        result = execute(collect_phases=True)
        _, _, assembly_time, solve_time, _, _ = result
        extract(result)
        if args.mode == "correctness-only":
            return {"required_phases_available": True, "phase_inventory": ["assembly", "linear_solve"]}
        return {"required_phases_available": True, "assembly": assembly_time, "linear_solve": solve_time,
                "assembly_scope": "matrix/vector allocation+assembly, lifting, ghost accumulation and BC application",
                "linear_solve_scope": "KSP allocation/configuration+zero guess+KSPSolve+independent true-residual acceptance+effective-policy/summary getters",
                "execution_extra": "field recovery included; protocol diagnostics/output inspection/result destruction excluded"}

    finish(args, "fenicsx", report, execute, extract, diagnostic,
           accuracy=lambda coordinates, values, associated: check_accuracy(coordinates, values, args.cells, args.dimension))


if __name__ == "__main__":
    args = worker_arguments()
    try:
        main(args)
    except (ComparisonError, ImportError) as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        raise SystemExit(2)
