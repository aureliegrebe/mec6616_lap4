from cProfile import label
import numpy as np
from mesh.meshGenerator import MeshGenerator
from mesh.meshPlotter import MeshPlotter
import sympy as sp
from cycler import cycler


import matplotlib.pyplot as plt
from solveurFVM import Solver_FVM, BC, BCType

N = 20
U = 1

def couette(p_fac,quad=True, cross_diff=False):
    mesher = MeshGenerator()
    # Générer un mesh 
    if quad: # Maillage quadratique structuré
        mesh_parameters = {'mesh_type': 'QUAD',
                   'Nx': N, 'Ny': N                  
                   }
    else: # Maillage triangulaire non-structuré
        mesh_parameters = {'mesh_type': 'TRI', 'lc': 1/N
                   }
    mesh_obj = mesher.rectangle([0,1,0,1], mesh_parameters)

    plotter = MeshPlotter()

    p_func = lambda x, y: -2 * p_fac * x

    u_bc_list = [
        [BC(BCType.NEUMANN, 0), BC(BCType.NEUMANN, 0)],
        [BC(BCType.DIRICHLET, 0), BC(BCType.DIRICHLET, 0)],
        [BC(BCType.NEUMANN, 0), BC(BCType.NEUMANN, 0)],
        [BC(BCType.DIRICHLET, U), BC(BCType.DIRICHLET, 0)],
    ]

    p_bc_list = [
        BC(BCType.DIRICHLET_FUNC, p_func),
        BC(BCType.DIRICHLET_FUNC, p_func),
        BC(BCType.DIRICHLET_FUNC, p_func),
        BC(BCType.DIRICHLET_FUNC, p_func),
    ]

    solver = Solver_FVM(mesh_obj, 1, 1, p_bc_list, u_bc_list,
                        pressure_func=p_func, alpha=0.7)

    # plotter.plot_mesh(mesh_obj, label_faces=True)

    solver.solve(iter_max=500, res_max=1e-5)
    plt.figure()
    plt.quiver(solver.centres[:,0], solver.centres[:,1],
                   solver.u[:,0], solver.u[:,1])

    start_slice = (solver.centres[:,0] < 0.05 )
    mid_slice = (solver.centres[:,0] > 0.45 ) & (solver.centres[:,0] < 0.55)
    end_slice = (solver.centres[:,0] > 0.95 )
    plt.figure()
    plt.plot(solver.centres[start_slice,1], solver.u[start_slice, 0], ".", label=f"début")
    plt.plot(solver.centres[mid_slice,1], solver.u[mid_slice, 0], "*", label=f"milieu")
    plt.plot(solver.centres[end_slice,1], solver.u[end_slice, 0], "x", label=f"fin")
    plt.legend()



def main():
    couette(-3)
    # couette(1)
    # couette(-6)
    plt.show()

if __name__ == "__main__":
    main()
