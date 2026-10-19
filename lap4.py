from cProfile import label
import numpy as np
from mesh.meshGenerator import MeshGenerator
from mesh.meshPlotter import MeshPlotter
import sympy as sp
from cycler import cycler


import matplotlib.pyplot as plt
from solveurFVM import Solver_FVM, BC, BCType

N_LIST = [2, 5, 10, 20, 50]
N = 20
U = 1

def couette(p_fac, theta = 0, quad=True, cross_diff=False):
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

    # Solution exacte
    analytical_sol = lambda y: y * (1 + p_fac * (1 - y))

    p_func = lambda x, y: -2 * p_fac * (x * np.cos(theta) + y * np.sin(theta))

    u_bc_list = [
        [BC(BCType.NEUMANN, 0), BC(BCType.NEUMANN, 0)],
        [BC(BCType.DIRICHLET, 0), BC(BCType.DIRICHLET, 0)],
        [BC(BCType.NEUMANN, 0), BC(BCType.NEUMANN, 0)],
        [BC(BCType.DIRICHLET, U * np.cos(theta)), BC(BCType.DIRICHLET,U*np.sin(theta))],
    ]

    p_bc_list = [
        BC(BCType.DIRICHLET_FUNC, p_func),
        BC(BCType.DIRICHLET_FUNC, p_func),
        BC(BCType.DIRICHLET_FUNC, p_func),
        BC(BCType.DIRICHLET_FUNC, p_func),
    ]

    vel_func = lambda x, y: (analytical_sol(y), 0)

    solver = Solver_FVM(mesh_obj, 1, 1, p_bc_list, u_bc_list,
                        pressure_func=p_func, velocity_func=vel_func,
                        alpha=0.5, theta = theta)

    if quad:
        meshstr = "_quad"
    else:
        meshstr = "_tri"
    if cross_diff:
        cdstr = "_cd"
    else:
        cdstr = "_nocd"
    thetastr = f"_theta_{theta}"
    pstr = f"_P_{p_fac}"

    if cross_diff:
        solver.cross_diff_iter = 5


    # plotter.plot_mesh(mesh_obj, label_faces=True)
    exact_y = np.linspace(0, 1, 100)

    solver.solve(iter_max=500, res_max=1e-5)
    plt.figure()
    plt.quiver(solver.centres[:,0], solver.centres[:,1],
                   solver.u[:,0], solver.u[:,1])
    plt.savefig("./figures/couette_vec" + meshstr + cdstr + thetastr + pstr + ".pdf")
    
    turned_x = (solver.centres[:,0]*np.cos(theta) + solver.centres[:,1]*np.sin(theta) )
    turned_y = (-solver.centres[:,0]*np.sin(theta) + solver.centres[:,1]*np.cos(theta) )
    start_slice = (turned_x < 0.05 )
    mid_slice = (turned_x > 0.45 ) & (turned_x < 0.55)
    end_slice = (turned_x > 0.95 )
    turned_u = solver.u[:,0] * np.cos(theta) + solver.u[:,1] * np.sin(theta)
    turned_v = -solver.u[:,0] * np.sin(theta) + solver.u[:,1] * np.cos(theta)
    plt.figure()
    plt.plot(exact_y, analytical_sol(exact_y), label="analytique")
    plt.plot(turned_y[start_slice], turned_u[start_slice], ".", label=f"début")
    plt.plot(turned_y[mid_slice], turned_u[mid_slice], "*", label=f"milieu")
    plt.plot(turned_y[end_slice], turned_u[end_slice], "x", label=f"fin")
    plt.legend()
    plt.savefig("./figures/couette_slice" + meshstr + cdstr + thetastr + pstr + ".pdf")


    h_list = []
    epsilon2 = []
    # Calcul de l'erreur en fonction du nombre de points
    for (i, n) in enumerate(N_LIST):
        if quad:
            mesh_parameters["Nx"] = n
            mesh_parameters["Ny"] = n
        else:
            mesh_parameters["lc"] = 1/n
        solver.set_mesh(mesher.rectangle([0,1,0,1], mesh_parameters),p_bc_list, u_bc_list)
        turned_x = (solver.centres[:,0]*np.cos(theta) + solver.centres[:,1]*np.sin(theta) )
        turned_y = (-solver.centres[:,0]*np.sin(theta) + solver.centres[:,1]*np.cos(theta) )
        exact_sol = np.array([analytical_sol(y_val) for y_val in turned_y])
        solver.solve(iter_max=500, res_max=1e-5)
        turned_u = solver.u[:,0] * np.cos(theta) + solver.u[:,1] * np.sin(theta)
        err = turned_u - exact_sol
        rms = np.sqrt(np.average(err**2))
        epsilon2.append(rms)
        h_list.append(solver.get_h())

    # plt.xlabel("x (m)")
    # plt.ylabel("Φ")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("./figures/couette" + meshstr + cdstr + thetastr + ".pdf")
    # Tracer l'erreur
    plt.figure()
    # plt.plot(L/np.array(N_LIST), epsilon2)
    plt.plot(h_list, epsilon2, ".-")
    # plt.plot(epsilon2, ".-")
    plt.grid()
    plt.xlabel("h (m)")
    plt.ylabel("RMS (err)")
    plt.yscale("log")
    plt.xscale("log")
    ax = plt.gca()
    plt.savefig("./figures/couette_err" + meshstr + cdstr + thetastr + pstr + ".pdf")

    p = float(np.log(epsilon2[-1]/epsilon2[-2])/np.log(h_list[-1]/h_list[-2]))
    print(meshstr + cdstr + thetastr + f" ordre observé: {p:.5f}")




def main():
    # couette(0)
    # couette(1)
    # couette(-3)
    # couette(0, quad=False, cross_diff=True)
    # couette(1, quad=False, cross_diff=True)
    # couette(-3, quad=False, cross_diff=True)
    # couette(0, theta=np.pi/6)
    # couette(1, theta=np.pi/6)
    # couette(-3, theta=np.pi/6)
    couette(0, theta=np.pi/6, quad=False, cross_diff=True)
    couette(1, theta=np.pi/6, quad=False, cross_diff=True)
    couette(-3, theta=np.pi/6, quad=False, cross_diff=True)
    # couette(-3, quad=False, cross_diff=True)
    plt.show()

if __name__ == "__main__":
    main()
