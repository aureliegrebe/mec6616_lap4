import numpy as np
from mesh.meshGenerator import MeshGenerator
from mesh.meshPlotter import MeshPlotter

import matplotlib.pyplot as plt
from solveur2d import Solver_2D, BC

# Paramètres des figures
plt.rcParams.update({"font.size": 22})
plt.rc("legend", fontsize=15)
plt.rcParams["figure.figsize"] = [8.5, 8.5]

# Différentes tailles de cellules pour calculer l'erreur
CELL_LENGTHS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
N_LIST = [2, 5, 10, 20, 50, 100]

# Constantes
T_A = 100 # °C
T_B = 200 # °C
L = 0.02 # m
K = 0.5 # W/m.K
Q = 1e6 # W/m3
N = 25 # nombre d'éléments pour le maillage triangle
Nx = 30 # nombre d'éléments en x pour le maillage quad
Ny = 10 # nombre d'éléments en y pour le maillage quad

# Fonction analytique de l'exemple 4.2 de Versteeg
ANALYTIC_FUNC = lambda x, y: ((T_B - T_A)/L + Q/(2*K)*(L-x))*x + T_A

# Nombre de trous attendus pour le mesh généré
trous = 0

def init_solver(quad:bool = True):
    mesher = MeshGenerator()
    
    # Générer un mesh 
    if quad: # Maillage quadratique structuré
        mesh_parameters = {'mesh_type': 'QUAD',
                   'Nx': Nx, 'Ny': Ny                  
                   }
    else: # Maillage triangulaire non-structuré
        mesh_parameters = {'mesh_type': 'TRI', 'lc': L/N
                   }
        
    mesh_obj = mesher.rectangle([0.0, L, 0.0, 1/2*L], mesh_parameters)
    
    # Conditions aux limites
    bc_list = [
        BC("DIRICHLET", T_A),
        BC("NEUMANN", 0),
        BC("DIRICHLET", T_B),
        BC("NEUMANN", 0),
    ]

    source_func = lambda x, y: Q 
    
    return Solver_2D(mesh_obj, K, bc_list, source_func)  


def solve_diffusion(solver: Solver_2D, cross_dif: bool = True):
    
    if cross_dif:
        solver.iterate(10)
        print(solver.residuel)
    else:
        solver.solve(True) # Sans cross-diffusion on calcule uniquement la première iteration
    
    x_analytique = np.linspace(0, L)
    plt.figure()
    plt.xlabel("x")
    plt.ylabel("T(°C)")
    plt.plot(x_analytique, ANALYTIC_FUNC(x_analytique, x_analytique))
    plt.plot(solver.centres[:,0],solver.phi,".")
    print(solver.residuel)
    
    # TODO: Calcul de l'erreur MMS + ordre de convergence

    epsilon2 = []

def convergence(solver: Solver_2D, quad=False, cross_diff=False):
    mesher = MeshGenerator()
    if quad:
        mesh_parameters = {'mesh_type': 'QUAD',
                   'Nx': Nx, 'Ny': Ny                  
                   }
    else:
        mesh_parameters = {'mesh_type': 'TRI',
                   'lc': L/N
                   }
    # Conditions aux limites
    bc_list = [
        BC("DIRICHLET", T_A),
        BC("NEUMANN", 0),
        BC("DIRICHLET", T_B),
        BC("NEUMANN", 0),
    ]
    solver.set_mesh(mesher.rectangle([0.0, L, 0.0, 1/2*L], mesh_parameters), bc_list)
    epsilon2 = []
    h_list = []
    
    # Caclul de l'erreur RMS
    for (i, n) in enumerate(N_LIST):
        if quad:
            mesh_parameters["Nx"] = n
            mesh_parameters["Ny"] = n
        else:
            mesh_parameters["lc"] = L/n
        solver.set_mesh(mesher.rectangle([0.0, L, 0.0, 1/2*L], mesh_parameters), bc_list)
        exact_sol = np.array(ANALYTIC_FUNC(solver.centres[:,0],
                                                solver.centres[:,1]))
        if cross_diff:
            solver.iterate()
        else:
            solver.solve(first_iter=True)
        err = np.linalg.norm(solver.phi - exact_sol, axis=0)
        rms = np.sqrt(np.average(err**2))
        epsilon2.append(rms)
        h_list.append(solver.get_h())

        if i == 3:
            plt.figure()
            plt.plot(solver.centres[:,0], exact_sol, 'o')
            plt.plot(solver.centres[:,0], solver.phi, '.')
            # plt.quiver(solver.centres[:,0], solver.centres[:,1],
            #            solver.phi()[:,0], solver.get_grad()[:,1])
            plt.xlabel("x")
            plt.ylabel("y")
        #     if exp:
        #         plt.savefig("./figures/field_grad_exp.png")
        #     elif quad:
        #         plt.savefig("./figures/field_grad_sin_quad.png")
        #     else:
        #         plt.savefig("./figures/field_grad_sin.png")

    p = float(np.log(epsilon2[-1]/epsilon2[-2])/np.log(N_LIST[-2]/N_LIST[-1]))
    print(f"Ordre de convergence: {p:.5f}")

    # Tracer l'erreur
    plt.figure()
    # plt.plot(L/np.array(N_LIST), epsilon2)
    plt.plot(h_list, epsilon2)
    plt.grid()
    plt.xlabel("Cell length (m)")
    plt.ylabel("RMS (err)")
    plt.yscale("log")
    plt.xscale("log")
    # plt.axis("equal")
    if quad:
        plt.savefig("./figures/rms_quad.pdf")
    elif cross_diff:
        plt.savefig("./figures/rms_tri_cross.pdf")
    else:
        plt.savefig("./figures/rms_tri_nocross.pdf")



def main():
    plotter = MeshPlotter()
    solver_tri_cd = init_solver(False)
    solver_tri_no_cd = init_solver(False)
    solver_quad = init_solver()
    plotter.plot_mesh(solver_tri_cd.mesh)
    plotter.plot_mesh(solver_quad.mesh)
    # Vérifier la relation d'Euler
    print('La relation Euler pour le mesh triangulaire est elle respectée ?', solver_tri_cd.euler_check(0))
    print('La relation Euler pour le mesh quadratique est elle respectée ?', solver_quad.euler_check(0))
    
    solve_diffusion(solver_tri_cd) # Maillage triangulaire avec cross-diffusion
    solve_diffusion(solver_tri_no_cd, False) # Maillage triangulaire sans cross-diffusion
    solve_diffusion(solver_quad, False) # Maillage quadratique structuré
    solver_tri_cd.to_vtk("./vtk/tri_cd.vtk")
    solver_tri_no_cd.to_vtk("./vtk/tri_no_cd.vtk")
    solver_quad.to_vtk("./vtk/quad.vtk")

    convergence(solver_tri_cd)
    convergence(solver_tri_cd, cross_diff=True)
    convergence(solver_quad, quad=True)

    plt.show()
    

if __name__ == '__main__':
    main()

