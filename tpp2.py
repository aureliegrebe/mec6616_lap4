from cProfile import label
import numpy as np
from mesh.meshGenerator import MeshGenerator
from mesh.meshPlotter import MeshPlotter

import matplotlib.pyplot as plt
from solveur2d import Solver_2D, BC, BCType

# Paramètres des figures
plt.rcParams.update({"font.size": 22})
plt.rc("legend", fontsize=15)
plt.rcParams["figure.figsize"] = [8.5, 8.5]

# Différentes tailles de cellules pour calculer l'erreur
CELL_LENGTHS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
N_LIST = [2, 5, 10, 20, 50, 100]

# Constantes
N = 25 # nombre d'éléments pour le maillage triangle
Nx = 30 # nombre d'éléments en x pour le maillage quad
Ny = 10 # nombre d'éléments en y pour le maillage quad
L = 2

THERM_COND = 1
DENSITY = 1

def velocity (x,y):
    return ((2*x**2 - x**4 - 1) * (y - y**3), -(2*y**2 - y**4 -1)*(x - x**3))

def T_mms(x,y):
    return (400 + 50*np.cos(np.pi*x) + 100*np.sin(np.pi*x*y))

def source(x,y):
    return np.pi*(-100.0*x**2*(x**2 - 1)*(y**4 - 2*y**2 + 1)*np.cos(np.pi*x*y) 
                                 + 100.0*np.pi*x**2*np.sin(np.pi*x*y) + 50.0*y*(y**2 - 1)*
                                 (2*y*np.cos(np.pi*x*y) - np.sin(np.pi*x))*(x**4 - 2*x**2 + 1) 
                                 + np.pi*(100.0*y**2*np.sin(np.pi*x*y) + 50.0*np.cos(np.pi*x)))

def init_solver(quad:bool = True, therm_cond = float, density = float, method = str):
    mesher = MeshGenerator()
    
    # Générer un mesh 
    if quad: # Maillage quadratique structuré
        mesh_parameters = {'mesh_type': 'QUAD',
                   'Nx': Nx, 'Ny': Ny                  
                   }
    else: # Maillage triangulaire non-structuré
        mesh_parameters = {'mesh_type': 'TRI', 'lc': 2/N
                   }
        
    mesh_obj = mesher.rectangle([-1,1,-1,1], mesh_parameters)
    
    # Conditions aux limites
    
    bc_list = [
        BC(BCType.DIRICHLET_FUNC, lambda x, y: T_mms(-1,y)),
        BC(BCType.DIRICHLET_FUNC, lambda x, y: T_mms(x,-1)),
        BC(BCType.DIRICHLET_FUNC, lambda x, y: T_mms(1,y)),
        BC(BCType.DIRICHLET_FUNC, lambda x, y: T_mms(x,1)),
    ]
    
    return Solver_2D(mesh_obj, therm_cond, density, method, bc_list, source, velocity)    

def solve_diff_conv(solver = Solver_2D):
    solver.solve(True)
    solver.iterate()
    print(solver.residuel)
    print(np.sum(solver.areas))

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(solver.centres[:,0],solver.centres[:,1],solver.phi, c = solver.phi, cmap = "viridis")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Solution numérique")

def convergence(solver: Solver_2D, quad=False, cross_diff=False, upwind=True):
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
        BC(BCType.DIRICHLET_FUNC, lambda x, y: T_mms(-1,y)),
        BC(BCType.DIRICHLET_FUNC, lambda x, y: T_mms(x,-1)),
        BC(BCType.DIRICHLET_FUNC, lambda x, y: T_mms(1,y)),
        BC(BCType.DIRICHLET_FUNC, lambda x, y: T_mms(x,1)),
    ]
    
    epsilon2 = []
    h_list = []
    solver.set_source_field(source)
    
    # Caclul de l'erreur RMS
    for (i, n) in enumerate(N_LIST):
        if quad:
            mesh_parameters["Nx"] = n
            mesh_parameters["Ny"] = n
        else:
            mesh_parameters["lc"] = L/n
        solver.set_mesh(mesher.rectangle([-1,1,-1,1], mesh_parameters), bc_list)
        exact_sol = np.array(T_mms(solver.centres[:,0],
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
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(solver.centres[:,0], solver.centres[:,1], exact_sol, label="analytique")
            ax.scatter(solver.centres[:,0], solver.centres[:,1], solver.phi, label="numérique")
            # plt.plot(solver.centres[:,1], solver.phi, '.')
            # plt.quiver(solver.centres[:,0], solver.centres[:,1],
            #            solver.phi()[:,0], solver.get_grad()[:,1])
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()

    # p = float(np.log(epsilon2[-1]/epsilon2[-2])/np.log(N_LIST[-2]/N_LIST[-1]))
    p = float(np.log(epsilon2[-1]/epsilon2[-2])/np.log(h_list[-1]/h_list[-2]))
    if upwind:
        print(f"Ordre de convergence (Upwind) : {p:.5f}")
    else:
        print(f"Ordre de convergence (centré) : {p:.5f}")


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
    # ax.xaxis.minorticks_off()
    # plt.xlim([0.005, 0.5])
    
    if upwind and quad:
        plt.savefig("./figures/mms_upwind_quad_rms.pdf")
    elif upwind and cross_diff:
        plt.savefig("./figures/mms_upwind_tri_cd_rms.pdf")
    elif upwind:
        plt.savefig("./figures/mms_upwind_tri_nocd_rms.pdf")
    elif quad:
        plt.savefig("./figures/mms_centre_quad_rms.pdf")
    elif cross_diff:
        plt.savefig("./figures/mms_centre_tri_cd_rms.pdf")
    else:
        plt.savefig("./figures/mms_centre_tri_nocd_rms.pdf")




def main():   
    for k in [1,10,100,1000]:
        upwind_solver = init_solver(True, k, DENSITY, "UPWIND")
        convergence(upwind_solver,quad=True, cross_diff=False, upwind=True)
        centre_solver = init_solver(True, k, DENSITY, "CENTRE")
        convergence(centre_solver,quad=True, cross_diff=False, upwind=False)
    plt.show()
    
if __name__ == '__main__':
    main()

