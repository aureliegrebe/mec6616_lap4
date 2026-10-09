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

# Nombre de trous attendus pour le mesh généré
trous = 0

def vel (x,y):
    return ((2*x**2 - x**4 - 1) * (y - y**3), -(2*y**2 - y**4 -1)*(x - x**3))

def T_mms(x,y):
    return (400 + 50*np.cos(np.pi*x) + 100*np.sin(np.pi*x*y))

def init_solver(quad:bool = True):
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

    source = lambda x, y: np.pi*(-100.0*x**2*(x**2 - 1)*(y**4 - 2*y**2 + 1)*np.cos(np.pi*x*y) 
                                 + 100.0*np.pi*x**2*np.sin(np.pi*x*y) + 50.0*y*(y**2 - 1)*
                                 (2*y*np.cos(np.pi*x*y) - np.sin(np.pi*x))*(x**4 - 2*x**2 + 1) 
                                 + np.pi*(100.0*y**2*np.sin(np.pi*x*y) + 50.0*np.cos(np.pi*x)))

    velocity = lambda x, y: vel(x,y)
    return Solver_2D(mesh_obj, 1, 1, "UPWIND", bc_list, source, velocity)    


def main():    
    solver = init_solver(True)
    solver.solve(True)
    solver.iterate()
    print(solver.residuel)
    print(np.sum(solver.areas))
    
    T_MMS = np.array([T_mms(x,y) for (x,y) in solver.centres])

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(solver.centres[:,0],solver.centres[:,1],solver.phi, c = solver.phi, cmap = "viridis")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Solution numérique")
    plt.show()
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(solver.centres[:,0],solver.centres[:,1],solver.phi, c = 'r', label = "numérique")
    ax.scatter(solver.centres[:,0],solver.centres[:,1],T_MMS, c = 'b', label = "analytique (MMS)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    main()

