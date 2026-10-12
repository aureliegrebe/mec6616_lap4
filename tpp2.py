from cProfile import label
import numpy as np
from mesh.meshGenerator import MeshGenerator
from mesh.meshPlotter import MeshPlotter
import sympy as sp


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

# Constantes
PHI_0 = 1.
PHI_L = 0.
L_EX = 1.0 # m
GAMMA = 0.1 # kg/(ms)
RHO_U_I = 2.1 # kg/(m²s)
RHO_U_II = 2.5 # kg/(m²s)
sol_analytique_5_1i = lambda x: PHI_0 + (np.exp(RHO_U_I * x / GAMMA) - 1) / \
    (np.exp(RHO_U_I * L_EX / GAMMA) - 1) * (PHI_L - PHI_0)

def velocity (x,y):
    return ((2*x**2 - x**4 - 1) * (y - y**3), -(2*y**2 - y**4 -1)*(x - x**3))

def T_mms(x,y):
    return (400 + 50*np.cos(np.pi*x) + 100*np.sin(np.pi*x*y))

def source(x, y, k):
    return np.pi * (50 * np.pi * k * (2 * x * x * np.sin(np.pi*x*y) +
        2*y*y*np.sin(np.pi*x*y) + np.cos(np.pi*x)) - 100.0*x*x*(x*x-
            1)*(np.pow(y, 4) - 2 * np.pow(y,2) + 1)*np.cos(np.pi*x*y) + 50.0*y*(y*y -
            1)*(2*y*np.cos(np.pi*x*y) - np.sin(np.pi*x))*(np.pow(x, 4) - 2*x*x + 1))

def sym_source():
    xs, ys, ks = sp.symbols('x y k')
    rho = 1.0

    u, v = velocity(xs, ys)

    T_expr = (
        400
        + 50*sp.cos(sp.pi*xs)
        + 100*sp.sin(sp.pi*xs*ys)
    )

    dT_dx = sp.diff(T_expr, xs)
    dT_dy = sp.diff(T_expr, ys)
    d2T_dx2 = sp.diff(T_expr, xs, 2)
    d2T_dy2 = sp.diff(T_expr, ys, 2)
    
    S_expr = sp.simplify(rho * (u * dT_dx + v * dT_dy) - k * (d2T_dx2 + d2T_dy2))
    S_expr_2 = sp.simplify(rho * (u * dT_dx + v * dT_dy) - ks * (d2T_dx2 + d2T_dy2))
    print(S_expr_2)

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
    
    source_func = lambda x,y: source(x,y,therm_cond)
    
    return Solver_2D(mesh_obj, therm_cond, density, method, bc_list, source_func, velocity)    

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

def convergence(solver: Solver_2D, quad=False, cross_diff=False, upwind=True, therm_cond = THERM_COND):
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
    source_func = lambda x,y: source(x,y,therm_cond)
    solver.set_source_field(source_func)
    
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
        err = solver.phi - exact_sol
        rms = np.sqrt(np.average(err**2))
        epsilon2.append(rms)
        h_list.append(solver.get_h())

        if i == 1:
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

def ex5_1i(quad=True, cross_diff=False, method="UPWIND"):
    mesher = MeshGenerator()
    solver = init_solver(quad, GAMMA, 1, method)
    # param = Settings()
    # param.diffusivity = GAMMA
    # param.density = 1.0
    # param.u = RHO_U_I
    # param.left_BC = BC("DIRICHLET", PHI_0)
    # param.right_BC = BC("DIRICHLET", PHI_L)
    # param.length = L
    # param.conv_scheme = "CENTRAL"
    vel_field = lambda x, y: (RHO_U_I, 0)
    solver.set_velocity_field(vel_field)

    if quad:
        mesh_parameters = {'mesh_type': 'QUAD',
                   'Nx': Nx, 'Ny': Ny                  
                   }
    else:
        mesh_parameters = {'mesh_type': 'TRI',
                   'lc': L_EX/N
                   }
    # Conditions aux limites
    bc_list = [
        BC(BCType.DIRICHLET, PHI_0),
        BC(BCType.NEUMANN, 0),
        BC(BCType.DIRICHLET, PHI_L),
        BC(BCType.NEUMANN, 0),
    ]
    
    solver.set_source_field(lambda x, y: 0)
    solver.set_mesh(mesher.rectangle([0,L_EX,0,L_EX], mesh_parameters), bc_list)

    if cross_diff:
        solver.iterate()
    else:
        solver.solve(first_iter=True)

    # Tracé des résultats par VFM et la solution analytique
    

    exact_sol = np.array(sol_analytique_5_1i(solver.centres[:,0]))
    fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(solver.centres[:,0], solver.centres[:,1], exact_sol, label="analytique")
    # ax.scatter(solver.centres[:,0], solver.centres[:,1], solver.phi, label="numérique")
    # plt.plot(solver.centres[:,0], exact_sol, '-')
    # plt.quiver(solver.centres[:,0], solver.centres[:,1],
    #            solver.phi()[:,0], solver.get_grad()[:,1])
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.legend()
    x = np.linspace(0, L_EX, 200)
    plt.plot(x, sol_analytique_5_1i(x), label="Solution analytique")
    # for n in N_LIST_PLOT:
    #     param.n = n
    #     X, T = solve(param)
    #     plt.plot(X, T, label=f"n = {n}")
    
    h_list = []
    epsilon2 = []
    # Calcul de l'erreur en fonction du nombre de points
    for (i, n) in enumerate(N_LIST):
        if quad:
            mesh_parameters["Nx"] = n
            mesh_parameters["Ny"] = n
        else:
            mesh_parameters["lc"] = L_EX/n
        solver.set_mesh(mesher.rectangle([0,L_EX,0,L_EX], mesh_parameters), bc_list)
        exact_sol = np.array(sol_analytique_5_1i(solver.centres[:,0]))
        x_slice = (solver.centres[:,1] > 0.49 * L_EX) & (solver.centres[:,1] < 0.51 * L_EX)
        if cross_diff:
            solver.iterate()
        else:
            solver.solve(first_iter=True)
        plt.plot(solver.centres[x_slice,0], solver.phi[x_slice], ".", label=f"n = {n}")
        # err = np.linalg.norm(solver.phi - exact_sol, axis=0)
        err = solver.phi - exact_sol
        rms = np.sqrt(np.average(err**2))
        # print(rms)
        epsilon2.append(rms)
        h_list.append(solver.get_h())
        # h_list.append(1/n)

    plt.xlabel("x (m)")
    plt.ylabel("Φ")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figures/ex5_1i.pdf")
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
    plt.savefig("./figures/epsilon5_1i.pdf")

    p = float(np.log(epsilon2[-1]/epsilon2[-2])/np.log(h_list[-1]/h_list[-2]))
    if method == "UPWIND":
        print(f"Ordre de convergence (Upwind) : {p:.5f}")
    else:
        print(f"Ordre de convergence (centré) : {p:.5f}")


def main():    
    ex5_1i()
    ex5_1i(method="CENTRE")
    upwind_solver = init_solver(True, 1, DENSITY, "UPWIND")
    convergence(upwind_solver,quad=True, cross_diff=False, upwind=True)
    centre_solver = init_solver(True, 1, DENSITY, "CENTRE")
    convergence(centre_solver,quad=True, cross_diff=False, upwind=False)
    # for k in [1,10,100, 1000]:
    #     upwind_solver = init_solver(True, THERM_COND, DENSITY, "UPWIND")
    #     convergence(upwind_solver,quad=True, cross_diff=False, upwind=True)
    #     centre_solver = init_solver(False, THERM_COND, DENSITY, "CENTRE")
    #     convergence(centre_solver,quad=True, cross_diff=False, upwind=False)
    plt.show()
    
if __name__ == '__main__':
    main()

