from enum import Enum
import numpy as np
from typing import Optional, List
from collections.abc import Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pyvista as pv

from mesh.mesh import Mesh
from mesh.meshConnectivity import MeshConnectivity
from mesh.meshGenerator import MeshGenerator
from mesh.meshPlotter import MeshPlotter

# TODO: Add source term

class BCType(Enum):
    DIRICHLET = 1
    NEUMANN = 2
    DIRICHLET_FUNC = 3
    NEUMANN_FUNC = 4

@dataclass
class BC():
    type: Optional[BCType] = None
    val: Optional[float | Callable] = None

class Solver_2D:
    """
    Solveur simple en 2D pour calculer divergences + gradients

    Paramètres
    ----------
    mesh: Mesh
        Maillage à utiliser dans le solver.
    scalar_func: Callable[[float, float], float]
        fonction f(x, y) qui défini le champ scalaire utilisé dans le calcul du
        gradient.
    therm_cond: float
        Conductivité thermique
    bc_list: List[BC]
        List de conditions aux frontières, dans le même ordre que les tags de
        frontières du maillage.
    source_func: Callable[[float, float], float]
        fonction définissant une terme source en fonction de (x, y) d'unités
        W/m^2.

    Attributs
    ---------
    mesh: Mesh
        Maillage
    conec: MeshConnectivity
        La connectivité du maillage
    bc_list: List[BC]
        List de conditions aux frontières, dans le même ordre que les tags de
        frontières du maillage.
    centres: ndarray
        Coordonnées pour les centres de chaque cellule
    grad: ndarray
        Gradient évalué à chaque centre de cellule
    div: ndarray
        Divergence évalué à chaque centre de cellule
    """
    def __init__(self, mesh: Mesh,
                 therm_cond: float,
                 density: float,
                 schema: str,
                 bc_list: List[BC] = [],
                 source_func: Optional[Callable[[float, float], float]] = None,
                 velocity_func: Optional[Callable[[float, float], float]] = None
                 ) -> None:
        self.therm_cond = therm_cond
        self.density = density
        self.schema = schema
        self.source_func = source_func
        self.velocity_func = velocity_func
        self.residuel = []
        self.plotter = None


        self.set_mesh(mesh, bc_list)

    def set_mesh(self, mesh: Mesh, bc_list: List[BC] = []) -> None:
        """
        Défini le maillage. Reset tout les attributs qui dépends du maillage.

        Paramètres
        ----------
        mesh: Mesh
            Maillage à utiliser dans le solver.
        bc_list: List[BC]
            List de conditions aux frontières, dans le même ordre que les tags de
            frontières du maillage.

        Retourne
        --------
        None
        """
        self.conec = MeshConnectivity(mesh)
        self.conec.compute_connectivity()
        self.mesh = mesh

        self.calculate_centres()
        self.calculate_normals()

        if self.source_func is not None:
            self.set_source_field()
        else:
            self.source = None

        if self.velocity_func is not None:
             self.set_velocity_field()
        else:
            self.velocity = None

        if bc_list:
            if len(bc_list) != self.mesh.number_of_boundaries:
                raise ValueError("Incorrect number of boundaries")
            self.bc_list = bc_list
        else:
            self.bc_list = [BC(None, None) for _ in 
                range(self.mesh.number_of_boundaries)]

        self.temp = None
        nb_elements = self.mesh.get_number_of_elements()
        self.A = np.zeros((nb_elements,nb_elements))
        self.b = np.zeros(nb_elements)

    def calculate_areas(self):
        """
        Calcule l'aire de chaque cellule utilisant la méthode des lacets

        Inspiré de https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
        """
        self.areas = np.zeros(self.mesh.number_of_elements)
        for i in range(self.mesh.number_of_elements):
            nodes = self.mesh.get_element_to_nodes(i)
            xy_nodes = np.array([self.mesh.get_node_to_xycoord(node) for node
                in nodes])
            x = xy_nodes[:,0]
            y = xy_nodes[:,1]
            self.areas[i] = 0.5 * np.abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x, 1)))

    def calculate_normals(self):
        """
        Calcule des vecteurs normaux de chaque face
        """
        self.faces_start = np.array([self.mesh.get_node_to_xycoord(idx) for idx, _ in
            self.mesh.face_to_nodes])
        self.faces_end = np.array([self.mesh.get_node_to_xycoord(idx) for _, idx in
            self.mesh.face_to_nodes])
        self.face_vecs = self.faces_end - self.faces_start
        self.face_size = np.linalg.norm(self.face_vecs, axis=1)
        self.normal = np.column_stack([self.face_vecs[:,1] / self.face_size,
                                      -self.face_vecs[:,0] / self.face_size])
        self.left_centres = self.centres[self.mesh.face_to_elements[:,0]]
        self.right_centres = self.centres[self.mesh.face_to_elements[:,1]]

        nbf = self.mesh.number_of_boundary_faces
        self.right_centres[:nbf] = self.faces_start[:nbf] + 0.5 * self.face_vecs[:nbf]

        ksi = self.right_centres - self.left_centres
        self.d_ksi = np.linalg.norm(ksi, axis=1)
        self.e_ksi = np.transpose(ksi.T / self.d_ksi)
        self.e_eta = np.transpose(self.face_vecs.T / self.face_size)

        self.p_nksi = np.vecdot(self.normal, self.e_ksi, axis=1)
        self.p_ksieta = np.vecdot(self.e_ksi, self.e_eta, axis=1)

    def set_source(self, B):
        if self.source is None:
            return
        B += self.source * self.areas
        

    def set_diff(self, B, first_iter: bool) -> None:
        diff_cnsts = self.therm_cond/self.p_nksi*self.face_size/self.d_ksi
        if first_iter:
            grad_phi = np.zeros([self.mesh.number_of_faces, 2])
        else:
            left_grad_phi = self.get_grad()[self.mesh.face_to_elements[:,0]]
            right_grad_phi = self.get_grad()[self.mesh.face_to_elements[:,1]]
            grad_phi = 0.5 * (left_grad_phi + right_grad_phi)
            nbf = self.mesh.number_of_boundary_faces
            grad_phi[:nbf] = left_grad_phi[:nbf]

        cross_diff = -self.therm_cond * self.p_ksieta/self.p_nksi*np.vecdot(grad_phi,self.e_eta, axis=1)*self.face_size

        # Parcours du maillage
        for (idx_f, (idx_g, idx_d)) in enumerate(self.mesh.face_to_elements):
            # Coefficients pour les cellules aux frontières
            if idx_f < self.mesh.number_of_boundary_faces:
                bc = self.bc_list[self.mesh.get_boundary_face_to_tag(idx_f)]
                if bc.type in {BCType.DIRICHLET, BCType.DIRICHLET_FUNC}:
                    if bc.type is BCType.DIRICHLET_FUNC:
                        bc_val = bc.val(*self.right_centres[idx_f])
                    else:
                        bc_val = bc.val
                    B[idx_g] += bc_val*diff_cnsts[idx_f]
                    if first_iter: # Si on est à la première itération, on caclule A
                        self.A[idx_g,idx_g] += diff_cnsts[idx_f]
                    B[idx_g] += cross_diff[idx_f]
                elif bc.type is BCType.NEUMANN:
                    B[idx_g] += self.therm_cond * self.face_size[idx_f] * bc.val
            # Coefficients pour les cellules intérieurs    
            else:
                if first_iter: # Si on est à la première itération, on caclule A et pose grad(phi) = 0
                    self.A[idx_g,idx_g] += diff_cnsts[idx_f]
                    self.A[idx_d,idx_d] += diff_cnsts[idx_f]
                    self.A[idx_g,idx_d] += -diff_cnsts[idx_f]
                    self.A[idx_d,idx_g] += -diff_cnsts[idx_f]
                B[idx_g] += cross_diff[idx_f]
                B[idx_d] += -cross_diff[idx_f]
    
    def set_conv(self, B, first_iter:bool) -> None:
        conv_cnsts = self.density * np.sum(self.normal * self.velocity, axis=1) * self.face_size
        for (idx_f, (idx_g, idx_d)) in enumerate(self.mesh.face_to_elements):
            # Coefficients pour les cellules aux frontières
            if idx_f < self.mesh.number_of_boundary_faces:
                bc = self.bc_list[self.mesh.get_boundary_face_to_tag(idx_f)]
                if self.schema == "UPWIND":
                    if bc.type in {BCType.DIRICHLET, BCType.DIRICHLET_FUNC}:
                        if bc.type is BCType.DIRICHLET_FUNC:
                            bc_val = bc.val(*self.right_centres[idx_f])
                        else:
                            bc_val = bc.val
                        B[idx_g] += bc_val * max(-conv_cnsts[idx_f],0)
                        if first_iter: # Si on est à la première itération, on caclule A
                            self.A[idx_g,idx_g] += max(conv_cnsts[idx_f],0)
                    elif bc.type is BCType.NEUMANN:
                        B[idx_g] += conv_cnsts[idx_f] * bc.val * self.p_nksi[idx_f] * self.d_ksi[idx_f] # TODO: à checker
                        if first_iter: # Si on est à la première itération, on caclule A
                            self.A[idx_g,idx_g] += max(conv_cnsts[idx_f],0) # TODO: ou juste F_i ??
                elif self.schema == "CENTRE": # TODO: retrouver les coeffs pour centre en 2D
                    if bc.type in {BCType.DIRICHLET, BCType.DIRICHLET_FUNC}:
                        B[idx_g] += 0
                        if first_iter: # Si on est à la première itération, on caclule A
                            self.A[idx_g,idx_g] += 0
                    elif bc.type is BCType.NEUMANN:
                        B[idx_g] += 0
                        if first_iter: # Si on est à la première itération, on caclule A
                            self.A[idx_g,idx_g] += 0
            # Coefficients pour les cellules intérieurs    
            else:
                if first_iter: # Si on est à la première itération, on caclule A
                    if self.schema == "CENTRE":
                        self.A[idx_g,idx_g] += conv_cnsts[idx_f] / 2
                        self.A[idx_d,idx_d] += - conv_cnsts[idx_f] / 2
                        self.A[idx_g,idx_d] += conv_cnsts[idx_f] / 2
                        self.A[idx_d,idx_g] += -conv_cnsts[idx_f] / 2
                    elif self.schema == "UPWIND":
                        self.A[idx_g,idx_g] += max(conv_cnsts[idx_f],0)
                        self.A[idx_d,idx_d] += max(-conv_cnsts[idx_f],0)
                        self.A[idx_g,idx_d] += - max(-conv_cnsts[idx_f],0)
                        self.A[idx_d,idx_g] += - max(conv_cnsts[idx_f],0)

    def solve(self, first_iter: bool):
        """
        Résoud l'équation de diffusion sur tout le maillage.
        
        Paramètres
        ----------
        first_iter: bool
            Indique s'il s'agit de la première itération ou non
        """
        # self.calculate_centres()
        nb_elem = self.mesh.number_of_elements
        B = np.zeros(nb_elem) # On ré-initialise B à chaque itération
        
        if first_iter:
            self.A = np.zeros((nb_elem, nb_elem))
    
        self.set_diff(B, first_iter) # On résoud les termes de diffusion
        self.set_conv(B, first_iter) # On résoud les termess de convexion
        self.set_source(B) # On ajoute les termes de source à B

        phi = np.linalg.solve(self.A, B) # On résoud l'équation encore une fois en utilisant le nouveau B
        self.b_old = self.b # On stocke le B de l'itération précédente
        self.b = B # On définit le B de cette itération
        self.phi = phi # On met à jour le champ
        self.grad = None # force recalcul du gradient
    
    def iterate(self, n=5) -> None:
        """
        Itère la solution de l'équation de diffusion pendant n itérations et crée une liste de résiduels.
        
        Paramètres
        ----------
        n: int
            Nombre d'itérations à effectuer

        Retourne
        --------
        None
        """
        for j in range(n):
            self.solve(j==0)
            if j == 0:
                self.residuel = []
            else:
                self.residuel.append(np.sqrt(np.mean((self.b - self.b_old)**2)))

    def get_phi(self) -> np.ndarray:
        if self.phi is None:
            self.solve()
        return self.phi


    def calculate_centres(self) -> None:
        """
        Calcule les centres de chaque cellule

        Paramètres
        ----------
        None

        Retourne
        --------
        None
        """
        self.calculate_areas()
        self.centres = np.zeros([self.mesh.number_of_elements, 2])
        for i in range(self.mesh.number_of_elements):
            nodes = self.mesh.get_element_to_nodes(i)
            xy_nodes = np.array([self.mesh.get_node_to_xycoord(node) for node
                in nodes])
            x = xy_nodes[:,0]
            y = xy_nodes[:,1]
            c_x = 0
            c_y = 0
            n = len(x)
            for j in range(n):
                c_x += (x[j] + x[(j+1) % n]) * (x[j] * y[(j+1) % n] - x[(j+1) % n] * y[j])
                c_y += (y[j] + y[(j+1) % n]) * (x[j] * y[(j+1) % n] - x[(j+1) % n] * y[j])
            self.centres[i, 0] = c_x / self.areas[i] / 6.
            self.centres[i, 1] = c_y / self.areas[i] / 6.

    
    def set_source_field(self, 
                         func: Optional[Callable[[float, float],float]]=None
                         ) -> None:
        """
        Définit le champ scalaire

        Paramètres
        ----------
        scalar_func: Callable[[float, float], float]
            fonction f(x, y) qui défini le champ scalaire utilisé comme terme source

        Retourne
        --------
        None
        """
        if func is not None:
            self.source_func = func
        self.source = np.array([self.source_func(x, y) for (x, y) in self.centres])

        self.grad = None # Pourquoi le mettre ici ? C'est déja dans solve

    def set_velocity_field(self, 
                      func: Optional[Callable[[float, float],(float, float)]] = None
                      ) -> None:
        """
        Définit le champ scalaire

        Paramètres
        ----------
        scalar_func: Callable[[float, float], float]
            fonction f(x, y) qui défini le champ scalaire utilisé comme champ de vitesses

        Retourne
        --------
        None
        """
        if func is not None:
            self.velocity_func = func
        centres_faces = (self.faces_start + self.faces_end) / 2
        self.velocity = np.array([self.velocity_func(x, y) for (x, y) in centres_faces]) 

        # Reset gradient as it is no longer valid
        self.grad = None

    def get_div(self) -> np.ndarray:
        """
        Retourne le divergence à chaque centre de cellule

        Le divergence est recalculé uniquement si ce n'est pas déjà à jour

        Paramètres
        ----------
        None

        Retourne
        --------
        divergence: ndarray
        """
        if self.div is None:
            self.calc_div()
        return self.div

    def calc_div(self) -> None:
        """
        Calcul le divergence à chaque centre de cellule

        Paramètres
        ----------
        None

        Retourne
        --------
        None
        """
        nb_elements = self.mesh.get_number_of_elements()
        nb_faces = self.mesh.get_number_of_faces()
        div = np.zeros(nb_elements)
        
        # Parcous des faces
        for i in range (0, nb_faces):
            Eg = self.mesh.get_face_to_elements(i)[0] 
            Ed = self.mesh.get_face_to_elements(i)[1]
            nodes = self.mesh.get_face_to_nodes(i)
            x1, y1 = self.mesh.get_node_to_xycoord(nodes[0])
            x2, y2 = self.mesh.get_node_to_xycoord(nodes[1])
            normale = np.array([ - (y2 - y1), x2 - x1]) # non-normalisée donc déja *ds
            
            # Calcul du flux
            if Ed >= 0:
                flux = np.dot((self.vec[Eg] + self.vec[Ed]) / 2, normale) 
            else:
                flux = np.dot(self.vec[Eg], normale)
            
            # Modification du div pour les éléments à gauche et droite de la face
            div[Eg] = div[Eg] + flux # /self.areas[Eg]
            if Ed != -1:
                div[Ed] = div[Ed] - flux # /self.areas[Ed]
        self.div = div

    def get_grad(self) -> np.ndarray:
        """
        Retourne le gradient à chaque centre de cellule

        Le gradient est recalculé uniquement si ce n'est pas déjà à jour

        Paramètres
        ----------
        None

        Retourne
        --------
        gradient: ndarray
        """
        if self.grad is None:
            self.calc_grad()
        return self.grad

    def calc_grad(self) -> None:
        """
        Calcul le gradient à chaque centre de cellule

        Paramètres
        ----------
        None

        Retourne
        --------
        None
        """
        ata = np.zeros([self.mesh.number_of_elements, 2, 2])
        b = np.zeros([self.mesh.number_of_elements, 2])
        for (idx_f, (idx_g, idx_d)) in enumerate(self.mesh.face_to_elements):
            # deal with boundary faces:
            if idx_f < self.mesh.number_of_boundary_faces:
                bc = self.bc_list[self.mesh.get_boundary_face_to_tag(idx_f)]
                edge_nodes = self.mesh.get_face_to_nodes(idx_f)
                xy_edge = (np.array(self.mesh.get_node_to_xycoord(edge_nodes[0])) +
                    np.array(self.mesh.get_node_to_xycoord(edge_nodes[1]))) / 2
                dxy = self.centres[idx_g] - xy_edge
                if bc.type in {BCType.DIRICHLET, BCType.DIRICHLET_FUNC}:
                    if bc.type is BCType.DIRICHLET_FUNC:
                        bc_val = bc.val(*self.right_centres[idx_f])
                    else:
                        bc_val = bc.val
                    scalar_diff = self.phi[idx_g] - bc_val
                elif bc.type is BCType.NEUMANN:
                    # calculate edge normal
                    edge_vec = (np.array(self.mesh.get_node_to_xycoord(edge_nodes[1])) -
                        np.array(self.mesh.get_node_to_xycoord(edge_nodes[0])))
                    normal = np.array([edge_vec[1], -edge_vec[0]]) / np.linalg.norm(edge_vec) # ici le mauvais signe ?

                    # project centroid onto edge
                    dxy = np.dot(dxy, normal) * normal

                    # calc scalar diff
                    scalar_diff = np.dot(self.centres[idx_g] - xy_edge, normal) * bc.val
            else:
                dxy = self.centres[idx_g] - self.centres[idx_d]
                scalar_diff = self.phi[idx_g] - self.phi[idx_d]
            a_ls = np.outer(dxy, dxy)
            b[idx_g] += scalar_diff * dxy
            ata[idx_g] += a_ls
            if idx_d >= 0:
                ata[idx_d] += a_ls
                b[idx_d] += scalar_diff * dxy

        # solve each element
        grad = np.zeros([self.mesh.number_of_elements, 2])
        for i in range(self.mesh.number_of_elements):
            grad[i] = np.linalg.solve(ata[i], b[i])

        self.grad = grad

    def euler_check(self, nb_holes: int) -> bool:
        """
        Valide le maillage à l'aide de l'égalité d'euler

        Paramètres
        ----------
        nb_holes: int
            Nombre de trous prévues dans le maillage

        Retourne
        --------
        valid: bool
        """
        f = self.mesh.get_number_of_elements()
        a = self.mesh.get_number_of_faces()
        s = self.mesh.get_number_of_nodes()
        h = nb_holes
        return(f - a + s == 1 - h)

    def to_vtk(self, filename: str) -> None:
        """
        Exporte le maillage avec les résultats en vtk

        Paramètres
        ----------
        filename: str
            Nom du fichier à sauvegarder

        Retourne
        --------
        None
        """
        if self.plotter is None:
            self.plotter = MeshPlotter()
        nodes, elements = self.plotter.prepare_data_for_pyvista(self.mesh)
        pv_mesh = pv.PolyData(nodes, elements)
        pv_mesh.cell_data['phi'] = self.phi
        writer = pv.PolyDataWriter(filename, pv_mesh)
        writer.write()

    def get_h(self) -> float:
        """
        Retourne le paramètre h pour calculer la convergence
        """
        return np.sqrt(np.average(self.areas))
    
if __name__ == "__main__":
    mesher = MeshGenerator()
    
    # Générer un mesh 
    mesh_parameters = {'mesh_type': 'QUAD',
                   'Nx': 60, 'Ny':60
                   }
    # mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
    mesh_obj = mesher.rectangle([-1,1,-1,1], mesh_parameters)
    
    def vel (x,y):
        return ((2*x**2 - x**4 - 1) * (y - y**3), -(2*y**2 - y**4 -1)*(x - x**3))
    
    def T_mms(x,y):
        return (400 + 50*np.cos(np.pi*x) + 100*np.sin(np.pi*x*y))
    
    bc_list = [
        BC(BCType.DIRICHLET_FUNC, lambda x, y: T_mms(-1,y)),
        BC(BCType.DIRICHLET_FUNC, lambda x, y: T_mms(x,-1)),
        BC(BCType.DIRICHLET_FUNC, lambda x, y: T_mms(1,y)),
        BC(BCType.DIRICHLET_FUNC, lambda x, y: T_mms(x,1)),
    ]
    # bc_list = [
    #     BC(BCType.DIRICHLET,0),
    #     BC(BCType.DIRICHLET,0),
    #     BC(BCType.DIRICHLET,0),
    #     BC(BCType.DIRICHLET,0),
    # ]
    
    # Depuis source.py
    source = lambda x, y: np.pi*(-100.0*x**2*(x**2 - 1)*(y**4 - 2*y**2 + 1)*np.cos(np.pi*x*y) 
                                 + 100.0*np.pi*x**2*np.sin(np.pi*x*y) + 50.0*y*(y**2 - 1)*
                                 (2*y*np.cos(np.pi*x*y) - np.sin(np.pi*x))*(x**4 - 2*x**2 + 1) 
                                 + np.pi*(100.0*y**2*np.sin(np.pi*x*y) + 50.0*np.cos(np.pi*x)))

    velocity = lambda x, y: vel(x,y)
    solver = Solver_2D(mesh_obj, 1, 1, "UPWIND", bc_list, source, velocity)    
    solver.solve(True)
    solver.iterate()
    print(solver.residuel)
    print(np.sum(solver.areas))
    T_MMS = np.array([T_mms(x,y) for (x,y) in solver.centres])
    #plt.figure()
    #plt.plot(solver.centres[:,0],solver.phi,".")
    #plt.plot(solver.centres[:,1],T_MMS)
    
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # plot = ax.scatter(solver.centres[:,0],solver.centres[:,1],T_MMS, c =T_MMS, cmap = "viridis")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("Solution analytique (MMS)")
    # plt.show()
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plot = ax.scatter(solver.centres[:,0],solver.centres[:,1],solver.phi, c = solver.phi, cmap = "viridis")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Solution numérique")
    plt.show()
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plot = ax.scatter(solver.centres[:,0],solver.centres[:,1],solver.phi, c = 'r', label = "numérique")
    plot = ax.scatter(solver.centres[:,0],solver.centres[:,1],T_MMS, c = 'b', label = "analytique (MMS)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

