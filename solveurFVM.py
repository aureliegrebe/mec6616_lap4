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
from solveur2d import Solver_2D

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

class Solver_FVM:

    def __init__(self, mesh: Mesh, 
                 density: float,
                 viscosity: float,
                 p_bc_list: List[BC] = [],
                 u_bc_list: List[BC] = [],
                 source_func: Optional[Callable[[float, float], [float, float]]] = None,
                 pressure_func: Optional[Callable[[float, float], float]] = None
                 ) -> None:
        # variables
        self.u = None # velocity field
        self.p = None # pressure (scalar) field
        self.source_func = source_func
        self.pressure_func = pressure_func
        self.density = density
        self.viscosity = viscosity


        self.set_mesh(mesh, p_bc_list, u_bc_list)


    def set_mesh(self, mesh: Mesh,
                 p_bc_list: List[BC] = [],
                 u_bc_list: List[BC] = []
                 ) -> None:
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
        self.conec = MeshConnectivity(mesh, verbose=False)
        self.conec.compute_connectivity()
        self.mesh = mesh

        self.calculate_centres()
        self.calculate_normals()

        if self.source_func is not None:
            self.set_source_field()
        else:
            self.source = None

        if self.pressure_func is not None:
            self.set_pressure_field()
        else:
            self.pressure = None

        if p_bc_list:
            if len(p_bc_list) != self.mesh.number_of_boundaries:
                raise ValueError("Incorrect number of boundaries")
            self.p_bc_list = p_bc_list
        else:
            self.p_bc_list = [BC(None, None) for _ in 
                range(self.mesh.number_of_boundaries)]

        if u_bc_list:
            if len(u_bc_list) != self.mesh.number_of_boundaries:
                raise ValueError("Incorrect number of boundaries")
            self.u_bc_list = u_bc_list
        else:
            self.u_bc_list = [BC(None, None) for _ in 
                range(self.mesh.number_of_boundaries)]

        self.gradp = None

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
                         func: Optional[Callable[[float, float],[float, float]]]=None
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
        # self.source = self.source_func(self.centres[:,0], self.centres[:,1])

    def set_pressure_field(self, 
                         func: Optional[Callable[[float, float], float]]=None
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
            self.pressure_func = func
        self.p = np.array([self.pressure_func(x, y) for (x, y) in self.centres])


    def get_gradp(self) -> np.ndarray:
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
        if self.gradp is None:
            self.calc_gradp()
        return self.gradp

    def calc_gradp(self) -> None:
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
                press_bc = self.p_bc_list[self.mesh.get_boundary_face_to_tag(idx_f)]
                edge_nodes = self.mesh.get_face_to_nodes(idx_f)
                xy_edge = (np.array(self.mesh.get_node_to_xycoord(edge_nodes[0])) +
                    np.array(self.mesh.get_node_to_xycoord(edge_nodes[1]))) / 2
                dxy = self.centres[idx_g] - xy_edge
                if press_bc.type in {BCType.DIRICHLET, BCType.DIRICHLET_FUNC}:
                    if press_bc.type is BCType.DIRICHLET_FUNC:
                        press_bc_val = press_bc.val(*self.right_centres[idx_f])
                    else:
                        press_bc_val = press_bc.val
                    scalar_diff = self.p[idx_g] - press_bc_val
                elif press_bc.type is BCType.NEUMANN:
                    # calculate edge normal
                    edge_vec = (np.array(self.mesh.get_node_to_xycoord(edge_nodes[1])) -
                        np.array(self.mesh.get_node_to_xycoord(edge_nodes[0])))
                    normal = np.array([edge_vec[1], -edge_vec[0]]) / np.linalg.norm(edge_vec) # ici le mauvais signe ?

                    # project centroid onto edge
                    dxy = np.dot(dxy, normal) * normal

                    # calc scalar diff
                    scalar_diff = np.dot(self.centres[idx_g] - xy_edge, normal) * press_bc.val
            else:
                dxy = self.centres[idx_g] - self.centres[idx_d]
                scalar_diff = self.p[idx_g] - self.p[idx_d]
            a_ls = np.outer(dxy, dxy)
            b[idx_g] += scalar_diff * dxy
            ata[idx_g] += a_ls
            if idx_d >= 0:
                ata[idx_d] += a_ls
                b[idx_d] += scalar_diff * dxy

        # solve each element
        gradp = np.zeros([self.mesh.number_of_elements, 2])
        for i in range(self.mesh.number_of_elements):
            gradp[i] = np.linalg.solve(ata[i], b[i])

        self.gradp = gradp

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
    mesh_obj = mesher.rectangle([-1,1,-1,1], mesh_parameters)

    p_func = lambda x, y: x * x + y

    u_bc_list = [
        BC(BCType.DIRICHLET, 0),
        BC(BCType.DIRICHLET, 0),
        BC(BCType.DIRICHLET, 0),
        BC(BCType.DIRICHLET, 0),
    ]

    p_bc_list = [
        BC(BCType.DIRICHLET_FUNC, p_func),
        BC(BCType.DIRICHLET_FUNC, p_func),
        BC(BCType.DIRICHLET_FUNC, p_func),
        BC(BCType.DIRICHLET_FUNC, p_func),
    ]

    solver = Solver_FVM(mesh_obj, 1, 1, p_bc_list, u_bc_list, pressure_func=p_func)

    plt.figure()
    plt.quiver(solver.centres[:,0], solver.centres[:,1],
                   solver.get_gradp()[:,0], solver.get_gradp()[:,1])
    plt.show()




