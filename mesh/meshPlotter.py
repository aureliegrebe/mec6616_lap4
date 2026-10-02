# -*- coding: utf-8 -*-
"""
Exemple de classe pour la visualisation avec pyvista
MEC6616 - Aérodynamique numérique
Date de création: 2022 - 01 - 15
Auteur: El Hadji Abdou Aziz NDIAYE
"""

import numpy as np
import pyvista as pv
import pyvistaqt as pvQt
from mesh import Mesh


class MeshPlotter:
    """
    Permet d'afficher les maillages en utilisant pyvista.
    
    Vous pouvez utiliser/modifier/adapter cette classe dans vos codes.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    """

    def __init(self):
        return

    def plot_mesh(self, mesh_obj: Mesh, label_points=False, label_elements=False, label_faces=False):
        """
        Affiche le maillage en utilisant le module pyvista.

        Parameters
        ----------
        mesh_obj: Mesh
            Objet contenant les données du maillage.
        label_points: bool, optional
            Flag permettant de controler l'affichage des labels des points.
        label_elements: bool, optional
            Flag permettant de controler l'affichage des labels des éléments.
        label_faces: bool, optional
            Flag permettant de controler l'affichage des labels des faces.
            Il faut d'abord calculer la connectivité pour pouvoir afficher les faces.

        Returns
        -------
        None

        """
        nodes, elements = self.prepare_data_for_pyvista(mesh_obj)
        pv_mesh = pv.PolyData(nodes, elements)
        pl = pvQt.BackgroundPlotter()  # Allow the execution to continue
        pl.add_mesh(pv_mesh, show_edges=True, line_width=2, color="w")
        label_points_coords = nodes + [0, 0, 0.01]
        if label_points:
            pl.add_point_labels(label_points_coords, [f'N{i}' for i in range(mesh_obj.get_number_of_nodes())], text_color='k')
        if label_elements:
            pl.add_point_labels(pv_mesh.cell_centers(), [f'E{i}' for i in range(mesh_obj.get_number_of_elements())], text_color='b')
        if label_faces:
            face_to_nodes = mesh_obj.get_faces_to_nodes()
            label_faces_coords = 0.5 * (nodes[face_to_nodes[:, 0], :] + nodes[face_to_nodes[:, 1], :]) + [0, 0, 0.01]
            label_boundary_faces = [f'F{i}-Tag{mesh_obj.get_boundary_face_to_tag(i)}' for i in range(mesh_obj.get_number_of_boundary_faces())]
            label_internal_faces = [f'F{i}' for i in range(mesh_obj.get_number_of_boundary_faces(), mesh_obj.get_number_of_faces())]
            pl.add_point_labels(label_faces_coords, label_boundary_faces + label_internal_faces)
        pl.camera_position = 'xy'
        pl.background_color = 'white'
        pl.add_text('Maillage', position="upper_edge", color='k')
        pl.show()

    def prepare_data_for_pyvista(self, mesh_obj: Mesh):
        """
        Prépare les données du maillage pour affichage avec le module pyvista.

        Parameters
        ---------
        mesh_obj: Mesh
            Objet contenant les données du maillage.

        Returns
        -------
        nodes: ndarray
            Coordonnées des noeuds (compatible avec pyvista)
        elements: ndarray
            Connectivité élément -> noeuds (compatible avec pyvista)

        """
        nodes = np.array([mesh_obj.get_nodes_to_xcoord(), mesh_obj.get_nodes_to_ycoord(), np.zeros(mesh_obj.get_number_of_nodes())]).T
        elements = np.zeros((mesh_obj.get_element_to_nodes_start(-1) + mesh_obj.get_number_of_elements(),), dtype=int)
        index = 0
        for i_element in range(mesh_obj.get_number_of_elements()):
            nodes_elem = mesh_obj.get_element_to_nodes(i_element)
            n_nodes = nodes_elem.shape[0]
            elements[index] = n_nodes
            elements[index + 1:index + 1 + n_nodes] = nodes_elem
            index = index + 1 + n_nodes
        return nodes, elements
