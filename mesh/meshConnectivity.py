# -*- coding: utf-8 -*-
"""
Module Maillage: calcul de la connectivité
MEC6616 - Aérodynamique numérique
Date de création: 2021 - 12 - 28
Auteur: El Hadji Abdou Aziz NDIAYE
"""

import numpy as np
from mesh import Mesh


class MeshConnectivity():
    """
    Calcule la connectivité du maillage.

    Parameters
    ----------
    mesh: Mesh
        Objet maillage. Les tables de connectivité finales seront sauvegardées dans cet objet.
    verbose: bool
        Paramètre pour le contrôle de l'affichage sur console.

    Attributes
    ----------
    _mesh: Mesh
        Objet Maillage
    _verbose: bool
        Contrôle de l'affichage sur console.
    _node_to_elements_start, node_to_elements: ndarray, ndarray
        Connectivité noeud -> éléments (utile pour batir les autres tables).
    _node_to_faces_start: ndarray
        Connectivité noeud -> faces (utile pour batir les autres tables).
    _face_to_nodes: ndarray
        Connectivité face -> noeuds.
    _number_of_faces: int
        Nombre de faces du maillage.
    _element_to_faces: ndarray
        Connectivité élément -> faces (utile pour batir les autres tables).
    __face_to_elements: ndarray
        Connectivité face -> éléments.
    _boundary_face_local_to_global, _boundary_face_global_to_local: ndarray, dict
        Tableau contenant les index initiaux des faces frontières.
        Les faces seront réarranger pour mettre les faces frontières au début des
        tables de connectivité.
    _internal_face_local_to_global: ndarray
        Tableau contenant les index initiaux des faces internes.
    _boundary_face_to_tag: ndarray
        Connectivité face frontière -> tag
    """

    def __init__(self, mesh: Mesh, verbose=True):
        self._mesh = mesh
        self._verbose = verbose

    def compute_connectivity(self):
        """
        Calcule les tables de connectivité.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        if self._verbose:
            print("Calcul de la connectivité:")
            print("\tTable noeud -> éléments ...")
        self.compute_node_to_elements()
        if self._verbose:
            print("\tTable face -> noeuds ...")
        self.compute_face_to_nodes()
        if self._verbose:
            print("\tTable élément -> faces ...")
        self.compute_element_to_faces()
        if self._verbose:
            print("\tTable face -> éléments ...")
        self.compute_face_to_elements()
        if self._verbose:
            print("\tTable boundary face -> tag ...")
        self.compute_boundary_face_to_tag()
        if self._verbose:
            print("\tSauvegarde des données de connectivité ...")
        self.save_connectivity()
        if self._verbose:
            print("Calcul de la connectivité terminé.")

    def compute_node_to_elements(self):
        """
        Calcule la connectivité noeud -> éléments.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self._node_to_elements_start = np.zeros((self._mesh.get_number_of_nodes() + 1,), dtype=int)
        # 1st pass
        for i_element in range(self._mesh.get_number_of_elements()):
            nodes = self._mesh.get_element_to_nodes(i_element)
            for node in nodes:
                self._node_to_elements_start[node + 1] += 1
        # Prefix
        for i_node in range(self._mesh.get_number_of_nodes()):
            self._node_to_elements_start[i_node + 1] += self._node_to_elements_start[i_node]
        # 2nd pass
        self._node_to_elements = np.zeros((self._node_to_elements_start[-1],), dtype=int)
        for i_element in range(self._mesh.get_number_of_elements()):
            nodes = self._mesh.get_element_to_nodes(i_element)
            for node in nodes:
                position_to_store = self._node_to_elements_start[node]
                self._node_to_elements[position_to_store] = i_element
                self._node_to_elements_start[node] = position_to_store + 1
        # Reshuffling
        for i_node in range(self._mesh.get_number_of_nodes(), 0, -1):
            self._node_to_elements_start[i_node] = self._node_to_elements_start[i_node - 1]
        self._node_to_elements_start[0] = 0

    def compute_face_to_nodes(self):
        """
        Calcule la connectivité face - > noeuds

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self._node_to_faces_start = np.zeros((self._mesh.get_number_of_nodes() + 1,), dtype=int)
        self._face_to_nodes = np.zeros((4 * self._mesh.get_number_of_elements(), 2), dtype=int)  # n_faces <= 4*n_elements for Quad and Tri
        n_faces = 0
        help_array = np.zeros((self._mesh.get_number_of_nodes(),), dtype=int) - 1  # allows to count each face once.
        for i_node in range(self._mesh.get_number_of_nodes()):
            start = self._node_to_elements_start[i_node]
            end = self._node_to_elements_start[i_node + 1]
            elements = self._node_to_elements[start:end]
            for element in elements:
                nodes = self._mesh.get_element_to_nodes(element)
                num_nodes = nodes.shape[0]
                index = np.where(nodes == i_node)[0][0]
                # Get 2 neighbors nodes
                index_nodes_to_check = [(index + 1) % num_nodes, (index + num_nodes - 1) % num_nodes]
                for node_index in index_nodes_to_check:
                    j_node = nodes[node_index]
                    if j_node > i_node and help_array[j_node] != i_node:
                        # create new face
                        self._face_to_nodes[n_faces, 0] = i_node
                        self._face_to_nodes[n_faces, 1] = j_node
                        n_faces += 1
                        help_array[j_node] = i_node
            self._node_to_faces_start[i_node + 1] = n_faces
        self._face_to_nodes = self._face_to_nodes[:n_faces, :]
        self._number_of_faces = n_faces

    def compute_element_to_faces(self):
        """
        Calcule la connectivité élément -> faces.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Observation: element_to_faces_start = element_to_nodes_start: 3 nodes (Tri) <-> 3 faces (Tri) same for Quad element
        self._element_to_faces = np.zeros((self._mesh.get_element_to_nodes_start(-1),), dtype=int)
        index = 0
        for i_element in range(self._mesh.get_number_of_elements()):
            nodes = self._mesh.get_element_to_nodes(i_element)
            for i_node in range(nodes.shape[0]):
                j_node = (i_node + 1) % nodes.shape[0]
                node_1 = min(nodes[i_node], nodes[j_node])
                node_2 = max(nodes[i_node], nodes[j_node])
                start = self._node_to_faces_start[node_1]
                end = self._node_to_faces_start[node_1 + 1]
                for face in range(start, end):
                    if self._face_to_nodes[face, 1] == node_2:
                        self._element_to_faces[index] = face
                        index += 1
                        break

    def compute_face_to_elements(self):
        """
        Calcule la connectivité face -> éléments.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self._face_to_elements = np.zeros((self._number_of_faces, 2), dtype=int)
        self._face_to_elements[:, 1] = -1
        index = np.zeros((self._number_of_faces, ), dtype='uint8')
        start = self._mesh.get_element_to_nodes_start(0)
        for i_element in range(self._mesh.get_number_of_elements()):
            end = self._mesh.get_element_to_nodes_start(i_element + 1)
            faces = self._element_to_faces[start:end]
            for face in faces:
                self._face_to_elements[face, index[face]] = i_element
                index[face] += 1
            start = end
        # face to elements orientation: elem 1 -> left, elem 2 -> right
        for i_face in range(self._number_of_faces):
            if self._face_to_elements[i_face, 1] > 0:  # !=-1
                face_nodes = self._face_to_nodes[i_face, :]
                elem_1 = self._face_to_elements[i_face, 0]
                elem_nodes = self._mesh.get_element_to_nodes(elem_1)
                for (i_node1, node1) in enumerate(elem_nodes):
                    if node1 == face_nodes[0]:
                        i_node2 = (i_node1 + 1) % elem_nodes.shape[0]
                        if elem_nodes[i_node2] != face_nodes[1]:
                            self._face_to_elements[i_face, 0] = self._face_to_elements[i_face, 1]
                            self._face_to_elements[i_face, 1] = elem_1
                        break
        # Local boundary face -> to global face
        self._boundary_face_local_to_global = np.where(index == 1)[0]
        self._internal_face_local_to_global = np.where(index == 2)[0]
        # Global face -> to local boundary face
        self._boundary_face_global_to_local = dict()
        for i_bound_face in range(self._mesh.get_number_of_boundary_faces()):
            bound_face_global = self._boundary_face_local_to_global[i_bound_face]
            self._boundary_face_global_to_local[bound_face_global] = i_bound_face

    def compute_boundary_face_to_tag(self):
        """
        Calcule la connectivité boundary face -> tag.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self._boundary_face_to_tag = np.zeros((self._mesh.get_number_of_boundary_faces(),), dtype='uint8')
        for i_bound in range(self._mesh.get_number_of_boundaries()):
            boundary_faces_nodes = self._mesh.get_boundary_faces_nodes(i_bound)
            n_faces = boundary_faces_nodes.shape[0] // 2
            for i_face in range(n_faces):
                node_a = boundary_faces_nodes[2 * i_face]
                node_b = boundary_faces_nodes[2 * i_face + 1]
                node_1 = min(node_a, node_b)
                node_2 = max(node_a, node_b)
                start = self._node_to_faces_start[node_1]
                end = self._node_to_faces_start[node_1 + 1]
                for face in range(start, end):
                    if self._face_to_nodes[face, 1] == node_2:
                        bound_face_local = self._boundary_face_global_to_local[face]
                        self._boundary_face_to_tag[bound_face_local] = i_bound
                        # Orientation
                        element_left = self._face_to_elements[face, 0]
                        elem_nodes = self._mesh.get_element_to_nodes(element_left)
                        for (i_node1, node1) in enumerate(elem_nodes):
                            if node1 == self._face_to_nodes[face, 0]:
                                i_node2 = (i_node1 + 1) % elem_nodes.shape[0]
                                if elem_nodes[i_node2] != self._face_to_nodes[face, 1]:
                                    # Swap the nodes
                                    temp = self._face_to_nodes[face, 0]
                                    self._face_to_nodes[face, 0] = self._face_to_nodes[face, 1]
                                    self._face_to_nodes[face, 1] = temp
                                break
                        break

    def save_connectivity(self):
        """
        Sauvegarde les tables de connectivité.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Number of faces
        self._mesh.set_number_of_faces(self._number_of_faces)
        # Face to nodes
        face_to_nodes = np.zeros((self._number_of_faces, 2), dtype=int)
        face_to_nodes[:self._mesh.get_number_of_boundary_faces(), :] = self._face_to_nodes[self._boundary_face_local_to_global, :]
        face_to_nodes[self._mesh.get_number_of_boundary_faces():, :] = self._face_to_nodes[self._internal_face_local_to_global, :]
        self._mesh.set_face_to_nodes(face_to_nodes)
        # Face to elements
        face_to_elements = np.zeros((self._mesh.number_of_faces, 2), dtype=int)
        face_to_elements[:self._mesh.get_number_of_boundary_faces(), :] = self._face_to_elements[self._boundary_face_local_to_global, :]
        face_to_elements[self._mesh.get_number_of_boundary_faces():, :] = self._face_to_elements[self._internal_face_local_to_global, :]
        self._mesh.set_face_to_elements(face_to_elements)
        # Boundary face to tag
        self._mesh.set_boundary_face_to_tag(self._boundary_face_to_tag)
