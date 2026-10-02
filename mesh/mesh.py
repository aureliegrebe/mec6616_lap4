# -*- coding: utf-8 -*-
"""
Module Maillage: stockage des données
MEC6616 - Aérodynamique numérique
Date de création: 2021 - 12 - 22
Auteur: El Hadji Abdou Aziz NDIAYE
"""

import numpy as np


class Mesh:
    """
    Stocke les données du maillage.

    Parmeters
    ---------
    node_to_xcoord: ndarray
        Coordonnées x des noeuds du maillage.
    node_to_ycoord: ndarray
        Coordonnées y des noeuds du maillage.
    element_to_nodes_start, element_to_nodes: ndarray, ndarray
        Connectivité élément -> noeuds.
    boundary_faces_nodes: list of ndarray
        Liste des noeuds des faces frontières.

    Attributes
    ----------
    number_of_nodes: int
        Nombre de noeuds du maillage.
    number_of_elements: int
        Nombre d'éléments du maillage.
    number_of_boundaries: int
        Nombre de frontières du domaine. Les tags des faces frontières seront
        0, 1, ..., number_of_boundaries-1
    number_of_boundary_faces: int
        Nombre de faces frontières du maillage.
    number_of_faces: int
        Nombre de faces du maillage.
    node_to_xcoord: ndarray, shape (number_of_nodes,)
        Coordonnées x des noeuds du maillage.
    node_to_ycoord: ndarray, shape (number_of_nodes,)
        Coordonnées y des noeuds du maillage.
    element_to_nodes_start: ndarray, shape (number_of_elements+1,)
        Tableau contenant le découpage des noeuds des éléments. Les neouds de l'élément i sont
        entre les index element_to_nodes_start[i] et element_to_nodes_start[i+1].
    element_to_nodes: ndarray
        Tableau contenant les noeuds des éléments
    boundary_faces_nodes: list of ndarray
        List des noeuds des faces frontières
    face_to_nodes: ndarray, shape (number_of_faces,2)
        Tableau contenant les noeuds des faces.
        Les `number_of_boundary_faces` premières faces correspondent aux faces frontières.
    face_to_elements: ndarray, shape (number_of_faces,2)
        Tableau contenant les éléments voisins des faces.
        La première colonne correspond aux voisins à gauche. Les faces frontières ont - 1
        comme voisin à droite.
    boundary_face_to_tag: ndarray, shape (number_of_boundary_faces,)
        Tableau contenant les tags des faces frontières.

    """

    def __init__(self, node_to_xcoord, node_to_ycoord, element_to_nodes_start, element_to_nodes, boundary_faces_nodes):
        # Nodes
        self.number_of_nodes = node_to_xcoord.shape[0]
        self.node_to_xcoord = node_to_xcoord
        self.node_to_ycoord = node_to_ycoord
        # Elements
        self.number_of_elements = element_to_nodes_start.shape[0] - 1
        self.element_to_nodes_start = element_to_nodes_start
        self.element_to_nodes = element_to_nodes
        # Boundary Faces
        self.number_of_boundaries = len(boundary_faces_nodes)
        self.number_of_boundary_faces = 0
        for i_bound in range(self.number_of_boundaries):
            self.number_of_boundary_faces += boundary_faces_nodes[i_bound].shape[0] // 2
        self.boundary_faces_nodes = boundary_faces_nodes
        # Connectivity
        self.number_of_faces = 0
        self.face_to_nodes = np.empty((0, 2), dtype=int)
        self.face_to_elements = np.empty((0, 2), dtype=int)
        self.boundary_face_to_tag = np.empty((0,), dtype='uint8')

    def get_number_of_boundaries(self):
        """
        Retourne le nombre de frontières du maillage.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Nombre de frontières du maillage.

        """
        return self.number_of_boundaries

    def get_number_of_nodes(self):
        """
        Retourne le nombre de noeuds du maillage.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Nombre de noeuds du maillage.

        """
        return self.number_of_nodes

    def get_number_of_elements(self):
        """
        Retourne le nombre d'éléments du maillage.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Nombre d'éléments du maillage.

        """
        return self.number_of_elements

    def get_number_of_faces(self):
        """
        Retourne le nombre de faces du maillage.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Nombre de faces du maillage.

        """
        return self.number_of_faces

    def get_number_of_boundary_faces(self):
        """
        Retourne le nombre de faces frontières du maillage.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Nombre de faces frontières du maillage.

        """
        return self.number_of_boundary_faces

    def get_boundary_faces_nodes(self, i_boundary):
        """
        Retourne les noeuds des faces composant la frontière spécifiée.

        Parameters
        ----------
        i_boundary: int
            Tag de la frontière que l'on souhaite récupérer les noeuds des faces.

        Returns
        -------
        ndarray
            Noeuds des faces de la frontière.

        """
        return self.boundary_faces_nodes[i_boundary]

    def get_node_to_xcoord(self, i_node):
        """
        Retourne la coordonnée x du noeud spécifié.

        Parameters
        ----------
        i_node: int
            Index du noeud que l'on souhaite récupérer la coordonnée.

        Returns
        -------
        ndarray
            Coordonnée x du noeud.

        """
        return self.node_to_xcoord[i_node]

    def get_node_to_ycoord(self, i_node):
        """
        Retourne la coordonnée y du noeud spécifié.

        Parameters
        ----------
        i_node: int
            Index du noeud que l'on souhaite récupérer la coordonnée.

        Returns
        -------
        ndarray
            Coordonnées y du noeud.

        """
        return self.node_to_ycoord[i_node]

    def get_node_to_xycoord(self, i_node):
        """
        Retourne les coordonnées x et y du noeud spécifié.

        Parameters
        ----------
        i_node: int
            Index du noeud que l'on souhaite récupérer les coordonnées.

        Returns
        -------
        tuple
            Coordonnées x et y du noeud.

        """
        return self.node_to_xcoord[i_node], self.node_to_ycoord[i_node]

    def get_nodes_to_xcoord(self):
        """
        Retourne les coordonnées x de tous les noeuds.

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Coordonnées x des noeuds.

        """
        return self.node_to_xcoord

    def get_nodes_to_ycoord(self):
        """
        Retourne les coordonnées y de tous les noeuds.

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Coordonnées y des noeuds.

        """
        return self.node_to_ycoord

    def get_element_to_nodes(self, i_element):
        """
        Retourne les noeuds qui composent l'élément spécifié.

        Parameters
        ----------
        i_element: int
            Index de l'élément que l'on souhaite récupérer les noeuds.

        Returns
        -------
        ndarray
            Noeuds de l'élément.

        """
        start = self.element_to_nodes_start[i_element]
        end = self.element_to_nodes_start[i_element + 1]
        return self.element_to_nodes[start:end]

    def get_element_to_nodes_start(self, i_element):
        """
        Retourne l'index de démarrage des noeuds composant l'élément
        spécifié dans le tableau `element_to_nodes`.

        Parameters
        ----------
        i_element: int
            Index de l'élément que l'on souhaite récupérer l'index de démarrage.

        Returns
        -------
        int
            Index de démarrage des noeuds composant l'élément.

        """
        return self.element_to_nodes_start[i_element]

    def get_elements_to_nodes(self):
        """
        Retourne la connectivité élément -> noeuds.

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Noeuds des éléments du maillage.

        """
        return self.element_to_nodes

    def get_elements_to_nodes_start(self):
        """
        Retourne les index de démarrage des noeuds de chaque élément
        dans le tableau `element_to_nodes`.

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Index de démarrage des noeuds de chaque élément.

        """
        return self.element_to_nodes_start

    def get_face_to_nodes(self, i_face):
        """
        Retourne les neouds composant la face spécifiée.

        Parameters
        ----------
        i_face: int
            Index de la face que l'on souhaite récupérer les noeuds.

        Returns
        -------
        ndarray
            Noeuds de la face.

        """
        return self.face_to_nodes[i_face, :]

    def get_faces_to_nodes(self):
        """
        Retourne la connectivité face -> noeuds.

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Noeuds de chaque face.

        """
        return self.face_to_nodes

    def get_face_to_elements(self, i_face):
        """
        Retourne les éléments voisins de la face spécifiée.

        Parameters
        ----------
        i_face: int
            Index de la face que l'on souhaite récupérer les éléments voisins.

        Returns
        -------
        ndarray
            Éléments voisins de la face.

        """
        return self.face_to_elements[i_face, :]

    def get_faces_to_elements(self):
        """
        Retourne la connectivité face -> elements.

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Éléments voisins de chaque face.

        """
        return self.face_to_elements

    def get_boundary_face_to_tag(self, i_face):
        """
        Retourne le tag de la face frontière spécifiée.

        Parameters
        ---------- 
        i_face: int
            Index de la face frontière que l'on souhaite récupérer le tag.

        Returns
        -------
        ndarray
            Tag de la face frontière.

        """
        return self.boundary_face_to_tag[i_face]

    def get_boundary_faces_to_tag(self):
        """
        Retourne la connectivité boundary face -> tag

        Parameters
        ---------- 
        None

        Returns
        -------
        ndarray
            Tags des faces frontières.

        """
        return self.boundary_face_to_tag

    def set_number_of_faces(self, number_of_faces):
        """
        Mute le nombre de faces.

        Parameters
        ---------- 
        number_of_faces: int
            Nouvelle valeur du nombre de faces.

        Returns
        -------
        None

        """
        self.number_of_faces = number_of_faces

    def set_face_to_nodes(self, face_to_nodes):
        """
        Mute la connectivité face -> noeuds.

        Parameters
        ---------- 
        face_to_nodes: int
            Nouvelle connectivité face - > noeuds.

        Returns
        -------
        None

        """
        self.face_to_nodes = face_to_nodes

    def set_face_to_elements(self, face_to_elements):
        """
        Mute la connectivité face -> éléments.

        Parameters
        ---------- 
        face_to_elements: int
            Nouvelle connectivité face - > éléments.

        Returns
        -------
        None

        """
        self.face_to_elements = face_to_elements

    def set_boundary_face_to_tag(self, boundary_face_to_tag):
        """
        Mute la connectivité boundary face -> tag.

        Parameters
        ---------- 
        boundary_face_to_tag: int
            Nouvelle connectivité boundary face - > tag.

        Returns
        -------
        None

        """
        self.boundary_face_to_tag = boundary_face_to_tag
