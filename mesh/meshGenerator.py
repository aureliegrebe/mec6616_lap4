# -*- coding: utf-8 -*-
"""
Module Maillage: Génération
MEC6616 - Aérodynamique numérique
Date de création: 2021 - 12 - 26
Auteurs: El Hadji Abdou Aziz NDIAYE, Eddy PETRO, Jean-Yves TRÉPANIER
"""

import gmsh
import numpy as np
import mesh


class MeshGenerator:
    """
    Génére des maillages 2D avec l'API Gmesh.

    La classe permet de générer des maillages composés de triangles et/ou de quadrialatères pour
    différents types de géométries.
    Les données de chaque maillage généré sont encapsulées dans un objet `Mesh`. Ces données sont :
        * Les coordonnées des noeuds
        * La connectivité élément -> noeuds : liste des noeuds de chaque élément
        * Les noeuds des faces situées à la frontière du domaine

    Parameters
    ----------
    verbose : bool, optionnel
        Permet de controler l'affichage sur le console.

    """

    def __init__(self, verbose=False):
        if verbose:
            self.verbose = 1
        else:
            self.verbose = 0
        return

    def rectangle(self, rectangle_corners, mesh_parameters):
        """
        Génère un maillage pour la géométrie du rectangle. 

                                tag=3
        y_max   -----------------------------------------
                |                                       |
          tag=0 |                                       | tag=2
                |                                       |
        y_min   -----------------------------------------
                                tag=1
                x_min                                   x_max

        Parameters
        ----------
        rectangle_corners : list|tuple|ndarray
            Tableau contenant les coordonnées des coins du rectangle [x_min, x_max, y_min, y_max]
        mesh_parameters : dict
            Dictionnaire contenant les paramètres du maillage à générer. Les clés sont :
            * 'mesh_type': string, optionnel
                Type des éléments du maillage. Les valeurs possibles sont :
                - 'TRI' : triangles (par défaut)
                - 'QUAD' : quadrilatères
                - 'MIX' : mix de triangles et de quadrilatères
            * 'lc': double
                taille caractéristique des éléments du maillage (uniquement pour les maillages non-structurés).
            * 'Nx': integer ; 'Ny': integer
                Nombre de cellules suivant les axes x et y (uniquement pour les maillages transfinis).

            Il faut fournir le paramètre 'lc' pour obtenir des maillages non-structurés 
            et les paramètres 'Nx' et 'Ny' pour le cas des maillages transfinis.

            Les maillages non-structurés de type "QUAD" peuvent parfois contenir des triangles.

        Returns
        -------
        mesh : Mesh object
            Objet mesh contenant les données du maillage généré.

        """
        # Initializing the Gmsh API
        gmsh.initialize()
        # Console output
        gmsh.option.setNumber("General.Terminal", self.verbose)

        # Gemetry
        gmsh.model.add("Rectangle")  # Name
        # # Points of the geometry
        if 'lc' in mesh_parameters:
            lc = mesh_parameters['lc']
        else:
            lc = 0.0
        #       Pt2 ----------- Pt1
        #       |     Edge 1     |
        # Edge2 |                | Edge4
        #       |     Edge 3     |
        #       Pt3 ----------- Pt4
        gmsh.model.geo.addPoint(rectangle_corners[1], rectangle_corners[3], 0.0, lc, 1)  # Pt1
        gmsh.model.geo.addPoint(rectangle_corners[0], rectangle_corners[3], 0.0, lc, 2)  # Pt2
        gmsh.model.geo.addPoint(rectangle_corners[0], rectangle_corners[2], 0.0, lc, 3)  # Pt3
        gmsh.model.geo.addPoint(rectangle_corners[1], rectangle_corners[2], 0.0, lc, 4)  # Pt4
        # # Edges of the geometry
        gmsh.model.geo.addLine(1, 2, 1)  # Edge 1 : Pt1 <-> Pt2
        gmsh.model.geo.addLine(2, 3, 2)  # Edge 2 : Pt2 <-> Pt3
        gmsh.model.geo.addLine(3, 4, 3)  # Edge 3 : Pt3 <-> Pt4
        gmsh.model.geo.addLine(4, 1, 4)  # Edge 4 : Pt4 <-> Pt1
        boundary_tags = [[2], [3], [4], [1]]
        # # Surface zone and Physicals Group
        gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.addPhysicalGroup(2, [1], 1)
        gmsh.model.setPhysicalName(2, 1, "DOMAINE")

        # Mesh
        if 'lc' not in mesh_parameters:
            nx = mesh_parameters['Nx']+1
            ny = mesh_parameters['Ny']+1
            gmsh.model.geo.mesh.setTransfiniteCurve(1, nx)
            gmsh.model.geo.mesh.setTransfiniteCurve(2, ny)
            gmsh.model.geo.mesh.setTransfiniteCurve(3, nx)
            gmsh.model.geo.mesh.setTransfiniteCurve(4, ny)
            gmsh.model.geo.mesh.setTransfiniteSurface(1, "Alternate")
        if mesh_parameters['mesh_type'] == "QUAD":
            gmsh.model.geo.mesh.setRecombine(2, 1)
            # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        elif mesh_parameters['mesh_type'] == "MIX":
            x_mid = 0.5*(rectangle_corners[0]+rectangle_corners[1])
            gmsh.model.geo.remove([(2, 1), (1, 1), (1, 3)])
            #       Pt2 ----- Pt5 ------ Pt1
            #       |    E3   |    E1    |
            #    E2 |         | E7       | E4
            #       |    E5   |    E6    |
            #       Pt3 ----- Pt6 ------ Pt4
            gmsh.model.geo.addPoint(x_mid, rectangle_corners[3], 0.0, lc, 5)  # Pt5
            gmsh.model.geo.addPoint(x_mid, rectangle_corners[2], 0.0, lc, 6)  # Pt6
            gmsh.model.geo.addLine(1, 5, 1)  # Edge 1 : Pt1 <-> Pt5
            gmsh.model.geo.addLine(5, 2, 3)  # Edge 3 : Pt5 <-> Pt2
            gmsh.model.geo.addLine(3, 6, 5)  # Edge 5 : Pt3 <-> Pt6
            gmsh.model.geo.addLine(6, 4, 6)  # Edge 6 : Pt6 <-> Pt4
            gmsh.model.geo.addLine(5, 6, 7)  # Edge 7 : Pt5 <-> Pt6
            gmsh.model.geo.addCurveLoop([1, 7, 6, 4], 2)
            gmsh.model.geo.addCurveLoop([3, 2, 5, -7], 3)
            gmsh.model.geo.addPlaneSurface([2], 2)
            gmsh.model.geo.addPlaneSurface([3], 3)
            gmsh.model.addPhysicalGroup(2, [2, 3], 2)
            gmsh.model.setPhysicalName(2, 2, "DOMAINE")
            if 'lc' not in mesh_parameters:
                nx = mesh_parameters['Nx']//2+1
                ny = mesh_parameters['Ny']+1
                gmsh.model.geo.mesh.setTransfiniteCurve(1, nx)
                gmsh.model.geo.mesh.setTransfiniteCurve(2, ny)
                gmsh.model.geo.mesh.setTransfiniteCurve(3, nx)
                gmsh.model.geo.mesh.setTransfiniteCurve(4, ny)
                gmsh.model.geo.mesh.setTransfiniteCurve(5, nx)
                gmsh.model.geo.mesh.setTransfiniteCurve(6, nx)
                gmsh.model.geo.mesh.setTransfiniteCurve(7, ny)
                gmsh.model.geo.mesh.setTransfiniteSurface(2, "Alternate")
                gmsh.model.geo.mesh.setTransfiniteSurface(3)
            gmsh.model.geo.mesh.setRecombine(2, 3)
            boundary_tags = [[2], [5, 6], [4], [1, 3]]
        gmsh.option.setNumber("Mesh.Smoothing", 100)
        # # Generation of the mesh
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        # gmsh.write("Rectangle.su2")  # write the mesh in a file
        # gmsh.fltk.run()  # display the mesh
        # # Getting the nodes coordinates
        _, node_coords, _ = gmsh.model.mesh.getNodes()
        # Getting the connectivity of the elements
        elem_types_domaine, _, elem_node_tags_domaine = gmsh.model.mesh.getElements(dim=2)
        # Getting the boundary elements
        elem_boundary = []
        for b_tags in boundary_tags:
            elem_bound = []
            for b_tag in b_tags:
                _, _, elem_node_tags = gmsh.model.mesh.getElements(dim=1, tag=b_tag)
                elem_bound.append(elem_node_tags[0])
            elem_bound = np.concatenate(elem_bound, dtype=int)
            elem_boundary.append(elem_bound)

        # Finalizing the Gmsh API
        gmsh.finalize()

        # Construction of the Mesh Object
        mesh_obj = self.generate_mesh_object(node_coords, elem_types_domaine, elem_node_tags_domaine, elem_boundary)

        return mesh_obj

    def back_step(self, H1, H2, L1, L2, mesh_parameters):
        """
        Génère un maillage pour la géométrie d'une marche descendante. 

                                 tag=3          
                -----------------------------------------
                |                 L2                    |
          tag=0 |  H1                                   |   
                |       tag=1                        H2 | tag=2  
          (0,0) --------------------                    | 
                        L1         | tag=1              |
                                   |                    |
                                   ----------------------
                                           tag=1

        Parameters
        ----------
        H1,H2,L1,L2 :  doubles
            Dimensions caractéristiques de la marche descendante.
        mesh_parameters : dict
            Dictionnaire contenant les paramètres du maillages à générer. Les clés sont :
            * 'mesh_type': string, optionnel
                Type des éléments du maillage. Les valeurs possibles sont :
                - 'TRI' : triangles (par défaut)
                - 'QUAD' : quadrilatères
                - 'MIX' : mixe de triangles et de quadrilatères
            * 'lc': double
                taille caractéristique des éléments du maillage (maillages non-structurés).
            * 'Nx': integer ; 'Ny': integer
                Nombre de cellules suivant les axes x et y (maillages transfinis).

            Il faut fournir le paramètre 'lc' pour obtenir des maillages non-structurés 
            et les paramètres 'Nx' et 'Ny' pour le cas des maillages transfinis.

            Les maillages non-structurés de type "QUAD" peuvent parfois contenir des triangles.

        Returns
        -------
        mesh : Mesh object
            Objet mesh contenant les données du maillage généré.

        """
        # Initializing the Gmsh API
        gmsh.initialize()
        # Console output
        gmsh.option.setNumber("General.Terminal", self.verbose)

        # Gemetry
        gmsh.model.add("BackStep")  # Name
        # # Points of the geometry
        if 'lc' in mesh_parameters:
            lc = mesh_parameters['lc']
        else:
            lc = 0.0
        #         P1-------------------------------------- P6
        #         |                   E6                   |
        #      E1 |                                        |
        #         |                                        | E5
        #         P2 --------------- P3                    |
        #                 E2         | E3                  |
        #                            |         E4          |
        #                            P4 ------------------ P5
        gmsh.model.geo.addPoint(0.0, H1,    0.0, lc, 1)  # Pt1
        gmsh.model.geo.addPoint(0.0, 0.0,   0.0, lc, 2)  # Pt2
        gmsh.model.geo.addPoint(L1,  0.0,   0.0, lc, 3)  # Pt3
        gmsh.model.geo.addPoint(L1,  H1-H2, 0.0, lc, 4)  # Pt4
        gmsh.model.geo.addPoint(L2,  H1-H2, 0.0, lc, 5)  # Pt5
        gmsh.model.geo.addPoint(L2,  H1,    0.0, lc, 6)  # Pt6
        gmsh.model.geo.addPoint(L1,  H1,    0.0, lc, 7)  # Pt7
        # # Edges of the geometry
        gmsh.model.geo.addLine(1, 2, 1)  # Edge 1 : Pt1 <-> Pt2
        gmsh.model.geo.addLine(2, 3, 2)  # Edge 2 : Pt2 <-> Pt3
        gmsh.model.geo.addLine(3, 4, 3)  # Edge 3 : Pt3 <-> Pt4
        gmsh.model.geo.addLine(4, 5, 4)  # Edge 4 : Pt4 <-> Pt5
        gmsh.model.geo.addLine(5, 6, 5)  # Edge 5 : Pt5 <-> Pt6
        gmsh.model.geo.addLine(6, 7, 6)  # Edge 6 : Pt6 <-> Pt7
        gmsh.model.geo.addLine(7, 1, 7)  # Edge 7 : Pt7 <-> Pt1
        gmsh.model.geo.addLine(7, 3, 8)  # Edge 8 : Pt7 <-> Pt3
        boundary_tags = [[1], [2, 3, 4], [5], [6, 7]]
        # # Surface zone and Physicals Group
        gmsh.model.geo.addCurveLoop([1, 2, -8, 7], 1)
        gmsh.model.geo.addCurveLoop([8, 3, 4, 5, 6], 2)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.addPlaneSurface([2], 2)
        gmsh.model.addPhysicalGroup(2, [1, 2], 1)
        gmsh.model.setPhysicalName(2, 1, "DOMAINE")

        # Mesh
        if 'lc' not in mesh_parameters:
            nx = mesh_parameters['Nx']+1
            nx1 = max(int(L1*nx/L2), 2)
            nx2 = max(nx - nx1, 2)
            nx = nx1 + nx2
            ny = mesh_parameters['Ny']+1
            ny1 = max(int(H1*ny/H2), 2)
            ny2 = max(ny - ny1, 2)
            ny = ny1 + ny2
            gmsh.model.geo.mesh.setTransfiniteCurve(1, ny1)
            gmsh.model.geo.mesh.setTransfiniteCurve(2, nx1)
            gmsh.model.geo.mesh.setTransfiniteCurve(3, ny2+1)
            gmsh.model.geo.mesh.setTransfiniteCurve(4, nx2)
            gmsh.model.geo.mesh.setTransfiniteCurve(5, ny)
            gmsh.model.geo.mesh.setTransfiniteCurve(6, nx2)
            gmsh.model.geo.mesh.setTransfiniteCurve(7, nx1)
            gmsh.model.geo.mesh.setTransfiniteCurve(8, ny1)
            gmsh.model.geo.mesh.setTransfiniteSurface(1, "Alternate")
            gmsh.model.geo.mesh.setTransfiniteSurface(2, "Right", cornerTags=[7, 4, 5, 6])
        if mesh_parameters['mesh_type'] == "QUAD":
            gmsh.model.geo.mesh.setRecombine(2, 1)
            gmsh.model.geo.mesh.setRecombine(2, 2)
        elif mesh_parameters['mesh_type'] == "MIX":
            gmsh.model.geo.mesh.setRecombine(2, 1)
        gmsh.option.setNumber("Mesh.Smoothing", 100)
        # # Generation of the mesh
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        # # Getting the nodes coordinates
        _, node_coords, _ = gmsh.model.mesh.getNodes()
        # Getting the connectivity of the elements
        elem_types_domaine, _, elem_node_tags_domaine = gmsh.model.mesh.getElements(dim=2)
        # Getting the boundary elements
        elem_boundary = []
        for b_tags in boundary_tags:
            elem_bound = []
            for b_tag in b_tags:
                _, _, elem_node_tags = gmsh.model.mesh.getElements(dim=1, tag=b_tag)
                elem_bound.append(elem_node_tags[0])
            elem_bound = np.concatenate(elem_bound, dtype=int)
            elem_boundary.append(elem_bound)

        # Finalizing the Gmsh API
        gmsh.finalize()

        # Construction of the Mesh Object
        mesh_obj = self.generate_mesh_object(node_coords, elem_types_domaine, elem_node_tags_domaine, elem_boundary)

        return mesh_obj

    def circle(self, rectangle_corners, rayon, mesh_parameters):
        """
        Génère un maillage pour la géométrie d'un cercle dans un rectangle. 

        y_max  -----------------------------------------
               |                                       |
        tag=0  |                 .-.                   |  tag=2
               |                 '-'tag=4              | 
               |                                       |
        y_min  -----------------------------------------
                               tag=1                          
              x_min                                   x_max        

        Parameters
        ----------
        rectangle_corners : list|tuple|ndarray
            Tableau contenant les coordonnées des coins du rectangle [x_min, x_max, y_min, y_max]
        rayon : double
            Rayon du cercle
        mesh_parameters : dict
            Dictionnaire contenant les paramètres du maillages à générer. Les clés sont :
            * 'mesh_type': string, optionnel
                Type des éléments du maillage. Les valeurs possibles sont :
                - 'TRI' : triangles (par défaut)
                - 'QUAD' : quadrilatères
            * 'lc_rectangle': double
                Longueur visée des arètes des éléments appliqué aux côtés du rectangle (maillages non-structurés).
            * 'lc_circle': double
                Longueur visée des arètes des éléments appliqué au circle (maillages non-structurés).
            * 'Nx', 'Ny', 'Nc': integers
                Nombre de cellules suivant les axes x et y et sur le cercle (maillages transfinis)

            Il faut fournir les paramètres 'lc_rectangle' et 'lc_circle' pour obtenir des maillages non-structurés 
            et les paramètres 'Nx' , 'Ny' et 'Nc' pour le cas des maillages transfinis.

            Les maillages non-structurés de type "QUAD" peuvent parfois contenir des triangles.

        Returns
        -------
        mesh : Mesh object
            Objet mesh contenant les données du maillage généré.

        """
        # Initializing the Gmsh API
        gmsh.initialize()
        # Console output
        gmsh.option.setNumber("General.Terminal", self.verbose)

        # Gemetry
        gmsh.model.add("Circle")  # Name
        # # Points of the geometry
        circle_center = [(rectangle_corners[0]+rectangle_corners[1])/2.0, (rectangle_corners[2]+rectangle_corners[3])/2.0]

        if 'lc_rectangle' in mesh_parameters:
            lc_rect = mesh_parameters['lc_rectangle']
            lc_circ = mesh_parameters['lc_circle']
            gmsh.model.geo.addPoint(rectangle_corners[0], rectangle_corners[3], 0.0, lc_rect, 1)  # Pt1
            gmsh.model.geo.addPoint(rectangle_corners[0], rectangle_corners[2], 0.0, lc_rect, 2)  # Pt2
            gmsh.model.geo.addPoint(rectangle_corners[1], rectangle_corners[2], 0.0, lc_rect, 3)  # Pt3
            gmsh.model.geo.addPoint(rectangle_corners[1], rectangle_corners[3], 0.0, lc_rect, 4)  # Pt4
            gmsh.model.geo.addPoint(circle_center[0],       circle_center[1],       0.0, lc_circ, 5)  # Pt5
            gmsh.model.geo.addPoint(circle_center[0]+rayon, circle_center[1],       0.0, lc_circ, 6)  # Pt6
            gmsh.model.geo.addPoint(circle_center[0],       circle_center[1]+rayon, 0.0, lc_circ, 7)  # Pt7
            gmsh.model.geo.addPoint(circle_center[0]-rayon, circle_center[1],       0.0, lc_circ, 8)  # Pt8
            gmsh.model.geo.addPoint(circle_center[0],       circle_center[1]-rayon, 0.0, lc_circ, 9)  # Pt9
            # # Edges of the geometry
            gmsh.model.geo.addLine(1, 2, 1)  # Edge 1 : Pt1 <-> Pt2
            gmsh.model.geo.addLine(2, 3, 2)  # Edge 2 : Pt2 <-> Pt3
            gmsh.model.geo.addLine(3, 4, 3)  # Edge 3 : Pt3 <-> Pt4
            gmsh.model.geo.addLine(4, 1, 4)  # Edge 4 : Pt4 <-> Pt1
            gmsh.model.geo.addCircleArc(6, 5, 7, 5)  # Edge 5 : arc Pt6 <-> Pt7
            gmsh.model.geo.addCircleArc(7, 5, 8, 6)  # Edge 6 : arc Pt7 <-> Pt8
            gmsh.model.geo.addCircleArc(8, 5, 9, 7)  # Edge 7 : arc Pt8 <-> Pt9
            gmsh.model.geo.addCircleArc(9, 5, 6, 8)  # Edge 8 : arc Pt9 <-> Pt6
            boundary_tags = [[1], [2], [3], [4], [5, 6, 7, 8]]
            # # Surface zone and Physicals Group
            gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
            gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 2)
            gmsh.model.geo.addPlaneSurface([1, 2], 1)
            gmsh.model.addPhysicalGroup(2, [1], 1)
            gmsh.model.setPhysicalName(2, 1, "DOMAINE")
            if mesh_parameters['mesh_type'] == "QUAD":
                gmsh.model.geo.mesh.setRecombine(2, 1)
        else:
            cospi4 = np.cos(np.pi/4)
            length_min = min(rectangle_corners[1]-rectangle_corners[0], rectangle_corners[3]-rectangle_corners[2])
            rext = max(min(3*rayon, 0.75*length_min/2.), 1.5*rayon)
            gmsh.model.geo.addPoint(rectangle_corners[0], rectangle_corners[3], 0.0, 0.0, 1)  # Pt1
            gmsh.model.geo.addPoint(rectangle_corners[0], rectangle_corners[2], 0.0, 0.0, 2)  # Pt2
            gmsh.model.geo.addPoint(rectangle_corners[1], rectangle_corners[2], 0.0, 0.0, 3)  # Pt3
            gmsh.model.geo.addPoint(rectangle_corners[1], rectangle_corners[3], 0.0, 0.0, 4)  # Pt4

            gmsh.model.geo.addPoint(circle_center[0],       circle_center[1],       0.0, 0.0, 5)  # Pt5

            gmsh.model.geo.addPoint(circle_center[0]+rayon*cospi4, circle_center[1]+rayon*cospi4, 0.0, 0.0, 6)  # Pt6
            gmsh.model.geo.addPoint(circle_center[0]-rayon*cospi4, circle_center[1]+rayon*cospi4, 0.0, 0.0, 7)  # Pt7
            gmsh.model.geo.addPoint(circle_center[0]-rayon*cospi4, circle_center[1]-rayon*cospi4, 0.0, 0.0, 8)  # Pt8
            gmsh.model.geo.addPoint(circle_center[0]+rayon*cospi4, circle_center[1]-rayon*cospi4, 0.0, 0.0, 9)  # Pt9

            gmsh.model.geo.addPoint(circle_center[0]+rext*cospi4, circle_center[1]+rext*cospi4, 0.0, 0.0, 10)  # Pt10
            gmsh.model.geo.addPoint(circle_center[0]-rext*cospi4, circle_center[1]+rext*cospi4, 0.0, 0.0, 11)  # Pt11
            gmsh.model.geo.addPoint(circle_center[0]-rext*cospi4, circle_center[1]-rext*cospi4, 0.0, 0.0, 12)  # Pt12
            gmsh.model.geo.addPoint(circle_center[0]+rext*cospi4, circle_center[1]-rext*cospi4, 0.0, 0.0, 13)  # Pt13

            gmsh.model.geo.addPoint(circle_center[0]+rext*cospi4, rectangle_corners[2], 0.0, 0.0, 14)  # Pt14
            gmsh.model.geo.addPoint(circle_center[0]-rext*cospi4, rectangle_corners[2], 0.0, 0.0, 15)  # Pt15
            gmsh.model.geo.addPoint(circle_center[0]+rext*cospi4, rectangle_corners[3], 0.0, 0.0, 16)  # Pt16
            gmsh.model.geo.addPoint(circle_center[0]-rext*cospi4, rectangle_corners[3], 0.0, 0.0, 17)  # Pt17

            gmsh.model.geo.addPoint(rectangle_corners[0], circle_center[1]+rext*cospi4, 0.0, 0.0, 18)  # Pt18
            gmsh.model.geo.addPoint(rectangle_corners[0], circle_center[1]-rext*cospi4, 0.0, 0.0, 19)  # Pt19
            gmsh.model.geo.addPoint(rectangle_corners[1], circle_center[1]+rext*cospi4, 0.0, 0.0, 20)  # Pt20
            gmsh.model.geo.addPoint(rectangle_corners[1], circle_center[1]-rext*cospi4, 0.0, 0.0, 21)  # Pt21

            gmsh.model.geo.addLine(1,  18,  1)
            gmsh.model.geo.addLine(18, 19,  2)
            gmsh.model.geo.addLine(19,  2,  3)
            gmsh.model.geo.addLine(2,  15,  4)
            gmsh.model.geo.addLine(15, 14,  5)
            gmsh.model.geo.addLine(14,  3,  6)
            gmsh.model.geo.addLine(3,  21,  7)
            gmsh.model.geo.addLine(21, 20,  8)
            gmsh.model.geo.addLine(20,  4,  9)
            gmsh.model.geo.addLine(4,  16, 10)
            gmsh.model.geo.addLine(16, 17, 11)
            gmsh.model.geo.addLine(17,  1, 12)
            gmsh.model.geo.addCircleArc(6,  5,  7, 13)
            gmsh.model.geo.addCircleArc(7,  5,  8, 14)
            gmsh.model.geo.addCircleArc(8,  5,  9, 15)
            gmsh.model.geo.addCircleArc(9,  5,  6, 16)
            gmsh.model.geo.addCircleArc(10, 5, 11, 17)
            gmsh.model.geo.addCircleArc(11, 5, 12, 18)
            gmsh.model.geo.addCircleArc(12, 5, 13, 19)
            gmsh.model.geo.addCircleArc(13, 5, 10, 20)
            gmsh.model.geo.addLine(13, 14, 21)
            gmsh.model.geo.addLine(12, 15, 22)
            gmsh.model.geo.addLine(10, 16, 23)
            gmsh.model.geo.addLine(11, 17, 24)
            gmsh.model.geo.addLine(11, 18, 25)
            gmsh.model.geo.addLine(12, 19, 26)
            gmsh.model.geo.addLine(10, 20, 27)
            gmsh.model.geo.addLine(13, 21, 28)
            gmsh.model.geo.addLine(6,  10, 29)
            gmsh.model.geo.addLine(7,  11, 30)
            gmsh.model.geo.addLine(8,  12, 31)
            gmsh.model.geo.addLine(9,  13, 32)
            gmsh.model.geo.addCurveLoop([1,   -25,  24,  12], 1)
            gmsh.model.geo.addCurveLoop([2,   -26, -18,  25], 2)
            gmsh.model.geo.addCurveLoop([3,     4, -22,  26], 3)
            gmsh.model.geo.addCurveLoop([22,    5, -21, -19], 4)
            gmsh.model.geo.addCurveLoop([21,    6,   7, -28], 5)
            gmsh.model.geo.addCurveLoop([-20,  28,   8, -27], 6)
            gmsh.model.geo.addCurveLoop([-23,  27,   9,  10], 7)
            gmsh.model.geo.addCurveLoop([-24, -17,  23,  11], 8)
            gmsh.model.geo.addCurveLoop([29,   17, -30, -13], 9)
            gmsh.model.geo.addCurveLoop([30,   18, -31, -14], 10)
            gmsh.model.geo.addCurveLoop([31,   19, -32, -15], 11)
            gmsh.model.geo.addCurveLoop([32,   20, -29, -16], 12)
            gmsh.model.geo.addPlaneSurface([1], 1)
            gmsh.model.geo.addPlaneSurface([2], 2)
            gmsh.model.geo.addPlaneSurface([3], 3)
            gmsh.model.geo.addPlaneSurface([4], 4)
            gmsh.model.geo.addPlaneSurface([5], 5)
            gmsh.model.geo.addPlaneSurface([6], 6)
            gmsh.model.geo.addPlaneSurface([7], 7)
            gmsh.model.geo.addPlaneSurface([8], 8)
            gmsh.model.geo.addPlaneSurface([9], 9)
            gmsh.model.geo.addPlaneSurface([10], 10)
            gmsh.model.geo.addPlaneSurface([11], 11)
            gmsh.model.geo.addPlaneSurface([12], 12)

            nx = mesh_parameters['Nx']//2+1
            ny = mesh_parameters['Ny']//2+1
            nc = mesh_parameters['Nc']//4+1
            gmsh.model.geo.mesh.setTransfiniteCurve(1, ny)
            gmsh.model.geo.mesh.setTransfiniteCurve(2, nc)
            gmsh.model.geo.mesh.setTransfiniteCurve(3, ny)
            gmsh.model.geo.mesh.setTransfiniteCurve(4, nx)
            gmsh.model.geo.mesh.setTransfiniteCurve(5, nc)
            gmsh.model.geo.mesh.setTransfiniteCurve(6, nx)
            gmsh.model.geo.mesh.setTransfiniteCurve(7, ny)
            gmsh.model.geo.mesh.setTransfiniteCurve(8, nc)
            gmsh.model.geo.mesh.setTransfiniteCurve(9, ny)
            gmsh.model.geo.mesh.setTransfiniteCurve(10, nx)
            gmsh.model.geo.mesh.setTransfiniteCurve(11, nc)
            gmsh.model.geo.mesh.setTransfiniteCurve(12, nx)
            gmsh.model.geo.mesh.setTransfiniteCurve(13, nc)
            gmsh.model.geo.mesh.setTransfiniteCurve(14, nc)
            gmsh.model.geo.mesh.setTransfiniteCurve(15, nc)
            gmsh.model.geo.mesh.setTransfiniteCurve(16, nc)
            gmsh.model.geo.mesh.setTransfiniteCurve(17, nc)
            gmsh.model.geo.mesh.setTransfiniteCurve(18, nc)
            gmsh.model.geo.mesh.setTransfiniteCurve(19, nc)
            gmsh.model.geo.mesh.setTransfiniteCurve(20, nc)
            gmsh.model.geo.mesh.setTransfiniteCurve(21, ny)
            gmsh.model.geo.mesh.setTransfiniteCurve(22, ny)
            gmsh.model.geo.mesh.setTransfiniteCurve(23, ny)
            gmsh.model.geo.mesh.setTransfiniteCurve(24, ny)
            gmsh.model.geo.mesh.setTransfiniteCurve(25, nx)
            gmsh.model.geo.mesh.setTransfiniteCurve(26, nx)
            gmsh.model.geo.mesh.setTransfiniteCurve(27, nx)
            gmsh.model.geo.mesh.setTransfiniteCurve(28, nx)
            gmsh.model.geo.mesh.setTransfiniteCurve(29, nc)
            gmsh.model.geo.mesh.setTransfiniteCurve(30, nc)
            gmsh.model.geo.mesh.setTransfiniteCurve(31, nc)
            gmsh.model.geo.mesh.setTransfiniteCurve(32, nc)
            gmsh.model.geo.mesh.setTransfiniteSurface(1, "Alternate")
            gmsh.model.geo.mesh.setTransfiniteSurface(2, "Alternate")
            gmsh.model.geo.mesh.setTransfiniteSurface(3, "Alternate")
            gmsh.model.geo.mesh.setTransfiniteSurface(4, "Alternate")
            gmsh.model.geo.mesh.setTransfiniteSurface(5, "Alternate")
            gmsh.model.geo.mesh.setTransfiniteSurface(6, "Alternate")
            gmsh.model.geo.mesh.setTransfiniteSurface(7, "Alternate")
            gmsh.model.geo.mesh.setTransfiniteSurface(8, "Alternate")
            gmsh.model.geo.mesh.setTransfiniteSurface(9, "Alternate")
            gmsh.model.geo.mesh.setTransfiniteSurface(10, "Alternate")
            gmsh.model.geo.mesh.setTransfiniteSurface(11, "Alternate")
            gmsh.model.geo.mesh.setTransfiniteSurface(12, "Alternate")
            gmsh.model.addPhysicalGroup(2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 1)
            gmsh.model.setPhysicalName(2, 1, "DOMAINE")
            if mesh_parameters['mesh_type'] == "QUAD":
                gmsh.model.geo.mesh.setRecombine(2, 1)
                gmsh.model.geo.mesh.setRecombine(2, 2)
                gmsh.model.geo.mesh.setRecombine(2, 3)
                gmsh.model.geo.mesh.setRecombine(2, 4)
                gmsh.model.geo.mesh.setRecombine(2, 5)
                gmsh.model.geo.mesh.setRecombine(2, 6)
                gmsh.model.geo.mesh.setRecombine(2, 7)
                gmsh.model.geo.mesh.setRecombine(2, 8)
                gmsh.model.geo.mesh.setRecombine(2, 9)
                gmsh.model.geo.mesh.setRecombine(2, 10)
                gmsh.model.geo.mesh.setRecombine(2, 11)
                gmsh.model.geo.mesh.setRecombine(2, 12)
            boundary_tags = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15, 16]]

        gmsh.option.setNumber("Mesh.Smoothing", 100)
        # # Generation of the mesh
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        # # Getting the nodes coordinates
        nodes_tag, node_coords, _ = gmsh.model.mesh.getNodes()
        # delete unused nodes
        # Getting the connectivity of the elements
        elem_types_domaine, _, elem_node_tags_domaine = gmsh.model.mesh.getElements(dim=2)
        max_tag = 0
        for i in range(len(elem_node_tags_domaine)):
            max_tag = max(max_tag, max(elem_node_tags_domaine[i]))
        index_to_delete = np.where(nodes_tag > max_tag)[0]
        ind1 = 3*index_to_delete
        ind2, ind3 = ind1 + 1, ind1 + 2
        index_to_delete = np.concatenate((ind1, ind2, ind3))
        node_coords = np.delete(node_coords, index_to_delete)
        # Getting the boundary elements
        elem_boundary = []
        for b_tags in boundary_tags:
            elem_bound = []
            for b_tag in b_tags:
                _, _, elem_node_tags = gmsh.model.mesh.getElements(dim=1, tag=b_tag)
                elem_bound.append(elem_node_tags[0])
            elem_bound = np.concatenate(elem_bound, dtype=int)
            elem_boundary.append(elem_bound)

        # Finalizing the Gmsh API
        gmsh.finalize()

        # Construction of the Mesh Object
        mesh_obj = self.generate_mesh_object(node_coords, elem_types_domaine, elem_node_tags_domaine, elem_boundary)

        return mesh_obj

    def quarter_annular(self, R1, R2, mesh_parameters):
        """
        Génère un maillage d'un quart d'anneau. 

            R2     --------  \
                   |             \                           
           tag = 0 |                \                       
                   |                  \    tag = 3      
                   |                   \      
            R1     ------               \
                          \              \
                           \             \
                    tag = 1 \            |
                             |           |
                             |__________ |

                                tag = 2     
        Parameters
        ----------
        R1 :  double
            Rayon interne.
        R2 : double
            Rayon externe.
        mesh_parameters : dict
            Dictionnaire contenant les paramètres du maillages à générer. Les clés sont :
            * 'mesh_type': string, optionnel
                Type des éléments du maillage. Les valeurs possibles sont :
                - 'TRI' : triangles (par défaut)
                - 'QUAD' : quadrilatères
                - 'MIX' : mixe de triangles et de quadrilatères
            * 'lc': double
                taille caractéristique des éléments du maillage (maillages non-structurés).
            * 'N1': integer ; 'N2': integer
                Nombre de cellules sur les quarts de cerlce (tags 1 et 3) et sur les autres
                frontières (tags 0 et 2) respectivement (maillages transfinis).

            Il faut fournir le paramètre 'lc' pour obtenir des maillages non-structurés 
            et les paramètres 'N1' et 'N2' pour le cas des maillages transfinis.

            Les maillages non-structurés de type "QUAD" peuvent parfois contenir des triangles.

        Returns
        -------
        mesh : Mesh object
            Objet mesh contenant les données du maillage généré.

        """
        # Initializing the Gmsh API
        gmsh.initialize()
        # Console output
        gmsh.option.setNumber("General.Terminal", self.verbose)

        # Gemetry
        gmsh.model.add("QuarterAnnular")  # Name
        # # Points of the geometry
        if 'lc' in mesh_parameters:
            lc = mesh_parameters['lc']
        else:
            lc = 0.0
        gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc, 1)  # Pt1
        gmsh.model.geo.addPoint(0.0, R2,  0.0, lc, 2)  # Pt2
        gmsh.model.geo.addPoint(0.0, R1,  0.0, lc, 3)  # Pt3
        gmsh.model.geo.addPoint(R1,  0.0, 0.0, lc, 4)  # Pt4
        gmsh.model.geo.addPoint(R2,  0.0, 0.0, lc, 5)  # Pt5
        # # Edges of the geometry
        gmsh.model.geo.addLine(2, 3, 1)  # Edge 1 : Pt2 <-> Pt3
        gmsh.model.geo.addCircleArc(3, 1, 4, 2)  # Edge 2 : Pt3 <-> Pt4
        gmsh.model.geo.addLine(4, 5, 3)  # Edge 3 : Pt4 <-> Pt5
        gmsh.model.geo.addCircleArc(5, 1, 2, 4)  # Edge 4 : Pt5 <-> Pt2
        boundary_tags = [[1], [2], [3], [4]]
        # # Surface zone and Physicals Group
        gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.addPhysicalGroup(2, [1], 1)
        gmsh.model.setPhysicalName(2, 1, "DOMAINE")

        # Mesh
        if 'lc' not in mesh_parameters:
            n1 = mesh_parameters['N1']+1
            n2 = mesh_parameters['N2']+1
            gmsh.model.geo.mesh.setTransfiniteCurve(1, n2)
            gmsh.model.geo.mesh.setTransfiniteCurve(2, n1)
            gmsh.model.geo.mesh.setTransfiniteCurve(3, n2)
            gmsh.model.geo.mesh.setTransfiniteCurve(4, n1)
            gmsh.model.geo.mesh.setTransfiniteSurface(1, "Alternate")
        if mesh_parameters['mesh_type'] == "QUAD":
            gmsh.model.geo.mesh.setRecombine(2, 1)
        gmsh.option.setNumber("Mesh.Smoothing", 100)
        # # Generation of the mesh
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        # # Getting the nodes coordinates
        nodes_tag, node_coords, _ = gmsh.model.mesh.getNodes()
        # delete unused nodes
        # Getting the connectivity of the elements
        elem_types_domaine, _, elem_node_tags_domaine = gmsh.model.mesh.getElements(dim=2)
        max_tag = 0
        for i in range(len(elem_node_tags_domaine)):
            max_tag = max(max_tag, max(elem_node_tags_domaine[i]))
        index_to_delete = np.where(nodes_tag > max_tag)[0]
        ind1 = 3*index_to_delete
        ind2, ind3 = ind1 + 1, ind1 + 2
        index_to_delete = np.concatenate((ind1, ind2, ind3))
        node_coords = np.delete(node_coords, index_to_delete)
        # Getting the boundary elements
        elem_boundary = []
        for b_tags in boundary_tags:
            elem_bound = []
            for b_tag in b_tags:
                _, _, elem_node_tags = gmsh.model.mesh.getElements(dim=1, tag=b_tag)
                elem_bound.append(elem_node_tags[0])
            elem_bound = np.concatenate(elem_bound, dtype=int)
            elem_boundary.append(elem_bound)

        # Finalizing the Gmsh API
        gmsh.finalize()

        # Construction of the Mesh Object
        mesh_obj = self.generate_mesh_object(node_coords, elem_types_domaine, elem_node_tags_domaine, elem_boundary)

        return mesh_obj

    def generate_mesh_object(self, node_coords, elem_types_domaine, elem_node_tags_domaine, elem_boundary):
        """
        Construit un objet mesh.

        Parameters
        ----------
        node_coords : ndarray, shape (3*number_of_nodes,)
            Tableau contenant les coordonnées des noeuds du maillage.
        elem_types_domaine : list 
            Liste contenant le type des éléments composant le maillage. 
            - element type = 2 correspond à un triangle
            - element type = 3 correspond à un quadrilatère
        elem_node_tags_domaine : list of ndarray
            Liste de même taille que elem_types_domaine. Chaque item (elem_node_tags_domaine[i]) 
            est un tableau contenant les noeuds des éléments de type elem_types_domaine[i]
        elem_boundary : list
            Liste contenant les noeuds des faces frontières

        Returns
        -------
        Mesh
            Objet mesh construit
        """
        # Nodes
        node_coords = node_coords.astype(float)
        number_of_nodes = node_coords.shape[0]//3
        node_to_coordinates = node_coords.reshape(number_of_nodes, 3)
        node_to_xcoord = node_to_coordinates[:, 0]
        node_to_ycoord = node_to_coordinates[:, 1]
        # Elements
        # elem_type+1 => gives the number of nodes of the element
        number_of_elements = 0
        for i_type, elem_type in enumerate(elem_types_domaine):
            number_of_elements += elem_node_tags_domaine[i_type].shape[0]//(elem_type+1)
        element_to_nodes_start = np.zeros((number_of_elements+1,), dtype=int)
        i_elem, offset = 1, 0
        for i_type, elem_type in enumerate(elem_types_domaine):
            n_elem = elem_node_tags_domaine[i_type].shape[0]//(elem_type+1)
            element_to_nodes_start[i_elem:i_elem+n_elem] = offset + elem_type+1 + np.arange(0, elem_node_tags_domaine[i_type].shape[0], elem_type+1)
            i_elem += n_elem
            offset += elem_node_tags_domaine[i_type].shape[0]
        element_to_nodes = np.concatenate(elem_node_tags_domaine).astype(int)
        element_to_nodes = element_to_nodes - 1  # start numbering by zero
        # Boundary faces
        for i_boun in range(len(elem_boundary)):
            elem_boundary[i_boun] = elem_boundary[i_boun] - 1  # start numbering by zero

        # Mesh object
        mesh_obj = mesh.Mesh(node_to_xcoord, node_to_ycoord, element_to_nodes_start, element_to_nodes, elem_boundary)
        return mesh_obj
