# -*- coding: utf-8 -*-
"""
Exemple Solveur Simple Moyenne - Semaine 3
MEC6616 - Aérodynamique numérique
Date de création: 2022 - 01 - 21
Auteurs: El Hadji Abdou Aziz NDIAYE et Jean-Yves Trépanier
"""

# import sys
# sys.path.append('mesh_path')

import numpy as np
from meshGenerator import MeshGenerator
from meshConnectivity import MeshConnectivity
from mesh import Mesh

import pyvista as pv
import pyvistaqt as pvQt
from meshPlotter import MeshPlotter

def compute_moyenne(mesh_obj: Mesh, bcdata):
    """
    Fonction qui implémente un solveur simple moyenne

    Parameters
    ----------
    mesh_obj: Mesh
        Objet maillage.
    bcdata: liste de tuples
        Conditions aux limites

    Returns
    -------
    ndarray
        Moyenne du champ scalaire dans chaque élément du maillage.

    """
    number_of_elements = mesh_obj.get_number_of_elements()
    #initialisation de la matrice et du membre de droite
    A = np.zeros((number_of_elements,number_of_elements))
    B = np.zeros(number_of_elements)    
    
    # Boundary faces
    for i_face in range(mesh_obj.get_number_of_boundary_faces()):
        
        tag = mesh_obj.get_boundary_face_to_tag(i_face)
        bc_type = bcdata[tag][0]
        bc_value = bcdata[tag][1]        

        if ( bc_type == 'DIRICHLET' ) :
            element_left = mesh_obj.get_face_to_elements(i_face)[0]            
            A[element_left,element_left]+=2
            B[element_left]+=2*bc_value       
        
    # Internal faces
    for i_face in range(mesh_obj.get_number_of_boundary_faces(), mesh_obj.get_number_of_faces()):
        elements = mesh_obj.get_face_to_elements(i_face)

        A[elements[1],elements[1]]+=1
        A[elements[0],elements[0]]+=1
        A[elements[0],elements[1]]=-1
        A[elements[1],elements[0]]=-1
    
    moyenne = np.linalg.solve(A,B)    
    
    return moyenne

def test_moyenne():
    """
    Fonction pour tester le calcul de la divergence d'un champ constant.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    
    mesher = MeshGenerator()
    plotter = MeshPlotter()
    
    print("Test du solveur simple moyenne ...\n")
    
    # Quart d'anneau
    mesh_parameters = {'mesh_type': 'QUAD', 'lc': 0.1}
    mesh_obj = mesher.quarter_annular(1.0, 2.0, mesh_parameters)
    conec = MeshConnectivity(mesh_obj, verbose=False)
    conec.compute_connectivity()
    
    print("\nTest 1: 4 conditions de Dirichlet \n")
    bcdata = (['DIRICHLET', 0 ], ['DIRICHLET', 1 ],
              ['DIRICHLET', 2 ], ['DIRICHLET', 3])
    
    moyenne = compute_moyenne(mesh_obj, bcdata)
    
    # Affichage de champ scalaire avec pyvista
    nodes, elements = plotter.prepare_data_for_pyvista(mesh_obj)
    pv_mesh = pv.PolyData(nodes, elements)
    pv_mesh['Champ Moyenne'] = moyenne

    pl = pvQt.BackgroundPlotter()
    # Tracé du champ
    print("\nVoir le champ moyenne dans la fenêtre de PyVista \n")
    pl.add_mesh(pv_mesh, show_edges=True, scalars="Champ Moyenne", cmap="RdBu")
    
    return

def main():
    """Fonction principale"""

    # Test du solveur simple moyenne
    test_moyenne()

    return

if __name__ == '__main__':
    main()
