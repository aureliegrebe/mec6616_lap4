# -*- coding: utf-8 -*-
"""
Exemple d'utilisation du module de maillage
MEC6616 - Aérodynamique numérique
Date de création: 2022 - 01 - 04
Auteur: El Hadji Abdou Aziz NDIAYE
"""

# IMPORT STATEMENTS
import sympy as sp
import numpy as np
import pyvista as pv
import pyvistaqt as pvQt
from meshGenerator import MeshGenerator
from meshConnectivity import MeshConnectivity
from meshPlotter import MeshPlotter

# PROGRAM


def pause():
    input('Press [return] to continue ...')


mesher = MeshGenerator()
plotter = MeshPlotter()

print('Rectangle : maillage non structuré avec des triangles')
mesh_parameters = {'mesh_type': 'TRI',
                   'lc': 0.5
                   }
mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters)
conec = MeshConnectivity(mesh_obj)
conec.compute_connectivity()
plotter.plot_mesh(mesh_obj, label_points=True, label_elements=True, label_faces=True)

print('Rectangle : maillage non structuré avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD',
                   'lc': 0.2
                   }
mesh_obj = mesher.rectangle([0.0, 2.0, 0.0, 1.0], mesh_parameters)
plotter.plot_mesh(mesh_obj)

print('Rectangle : maillage non structuré mix')
mesh_parameters = {'mesh_type': 'MIX',
                   'lc': 0.2
                   }
mesh_obj = mesher.rectangle([0.0, 2.0, 0.0, 1.0], mesh_parameters)
plotter.plot_mesh(mesh_obj)
pause()

print('Rectangle : maillage transfinis avec des triangles')
mesh_parameters = {'mesh_type': 'TRI',
                   'Nx': 5,
                   'Ny': 4
                   }
mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters)
plotter.plot_mesh(mesh_obj)

print('Rectangle : maillage transfinis avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD',
                   'Nx': 10,
                   'Ny': 15
                   }
mesh_obj = mesher.rectangle([0.0, 1.0, 1.0, 2.5], mesh_parameters)
plotter.plot_mesh(mesh_obj)

print('Rectangle : maillage transfinis mix')
mesh_parameters = {'mesh_type': 'MIX',
                   'Nx': 16,
                   'Ny': 10
                   }
mesh_obj = mesher.rectangle([0.0, 2.0, 0.0, 1.0], mesh_parameters)
plotter.plot_mesh(mesh_obj)
pause()

# Affichage de champ scalaire avec pyvista
x, y = sp.symbols('x y')
T = 10 + 4*sp.sin(np.pi*x/2) - 3*sp.cos(np.pi*y/2) + 2.5*sp.sin(np.pi*x*y/4)
fT = sp.lambdify([x, y], T, 'numpy')
cell_centers = np.zeros((mesh_obj.get_number_of_elements(), 2))
for i_element in range(mesh_obj.get_number_of_elements()):
    center_coords = np.array([0.0, 0.0])
    nodes = mesh_obj.get_element_to_nodes(i_element)
    for node in nodes:
        center_coords[0] += mesh_obj.get_node_to_xcoord(node)
        center_coords[1] += mesh_obj.get_node_to_ycoord(node)
    center_coords /= nodes.shape[0]
    cell_centers[i_element, :] = center_coords
nodes, elements = plotter.prepare_data_for_pyvista(mesh_obj)
pv_mesh = pv.PolyData(nodes, elements)
pv_mesh['Champ T'] = fT(cell_centers[:, 0], cell_centers[:, 1])

pl = pvQt.BackgroundPlotter()
# Tracé du champ
pl.add_mesh(pv_mesh, show_edges=True, scalars="Champ T", cmap="RdBu")
# Tracé des iso-lignes
nodes_xcoords = mesh_obj.get_nodes_to_xcoord()
nodes_ycoords = mesh_obj.get_nodes_to_ycoord()
pv_mesh['Champ T (noeuds)'] = fT(nodes_xcoords, nodes_ycoords)
contours = pv_mesh.contour(isosurfaces=15, scalars="Champ T (noeuds)")
pl.add_mesh(contours, color='k', show_scalar_bar=False, line_width=2)
pl.camera_position = 'xy'
pl.add_text('Champ scalaire T', position="upper_edge")
pl.show()
pause()

print('Back Step : maillage non structuré avec des triangles')
mesh_parameters = {'mesh_type': 'TRI',
                   'lc': 0.3
                   }
mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
plotter.plot_mesh(mesh_obj)

print('Back Step : maillage non structuré avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD',
                   'lc': 0.2
                   }
mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
plotter.plot_mesh(mesh_obj)

print('Back Step : maillage non structuré mix')
mesh_parameters = {'mesh_type': 'MIX',
                   'lc': 0.2
                   }
mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
plotter.plot_mesh(mesh_obj)
pause()

print('Back Step : maillage transfini avec des triangles')
mesh_parameters = {'mesh_type': 'TRI',
                   'Nx': 30,
                   'Ny': 25
                   }
mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
plotter.plot_mesh(mesh_obj)

print('Back Step : maillage transfini avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD',
                   'Nx': 30,
                   'Ny': 25
                   }
mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
plotter.plot_mesh(mesh_obj)

print('Back Step : maillage transfini mix')
mesh_parameters = {'mesh_type': 'MIX',
                   'Nx': 20,
                   'Ny': 25
                   }
mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
plotter.plot_mesh(mesh_obj)
pause()

print('Cercle : maillage non structuré avec des triangles')
mesh_parameters = {'mesh_type': 'TRI',
                   'lc_rectangle': 0.3,
                   'lc_circle': 0.1
                   }
rayon = 0.25
mesh_obj = mesher.circle([0.0, 2.0, 0.0, 1.0], rayon, mesh_parameters)
plotter.plot_mesh(mesh_obj)

print('Cercle : maillage non structuré avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD',
                   'lc_rectangle': 0.2,
                   'lc_circle': 0.05
                   }
rayon = 0.25
mesh_obj = mesher.circle([0.0, 2.0, 0.0, 1.0], rayon, mesh_parameters)
plotter.plot_mesh(mesh_obj)
pause()


print('Cercle : maillage transfini avec des triangles')
mesh_parameters = {'mesh_type': 'TRI',
                   'Nx': 25,
                   'Ny': 5,
                   'Nc': 20
                   }
rayon = 0.25
mesh_obj = mesher.circle([0.0, 2.0, 0.0, 1.0], rayon, mesh_parameters)
plotter.plot_mesh(mesh_obj)

print('Cercle : maillage transfini avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD',
                   'Nx': 60,
                   'Ny': 20,
                   'Nc': 60
                   }
rayon = 0.25
mesh_obj = mesher.circle([0.0, 5.0, 0.0, 3.0], rayon, mesh_parameters)
plotter.plot_mesh(mesh_obj)
pause()

print('Quart Anneau : maillage non structuré avec des triangles')
mesh_parameters = {'mesh_type': 'TRI', 'lc': 0.2}
mesh_obj = mesher.quarter_annular(1.0, 2.0, mesh_parameters)
plotter.plot_mesh(mesh_obj)

print('Quart Anneau : maillage non structuré avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD', 'lc': 0.1}
mesh_obj = mesher.quarter_annular(1.0, 2.0, mesh_parameters)
plotter.plot_mesh(mesh_obj)
pause()


print('Quart Anneau : maillage transfini avec des triangles')
mesh_parameters = {'mesh_type': 'TRI',
                   'N1': 50,
                   'N2': 10
                   }
mesh_obj = mesher.quarter_annular(2.0, 4.0, mesh_parameters)
plotter.plot_mesh(mesh_obj)

print('Quart Anneau : maillage transfini avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD',
                   'N1': 50,
                   'N2': 10
                   }
mesh_obj = mesher.quarter_annular(2.0, 3.0, mesh_parameters)
plotter.plot_mesh(mesh_obj)
pause()
