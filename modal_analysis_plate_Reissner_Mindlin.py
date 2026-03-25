import numpy as np
import ufl
import basix

from mpi4py import MPI
from dolfinx import fem, io
import dolfinx.fem.petsc
from dolfinx.fem import Constant, dirichletbc, Function, functionspace, locate_dofs_topological
import dolfinx.mesh
from dolfinx.io import XDMFFile


L  = 0.3
W_dim = 0.24
N = 10
domain = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, [[0, 0], [L, W_dim]], [N, N] , cell_type=dolfinx.mesh.CellType.triangle, diagonal=dolfinx.mesh.DiagonalType.crossed
)

# material parameters
thick = 0.00105
E = 210.0e3
nu = 0.3
rho = 2700
om = 8

# bending stiffness
D = fem.Constant(domain, E * thick**3 / (1 - nu**2) / 12.0)
# shear stiffness
F = fem.Constant(domain, E / 2 / (1 + nu) * thick * 5.0 / 6.0)
# Inertilal constant
M = fem.Constant(domain , rho * thick)


# uniform transversal load
f = fem.Constant(domain, 0.0)


### Definition des fonctions d'espace 

# Useful function for defining strains and stresses
def curvature(u):
    (w, theta) = ufl.split(u)
    return ufl.as_vector(
        [theta[0].dx(0), theta[1].dx(1), theta[0].dx(1) + theta[1].dx(0)]
    )


def shear_strain(u):
    (w, theta) = ufl.split(u)
    return ufl.grad(w) - theta


def bending_moment(u):
    DD = ufl.as_matrix([[D, nu * D, 0], [nu * D, D, 0], [0, 0, D * (1 - nu) / 2.0]])
    return ufl.dot(DD, curvature(u))

def inertial(u) :
    MM = ufl.as_matrix([[M, 0, 0], [0, M*thick**2/12, 0], [0, 0, M*thick**2/12]])
    return ufl.dot(MM, u)


def shear_force(u):
    return F * shear_strain(u)

# Definition of function space for U:displacement, T:rotation
Ue = basix.ufl.element("P", domain.basix_cell(), 2)
Te = basix.ufl.element("P", domain.basix_cell(), 1, shape=(2,))
V = fem.functionspace(domain, basix.ufl.mixed_element([Ue, Te]))

# Functions
u = fem.Function(V, name="Unknown")
u_ = ufl.TestFunction(V)
(w_, theta_) = ufl.split(u_)
du = ufl.TrialFunction(V)

# Linear and bilinear forms
dx = ufl.Measure("dx", domain=domain)
L = f * w_ * dx
a = (
    ufl.dot(bending_moment(u_), curvature(du))
    + ufl.dot(shear_force(u_), shear_strain(du))
    
) * dx

# Boundary of the plate
def border(x):
    return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 0))
    


facet_dim = 1
clamped_facets = dolfinx.mesh.locate_entities_boundary(domain, facet_dim, border)
clamped_dofs = fem.locate_dofs_topological(V, facet_dim, clamped_facets)

u0 = fem.Function(V)
bcs = []

from dolfinx.fem.petsc import LinearProblem

problem = LinearProblem(
    a, L, u=u, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="Plate"
)
problem.solve()

with io.VTKFile(domain.comm, "plates.xdmf", "w") as vtk:
    w = u.sub(0).collapse()
    w.name = "Deflection"
    vtk.write_function(w)


import pyvista
from dolfinx import plot

pyvista.set_jupyter_backend("static")

Vw = w.function_space
w_topology, w_cell_types, w_geometry = plot.vtk_mesh(Vw)
w_grid = pyvista.UnstructuredGrid(w_topology, w_cell_types, w_geometry)
w_grid.point_data["Deflection"] = w.x.array
w_grid.set_active_scalars("Deflection")
warped = w_grid.warp_by_scalar("Deflection", factor=5)

plotter = pyvista.Plotter()
plotter.add_mesh(
    warped,
    show_scalar_bar=True,
    scalars="Deflection",
)
edges = warped.extract_all_edges()
plotter.add_mesh(edges, color="k", line_width=1)
plotter.show()

theta = u.sub(1).collapse()
Vt = theta.function_space
theta_topology, theta_cell_types, theta_geometry = plot.vtk_mesh(Vt)
theta_grid = pyvista.UnstructuredGrid(theta_topology, theta_cell_types, theta_geometry)
beta_3D = np.zeros((theta_geometry.shape[0], 3))
beta_3D[:, :2] = theta.x.array.reshape(-1, 2) @ np.array([[0, -1], [1, 0]])
theta_grid["beta"] = beta_3D
theta_grid.set_active_vectors("beta")

plotter = pyvista.Plotter()
plotter.add_mesh(
    theta_grid.arrows, lighting=False, scalar_bar_args={"title": "Rotation Magnitude"}
)
plotter.add_mesh(theta_grid, color="grey", ambient=0.6, opacity=0.5, show_edges=True)
plotter.show()

### Analyse Modal

m_form =  ufl.dot(inertial(u_), du) * dx

K = fem.petsc.assemble_matrix(fem.form(a), bcs)
K.assemble()
M = fem.petsc.assemble_matrix(fem.form(m_form), bcs)
M.assemble()

# Conversion directe via une matrice intermédiaire dense PETSc
M_numpy = M.convert("dense").getDenseArray()

print(f"Forme NumPy : {M_numpy.shape}")


# https://fenicsproject.discourse.group/t/modal-analysis-using-dolfin-x/7349/8 
# Base du programme

from petsc4py import PETSc
from slepc4py import SLEPc

# Create and configure eigenvalue solver
N_eig = 40
eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
eigensolver.setDimensions(N_eig)
eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
st = SLEPc.ST().create(MPI.COMM_WORLD)
st.setType(SLEPc.ST.Type.SINVERT)
st.setShift(0.1)
st.setFromOptions()
eigensolver.setST(st)
eigensolver.setOperators(K, M)
eigensolver.setFromOptions()

# Compute eigenvalue-eigenvector pairs
eigensolver.solve()
evs = eigensolver.getConverged()
vr, vi = K.getVecs()
u_output = Function(V)
u_output.name = "Eigenvector"
evs = eigensolver.getConverged()
print(f"Nombre de modes convergés : {evs}")

# Configuration de la grille PyVista (2 lignes, 3 colonnes)
plotter = pyvista.Plotter(shape=(2, 3), window_size=[1000, 700])
plotter.set_background("white")

# On cherche à afficher 6 modes "utiles" (on saute les corps rigides)
modes_trouves = 0
for i in range(evs):
    if modes_trouves >= 6: break
    
    # Récupération de la valeur propre
    l = eigensolver.getEigenpair(i, vr, vi)
    freq = np.sqrt(max(0, l.real)) / (2 / np.pi) # f = sqrt(lambda)/2pi
    
    # On ignore les fréquences quasi-nulles (< 1Hz)
    if freq < 40.0: continue
    
    # Extraction de la déflexion w
    u_output.x.array[:] = vr.array
    w_mode = u_output.sub(0).collapse()
    
    # Création du maillage PyVista pour ce subplot
    topology, cell_types, geometry = plot.vtk_mesh(w_mode.function_space)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    
    # Normalisation pour l'affichage (max amplitude = 1)
    grid.point_data["Amplitude"] = w_mode.x.array / np.max(np.abs(w_mode.x.array))
    warped = grid.warp_by_scalar("Amplitude", factor=0.05) # Ajuste factor si besoin
    
    # Placement dans la grille
    plotter.subplot(modes_trouves // 3, modes_trouves % 3)
    plotter.add_text(f"Mode {i} : {freq:.1f} Hz", font_size=10, color="black")
    plotter.add_mesh(warped, cmap="viridis", show_scalar_bar=False)
    plotter.add_mesh(warped.extract_all_edges(), color="black", opacity=0.1)
    plotter.view_isometric()
    
    modes_trouves += 1

plotter.show()