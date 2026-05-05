import numpy as np
import ufl
import basix
from mpi4py import MPI
from dolfinx import fem, mesh
from slepc4py import SLEPc
from dolfinx.fem.petsc import assemble_matrix
import scipy

# ==============================================================================
# 1. INITIALISATION GLOBALE 
# ==============================================================================
L, W_dim = 298, 242
Nx, Ny = 100, 100 # Conseil : réduisez le maillage (ex: 50x50 ou 100x100) pour le problème inverse

domain = mesh.create_rectangle(
    MPI.COMM_WORLD, [[0, 0], [L, W_dim]], [Nx, Ny], cell_type=mesh.CellType.quadrilateral
)

# Paramètres à optimiser définis comme des Constantes FEniCS (Modifiables à la volée)
thick = fem.Constant(domain, 1.15)
E_const = fem.Constant(domain, 200e9) 
nu_const = fem.Constant(domain, 0.30)

rho = 0.665 * 1e-3/L/W_dim/thick

# Constantes de raideur et masse
A = E_const * thick / (1 - nu_const**2)
D = E_const * thick**3 / (1 - nu_const**2) / 12.0
F = E_const / 2 / (1 + nu_const) * thick * 5.0 / 6.0
M_rho = rho * thick

# Espaces de fonctions (Plaque de Reissner-Mindlin)
Pe = basix.ufl.element("S", domain.basix_cell(), 2, shape=(2,))
Ue = basix.ufl.element("S", domain.basix_cell(), 2)
Te = basix.ufl.element("S", domain.basix_cell(), 2, shape=(2,))
V = fem.functionspace(domain, basix.ufl.mixed_element([Pe, Ue, Te]))

u_ = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

# Fonctions cinématiques
def membrane_strain(u):
    psi, w, theta = ufl.split(u)
    return ufl.as_vector([psi[0].dx(0), psi[1].dx(1), psi[0].dx(1) + psi[1].dx(0)])

def curvature(u):
    psi, w, theta = ufl.split(u)
    return ufl.as_vector([theta[0].dx(0), theta[1].dx(1), theta[0].dx(1) + theta[1].dx(0)])

def shear_strain(u):
    psi, w, theta = ufl.split(u)
    return ufl.as_vector([w.dx(0) - theta[0], w.dx(1) - theta[1]])

# Formes bilinéaires (Raideur K et Masse M)
AA = ufl.as_matrix([[A, nu_const * A, 0], [nu_const * A, A, 0], [0, 0, A * (1 - nu_const) / 2.0]])
DD = ufl.as_matrix([[D, nu_const * D, 0], [nu_const * D, D, 0], [0, 0, D * (1 - nu_const) / 2.0]])
MM = ufl.as_matrix([
    [M_rho, 0, 0, 0, 0], 
    [0, M_rho, 0, 0, 0], 
    [0, 0, M_rho, 0, 0], 
    [0, 0, 0, M_rho*thick**2/12, 0], 
    [0, 0, 0, 0, M_rho*thick**2/12]
])

k_form = (
    ufl.inner(ufl.dot(AA, membrane_strain(du)), membrane_strain(u_)) +
    ufl.inner(ufl.dot(DD, curvature(du)), curvature(u_)) +
    ufl.inner(F * shear_strain(du), shear_strain(u_))
) * ufl.Measure("dx", domain=domain)

m_form = ufl.inner(ufl.dot(MM, du), u_) * ufl.Measure("dx", domain=domain)

# Pré-compilation des formulaires
form_K = fem.form(k_form)
form_M = fem.form(m_form)

# ==============================================================================
# 2. LA FONCTION CIBLE 
# ==============================================================================
def frequences_propres(params, nb_modes=15, freq_min_cutoff=10.0):
    """
    Met à jour les paramètres matériaux, assemble les matrices et calcule les modes.
    """
    # 1. Mise à jour instantanée des propriétés matérielles
    E_const.value = params[0]
    nu_const.value = params[1]

    # 2. Assemblage des matrices
    K = assemble_matrix(form_K, bcs=[])
    K.assemble()
    M_mat = assemble_matrix(form_M, bcs=[])
    M_mat.assemble()
    # 3. Configuration du solveur SLEPc
    eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
    eigensolver.setDimensions(nb_modes)
    eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    
    st = SLEPc.ST().create(MPI.COMM_WORLD)
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(1.0) # Léger shift pour éviter la singularité absolue à 0 Hz
    eigensolver.setST(st)
    eigensolver.setOperators(K, M_mat)
    
    # 4. Résolution
    eigensolver.solve()
    evs = eigensolver.getConverged()
    
    # 5. Extraction des fréquences
    frequences_calculees = []
    for i in range(evs):
        l = eigensolver.getEigenvalue(i)
        freq = np.sqrt(max(0, l.real)) / (2 * np.pi)
        
        # On ignore les modes de corps rigide (fréquences ~ 0 Hz)
        if freq > freq_min_cutoff: 
            frequences_calculees.append(freq)
            
    # Nettoyage mémoire PETSc
    K.destroy()
    M_mat.destroy()
    eigensolver.destroy()
            
    return np.array(frequences_calculees)

import scipy.optimize as opt
import numpy as np

# ==============================================================================
# 3. CRÉATION DE LA FONCTION COÛT (L'arbitre de l'optimisation)
# ==============================================================================
save = {}

def fonction_cout(x, freq_cibles, nb_modes, freq_min):
    """
    x[0] = Module d'Young E (en GPa pour l'optimiseur)
    x[1] = Coefficient de Poisson nu
    """
    global save

    E_test_gpa = x[0] 
    nu_test = x[1] 
    
    # 1. Barrières de sécurité physiques (Si l'optimiseur part dans le décor)
    if nu_test <= 0.0 or nu_test >= 0.5 or E_test_gpa <= 0.0:
        save[(E_test_gpa , nu_test)] = 1e9
        return 1e9 # On renvoie une erreur immense pour le repousser

    # 2. Conversion en vraies unités physiques pour votre modèle FEniCS
    E_vrai = E_test_gpa * 1e3
    
    # 3. On lance votre modèle allégé
    freqs_calc = frequences_propres((E_vrai, nu_test), nb_modes=nb_modes, freq_min_cutoff=freq_min)
    
    # 4. On compare (Méthode des moindres carrés)
    # On s'assure de comparer le même nombre de modes (au cas où le modèle en calcule moins)
    n = min(len(freqs_calc), len(freq_cibles))
    
    if n == 0:
        return 1e9 # Pénalité si le solveur a échoué
        
    # Calcul de l'erreur : Somme des (Fréquence_Calculée - Fréquence_Réelle)²

    erreur = np.sum(((freqs_calc[:n] - freq_cibles[:n])/freq_cibles[:n])**2) * 10000.0

    # Ajout d'une petite pénalité s'il manque des modes calculés
    erreur += (len(freq_cibles) - n) * 1000 
    
    # Affichage en temps réel pour voir l'optimiseur travailler !
    print(f"Test E = {E_test_gpa:.2f} GPa | nu = {nu_test:.4f} ---> Erreur : {erreur:.4f} ")
    
    save[(E_test_gpa , nu_test)] = erreur

    return erreur


import matplotlib
matplotlib.use('TkAgg') # Ou 'Qt5Agg'
import matplotlib.pyplot as plt
#%%
def aff_dico(save_dic):
    xy = save_dic.keys()
    x = [i[0]  for i in xy ]
    y = [i[1]  for i in xy ]
    z = save_dic.values()

    fig = plt.figure()
    ax_3D = fig.add_subplot(111, projection = "3d")
    ax_3D.scatter(x,y,z,marker="^")

    ax_3D.set_xlim(180,210)
    ax_3D.set_ylim(0.2,0.4)
    ax_3D.set_zlim(0,50)

    ax_3D.set_xlabel('E : module de Young')
    ax_3D.set_ylabel('nu : module de Poisson')
    ax_3D.set_zlabel('Erreur')

    fig.show()
#%%

if __name__ == "__main__":

    freq_real = np.array([50.0,66.0 ,110.5, 123.0, 145.5, 195.0, 240.5, 249.5, 293.0, 332.0 , 375.5, 386.5, 416.5, 449.0])#, 582.5, 591.0, 598.0, 632.0, 669.5, 721.5, 798.5])

    # Le point de départ normalisé (x0 = 0.5 donne E=200 GPa, nu=0.30)
    x0_initial = np.array([200.0, 0.30])
    
    print("Début du problème inverse...")

    resultat = opt.minimize(
        fonction_cout, 
        x0_initial, 
        args=(freq_real, 30, 15.0),
        method='Powell',
        options={'disp': True}
    )
    
    print("\nOPTI")
    print("="*40)
    
    E_final_gpa = resultat.x[0] 
    nu_final = resultat.x[1] 
    
    print(f"Module d'Young E optimal : {E_final_gpa:.4f} GPa")
    print(f"Coef de Poisson nu optimal : {nu_final:.6f}")
    print("="*40)

    print("Fermez la fenêtre 3D pour terminer le programme.")
    plt.ioff()
    plt.show()
    