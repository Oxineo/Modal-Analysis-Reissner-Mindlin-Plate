#%%
import scipy
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata



map = scipy.io.loadmat('/home/adm-discohbot/Documents/Stage_Recherche_M2_Arthur/Modal-Analysis-Reissner-Mindlin-Plate/map_chaleur/central_droit_manip_21_04_2026.mat')

frame = np.array(map['Frame'])

#%%
import matplotlib
matplotlib.use('Qt5Agg')


plt.figure(num = "Map Chaleur")

plt.imshow(frame, cmap='jet')
plt.colorbar()
plt.title("Carte de Chaleur")
plt.xlabel("X")
plt.ylabel("Y")
plt.show(block = False)
#%%
p_00_plaque = np.array([155,38])
p_10_plaque = np.array([156,417])
p_11_plaque = np.array([468,374])
p_01_plaque = np.array([463,55])


### Rognage de l'image pour n'afficher que la plaque

def var_plaque(ksi) :
    x = (1-ksi[0])*(1-ksi[1])*p_00_plaque + ksi[0]*(1-ksi[1])*p_10_plaque + ksi[0]*ksi[1]*p_11_plaque + (1-ksi[0])*ksi[1]*p_01_plaque
    return x

def rognage(frame, n) :
    x = np.zeros((n,n,2))
    for i in range(n) :
        for j in range(n) :
            ksi = np.array([i/(n-1),j/(n-1)])
            x[i,j] = var_plaque(ksi)
    print(x.shape)
    points = x.reshape(-1,2)
    print(points.shape)
    values = frame[points[:,1].astype(int), points[:,0].astype(int)]
    grid_x, grid_y = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    grid_z = griddata(points, values, (grid_x, grid_y) )
    return grid_z

def carte_chaleur(frame , nx , ny) :
    x = np.zeros((nx,ny,2))
    for i in range(nx) :
        for j in range(ny) :
            ksi = np.array([i/(nx-1),j/(ny-1)])
            x[i,j] = var_plaque(ksi)
    values = frame[x[:,:,1].astype(int), x[:,:,0].astype(int)]
    return values

#%%
nx = 600
ny = 600

frame_rognee = carte_chaleur(frame, nx, ny)
plt.figure(num = "Map Chaleur Rognée")
plt.imshow(frame_rognee, cmap='hot')
plt.colorbar()
plt.title("Carte de Chaleur Rognée")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# %%

#np.save('/home/adm-discohbot/Documents/Stage_Recherche_M2_Arthur/Modal-Analysis-Reissner-Mindlin-Plate/map_chaleur/central_bas_2.npy', frame_rognee)

