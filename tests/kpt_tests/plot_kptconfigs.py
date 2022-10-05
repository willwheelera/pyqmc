import numpy as np
import matplotlib.pyplot as plt
from kptpyqmc.coord import KptConfigs
from kptpyqmc.mc import initial_guess
import pyqmc.api as pyq


if __name__ == "__main__":
    
    chkfile = "/home/will/Research/LiH/LiH_uks_nk8_0.1.chkfile"
    mol, mf = pyq.recover_pyscf(chkfile)
    mf = mf.to_uhf(mf)
    L = mol.lattice_vectors()[:2, :2]
    print(L)
    #S = np.ones((3, 3)) - 2 * np.eye(3)
    S = np.diag([2, 2, 1])
    nk = int(np.around(np.linalg.det(S)))

    atoms = mol.atom_coords()[:, :2]
    plt.scatter(atoms[:, 0], atoms[:, 1], c="k")
    configs = initial_guess(mol, 1, nk=nk, r=0.1)
    for s, n in enumerate(mol.nelec):
        print(s, n)
        ntot = n * nk
        kcoords = np.array_split(configs.configs, nk, axis=1)
        for kc in kcoords:
            plt.scatter(kc[0, :, 0], kc[0, :, 1], marker=["^", "v"][s])
    plt.plot(0, 0, *L[0])
    plt.plot(0, 0, *L[1])
    plt.show()
    
