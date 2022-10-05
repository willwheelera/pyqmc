import numpy as np
import pandas as pd
from kptpyqmc.api import recover_pyscf
from kptpyqmc.mc import initial_guess, vmc
from kptpyqmc.supercell import get_supercell_kpts, get_supercell
from kptpyqmc.slater import Slater
import kptpyqmc.orbitals
from pyscf.pbc import gto, scf
from pyscf.pbc.dft.multigrid import multigrid
from pyscf.scf.addons import remove_linear_dep_
import time



class KineticEnergyAccumulator:
    """Returns local energy of each configuration in a dictionary."""

    def __init__(self, mol, threshold=10, **kwargs):
        import kptpyqmc.energy as energy
        self.mol = mol 
        self.threshold = threshold
        if hasattr(mol, "a"):
            self.kinetic = energy.kinetic_pbc
        else:
            self.kinetic = energy.kinetic

    def __call__(self, configs, wf):
        ke, grad2 = self.kinetic(configs, wf) 
        return {
            "ke": ke, 
            "grad2": grad2,
        }

    def avg(self, configs, wf):
        return {k: np.mean(it, axis=0) for k, it in self(configs, wf).items()}

    def has_nonlocal_moves(self):
        return self.mol._ecp != {}

    def keys(self):
        return set(["ke", "grad2"])


if __name__ == "__main__":
    
    chkfile = "/home/will/Research/LiH/LiH_uks_nk8_0.1.chkfile"
    mol, mf = recover_pyscf(chkfile)
    S = (np.ones((3, 3)) - 2 * np.eye(3))
    runtest(mol, mf, S)
