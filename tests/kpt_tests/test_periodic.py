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
import uuid


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


def runtest(mol, mf, S):
    kinds = kptpyqmc.orbitals.get_k_indices(mol, mf, get_supercell_kpts(mol.lattice_vectors(), S))
    kpts = mf.kpts[kinds]
    dm = mf.make_rdm1()
    print("original dm shape", dm.shape)
    if len(dm.shape) == 4:
        dm = np.sum(dm, axis=0)
    dm = dm[kinds]

    #####################################
    ## evaluate KE in PySCF
    #####################################
    ke_mat = mol.pbc_intor("int1e_kin", hermi=1, kpts=np.array(kpts))
    ke_mat = np.asarray(ke_mat)
    print("ke_mat", ke_mat.shape)
    print("dm", dm.shape)
    pyscfke = np.real(np.einsum("kij,kji->", ke_mat, dm))
    print("PySCF kinetic energy: {0}".format(pyscfke))

    #####################################
    ## evaluate KE integral with VMC
    #####################################
    kptcell = get_supercell(get_supercell(mol, np.eye(3)), S)
    wf = Slater(kptcell, mf)
    coords = initial_guess(mol, 120, 0.7, nk=len(kinds))
    warmup = 10
    start = time.time()
    ke_acc = KineticEnergyAccumulator(mol)
    df, coords = vmc(
        wf,
        coords,
        nsteps=100 + warmup,
        tstep=1,
        accumulators={"energy": ke_acc},
        verbose=False,
    )
    print("VMC time", time.time() - start)

    df = pd.DataFrame(df)
    dfke = pyq.avg_reblock(df["energyke"][warmup:], 10)
    vmcke, err = dfke.mean(), dfke.sem()
    print("VMC kinetic energy: {0} +- {1}".format(vmcke, err))

    assert (
        np.abs(vmcke - pyscfke) < 5 * err
    ), "energy diff not within 5 sigma ({0:.6f}): energies \n{1} \n{2}".format(
        5 * err, vmcke, pyscfke
    )


if __name__ == "__main__":
    
    chkfile = "/home/will/Research/LiH/LiH_uks_nk8_0.1.chkfile"
    mol, mf = recover_pyscf(chkfile)
    S = (np.ones((3, 3)) - 2 * np.eye(3))
    runtest(mol, mf, S)
