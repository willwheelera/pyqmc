import numpy as np
from pyqmc.coord import PeriodicConfigs, PeriodicElectron
from pyqmc.slater import Slater
import pyqmc.api as pyq
import sys
from kptpyqmc.coord import KptConfigs, KptElectron
from kptpyqmc.orbitals import PBCOrbitalEvaluatorKpoints

chkfile = "/home/will/Research/LiH/LiH_uks_nk8_0.1.chkfile"
#chkfile = "/home/will/Research/model_fitting/si_model_fitting/sipyscf/prim/si_pbe_prim/scf.py.chkfile"
mol, mf = pyq.recover_pyscf(chkfile)
mf = mf.to_uhf(mf)
S = np.ones((3, 3)) - 2 * np.eye(3)
supercell = pyq.get_supercell(mol, S)
wf = Slater(supercell, mf)
configs = pyq.initial_guess(supercell, 100)
mask = np.random.random(configs.configs.shape[0]) > .6

eval_str = "GTOval_sph_deriv2"
aos_sup = wf.orbitals.aos(eval_str, configs, mask=mask)
shape = list(aos_sup.shape)
shape[-2] = mask.sum()
shape.insert(-1, configs.configs.shape[1])
aos_sup = aos_sup.reshape(shape)


kpt_configs = KptConfigs(configs.configs, mol.lattice_vectors(), wrap=np.dot(configs.wrap, S), nk=len(wf.orbitals._kpts))
kptcell = pyq.get_supercell(pyq.get_supercell(mol, np.eye(3)), S)
kpt_orbs = PBCOrbitalEvaluatorKpoints.from_mean_field(kptcell, mf)[-1]
aos_kpt = kpt_orbs.aos(eval_str, kpt_configs, mask=mask)

print(aos_sup.shape)
print(len(aos_kpt), aos_kpt[0].shape)

for i in range(len(wf.orbitals._kpts)):
    select = kpt_configs.k_indices == i
    n = np.linalg.norm(aos_kpt[i] - aos_sup[i][..., select, :])
    print("k ind", i, "diff", n)
    assert n < 1e-8
