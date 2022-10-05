import numpy as np
from pyqmc.coord import PeriodicConfigs, PeriodicElectron
from pyqmc.slater import Slater
import pyqmc.api as pyq
import sys
from kptpyqmc.coord import KptConfigs, KptElectron
from kptpyqmc.orbitals import PBCOrbitalEvaluatorKpoints
from kptpyqmc.slater import Slater as KptSlater

chkfile = "/home/will/Research/LiH/LiH_uks_nk8_0.1.chkfile"
#chkfile = "/home/will/Research/model_fitting/si_model_fitting/sipyscf/prim/si_pbe_prim/scf.py.chkfile"
mol, mf = pyq.recover_pyscf(chkfile)
mf = mf.to_uhf(mf)
S = np.ones((3, 3)) - 2 * np.eye(3)
supercell = pyq.get_supercell(mol, S)

wf = Slater(supercell, mf)
configs = pyq.initial_guess(supercell, 100)

kpt_configs = KptConfigs(configs.configs, mol.lattice_vectors(), wrap=np.dot(configs.wrap, S), nk=len(wf.orbitals._kpts))

twists = np.zeros((4, 3))
twists[1:] = (np.ones((3, 3)) - np.eye(3)) / 2
twists = twists[[0, 3, 1, 2]]
wfs = [Slater(mol, mf, twist=t) for t in twists]
phases, logvals = list(zip(*(wf.recompute(kpt_configs.select(ki)) for ki, wf in enumerate(wfs))))

kptcell = pyq.get_supercell(pyq.get_supercell(mol, np.eye(3)), S)
kptwf = KptSlater(kptcell, mf)
kphase, klogval = kptwf.recompute(kpt_configs)

pphase = np.prod(phases, axis=0)
plogval = np.sum(logvals, axis=0)

print("phase")
print("ratio-1", np.linalg.norm(kphase / pphase - 1))
print("logval")
print("diff", np.linalg.norm(klogval - plogval))


