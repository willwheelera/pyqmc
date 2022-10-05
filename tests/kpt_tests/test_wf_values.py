import numpy as np
import kptpyqmc.testwf as testwf
from kptpyqmc.gpu import cp, asnumpy
from kptpyqmc.slater import Slater
from kptpyqmc.multiplywf import MultiplyWF
from kptpyqmc.manybody_jastrow import J3
from kptpyqmc.wftools import generate_jastrow
from kptpyqmc.coord import KptConfigs
import kptpyqmc.api as pyq


def run_tests(wf, epos, epsilon):

    #_, epos = pyq.vmc(wf, epos, nblocks=1, nsteps=2, tstep=1)  # move off node

    d = testwf.test_updateinternals(wf, epos)
    for k, item in d.items():
        print(k, item)
    for k, item in d.items():
        assert item < epsilon

    testwf.test_mask(wf, 0, epos.electron(1))

    for fname, func in zip(
        ["gradient", "laplacian", "pgradient"][:2],
        [
            testwf.test_wf_gradient,
            testwf.test_wf_laplacian,
            #testwf.test_wf_pgradient,
        ],
    ):
        err = [func(wf, epos, delta) for delta in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]]
        assert min(err) < epsilon, "epsilon {0}".format(epsilon)

    for fname, func in zip(
        ["gradient_value", "gradient_laplacian"],
        [
            testwf.test_wf_gradient_value,
            testwf.test_wf_gradient_laplacian,
        ],
    ):
        d = func(wf, epos)
        for k, v in d.items():
            assert v < 1e-10, (k, v)


def test_pbc_wfs(epsilon=1e-5, nconf=10):
    """
    Ensure that the wave function objects are consistent in several situations.
    """
    chkfile = "/home/will/Research/LiH/LiH_uks_nk8_0.1.chkfile"
    mol, mf = pyq.recover_pyscf(chkfile) #H_pbc_sto3g_krks

    supercell = pyq.get_supercell(mol, S=np.eye(3))
    supercell = pyq.get_supercell(supercell, S=(np.ones((3, 3)) - 2 * np.eye(3)))
    epos = pyq.initial_guess(supercell, nconf)
    epos = KptConfigs(epos.configs, mol.lattice_vectors(), wrap=epos.wrap, nk=4)
    
    for wf in [
        #MultiplyWF(Slater(supercell, mf), generate_jastrow(supercell)[0]),
        Slater(supercell, mf),
    ]:
        for k in wf.parameters:
            if "mo_coeff" not in k and k != "det_coeff":
                wf.parameters[k] = cp.asarray(np.random.rand(*wf.parameters[k].shape))

        #_, epos = pyq.vmc(wf, epos, nblocks=1, nsteps=2, tstep=1)  # move off node
        run_tests(wf, epos, epsilon)



def test_manual_slater(H2_ccecp_rhf, epsilon=1e-5):
    mol, mf = H2_ccecp_rhf

    determinants = [(1.0, [[0], [0]]), (-0.2, [[1], [1]])]
    wf = Slater(mol, mf, determinants=determinants)
    configs = pyq.initial_guess(mol, 10)
    run_tests(wf, configs, epsilon)


def test_manual_pbcs_fail(H_pbc_sto3g_krks, epsilon=1e-5, nconf=10):
    """
    This test makes sure that the number of k-points must match the number of k-points
    in the mf object.
    """
    mol, mf = H_pbc_sto3g_krks
    supercell = np.identity(3, dtype=int)
    supercell[0, 0] = 2
    mol = pyq.get_supercell(mol, supercell)
    try:
        determinants = [
            (1.0, [[0, 1], [0, 1]], [[0, 1], [0, 1]]),  # first determinant
            (-0.2, [[0, 2], [0, 1]], [[0, 2], [0, 1]]),  # second determinant
        ]
        wf = Slater(mol, mf, determinants=determinants)
        raise Exception("Should have failed here")
    except:
        pass


def test_manual_pbcs_correct(H_pbc_sto3g_kuks, epsilon=1e-5, nconf=10):
    """
    This test makes sure that the number of k-points must match the number of k-points
    in the mf object.
    """
    from pyqmc.determinant_tools import create_pbc_determinant

    mol, mf = H_pbc_sto3g_kuks
    supercell = np.identity(3, dtype=int)
    supercell[0, 0] = 2
    mol = pyq.get_supercell(mol, supercell)

    determinants = [
        (1.0, create_pbc_determinant(mol, mf, [])),
        (-0.2, create_pbc_determinant(mol, mf, [(0, 0, 0, 0, 1)])),
    ]
    wf = Slater(mol, mf, determinants=determinants)
    configs = pyq.initial_guess(mol, 10)
    run_tests(wf, configs, epsilon)


if __name__ == "__main__":
    test_pbc_wfs(nconf=8)
