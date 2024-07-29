import pyscf.gto
import numpy as np
from pyqmc.pbc import enforce_pbc
import pyqmc.gto as gto
import pyqmc.pbcgto as pbcgto

def _gradient(fval, fgrad, mol=None, delta=1e-5):
    rvec = np.random.randn(500, 3) * 3
    grad = fgrad(rvec)
    val0 = fval(rvec)
    assert np.amax(np.abs(grad[0] - val0)) < 1e-6
    numeric = np.zeros(grad[1:].shape)
    for d in range(3):
        pos = rvec.copy()
        pos[..., d] += delta
        plusval = fval(pos)
        pos[..., d] -= 2 * delta
        minuval = fval(pos)
        numeric[d, ...] = (plusval - minuval) / (2 * delta)
    maxerror = np.max(np.abs(grad[1:] - numeric))
    return (maxerror)


def _laplacian(fgrad, flap, mol=None, delta=1e-5):
    rvec = np.asarray(np.random.randn(500, 3)) 
    lap = flap(rvec)
    grad0 = fgrad(rvec)
    assert np.amax(np.abs(lap[:4] - grad0[:4])) < 1e-6
    numeric = np.zeros(lap.shape[1:])
    for d in range(3):
        pos = rvec.copy()
        pos[..., d] += delta
        plusgrad = fgrad(pos)[1:]
        pos[..., d] -= 2 * delta
        minugrad = fgrad(pos)[1:]
        numeric += (plusgrad[d] - minugrad[d]) / (2 * delta)
    maxerror = np.max(np.abs(lap[4] - numeric))
    return (maxerror)


def test_spherical_radial_funcs():
    coeffs = []
    coeffs.append(np.array([[0.2, 1.]]))
    coeffs.append(np.array([[1.7, 1.]]))
    coeffs.append(np.array([
        [84.322332, 2e-06],
        [44.203528, 0.004103],
        [23.288963, -0.04684],
        [13.385163, 0.052833],
        [7.518052, 0.218094],
        [4.101835, -0.044999],
        [2.253571, -0.287386],
        [1.134924, -0.71322],
        [0.56155, 0.249174],
        [0.201961, 1.299872],
        [0.108698, 0.192119],
        [0.053619, -0.658616],
        [0.025823, -0.521047],
    ]))
    coeffs.append(np.array([
        [152.736742, 5.1e-05],
        [50.772485, 0.008769],
        [26.253589, 0.047921],
        [12.137022, 0.132475],
        [5.853719, 0.297279],
        [2.856224, 0.275018],
        [1.386132, -0.222842],
        [0.670802, -0.619067],
        [0.33028, -0.07629],
        [0.170907, 0.256056],
        [0.086794, 0.541482],
    ]))
    sph_funcs = [lambda x: gto.eval_spherical(max_l, x).sum(axis=0) for max_l in [2, 3, 4, 5]]
    sph_grads = [lambda x: gto.eval_spherical_grad(max_l, x).sum(axis=0) for max_l in [2, 3, 4, 5]]
    rad_funcs = [lambda x: gto.radial_gto(np.sum(x**2, axis=-1), c) for c in coeffs]
    rad_grads = [lambda x: gto.radial_gto_grad(np.sum(x**2, axis=-1), x, c) for c in coeffs]
    rad_laps = [lambda x: gto.radial_gto_lap(np.sum(x**2, axis=-1), x, c) for c in coeffs]

    tol = 3e-5
    print("spherical")
    for sval, sgrad in zip(sph_funcs, sph_grads):
        gerr = _gradient(sval, sgrad, 1e-5)
        print("grad", gerr)
        assert gerr < tol
    print("radial")
    for rval, rgrad, rlap in zip(rad_funcs, rad_grads, rad_laps):
        gerr = _gradient(rval, rgrad, 1e-5)
        print("grad", gerr)
        lerr = _laplacian(rgrad, rlap, 1e-5)
        print("lap", lerr)
        assert gerr < tol
        assert lerr < tol

def test_mol():
    mol = pyscf.gto.M(atom="Mn 0. 0. 0.; N 0. 0. 2.5", ecp="ccecp", basis="ccecp-ccpvtz", unit="B", spin=0)
    orbitals = gto.AtomicOrbitalEvaluator(mol)
    orbval = lambda x: orbitals.eval_gto("GTOval_sph", (x))
    orbgrad = lambda x: orbitals.eval_gto("GTOval_sph_deriv1", (x))
    orblap = lambda x: orbitals.eval_gto("GTOval_sph_deriv2", (x))

    graderr = _gradient(orbval, orbgrad, mol)
    laperr = _laplacian(orbgrad, orblap, mol)
    print("mol orbitals")
    print("gradient", graderr)
    print("laplacian", laperr)
    orbval, pyscfval, _ = mol_against_pyscf(mol, orbitals)
    err = np.amax(np.abs(orbval-pyscfval))
    tol = 3e-5
    assert err < tol, err
    assert graderr < tol, graderr
    assert laperr < tol, laperr

def test_pbc():
    mol = pyscf.pbc.gto.M(atom="Li 0. 0. 0.; Li 3.3 3.3 3.3", basis="ccecp-ccpvdz", unit="B", a=np.eye(3)*2*3.3)
    mol.precision = 1e-8
    mol.build()
    kpts = np.pi * np.mgrid[:2,:2,:2].reshape(3, -1).T @ np.linalg.inv(mol.lattice_vectors())
    orbitals = pbcgto.PeriodicAtomicOrbitalEvaluator(mol, eval_gto_precision=1e-5, kpts=kpts)
    orbval = lambda x: orbitals.eval_gto("GTOval_sph", x)
    orbgrad = lambda x: orbitals.eval_gto("GTOval_sph_deriv1", x).swapaxes(0, 1)
    orblap = lambda x: orbitals.eval_gto("GTOval_sph_deriv2", x).swapaxes(0, 1)

    graderr = _gradient(orbval, orbgrad, mol)
    laperr = _laplacian(orbgrad, orblap, mol)
    print("pbc orbitals")
    print("gradient", graderr)
    print("laplacian", laperr)
    orbval, pyscfval, _ = pbc_against_pyscf(mol, orbitals)
    err = np.amax(np.abs(orbval-pyscfval))

    tol = 3e-5
    assert err < tol, err
    assert graderr < tol, graderr
    assert laperr < tol, laperr


def mol_against_pyscf(mol, orbitals, N=100, eval_str="GTOval_sph_deriv2"):
    coords = np.zeros((N, 3))
    coords[:, 0] = np.linspace(-0, 7, N)
    coords[:, 1] = np.linspace(-0, 7, N)
    coords[:, 2] = np.linspace(-.0, 7, N)
    
    pyscfval = mol.eval_gto(eval_str, coords)
    if len(pyscfval) == 10:
        pyscfval[4] = pyscfval[[4, 7, 9]].sum(axis=0)
        pyscfval = pyscfval[:5]
    orbval = orbitals.eval_gto(eval_str, coords)
    return orbval, pyscfval, coords
    #diff = orbval - pyscfval

    #return np.amax(np.abs(diff))
    
def pbc_against_pyscf(cell, orbitals, N=100, eval_str="GTOval_sph_deriv2"):
    #coords, _ = enforce_pbc(cell.lattice_vectors(), np.random.randn(500, 3) * 3)
    #coords = np.mgrid[4.9:5.1:3j, 4.9:5.1:3j, 4.9:5.1:3j].reshape(3, -1).T / 10
    coords = np.zeros((N, 3))
    coords[:, 0] = np.linspace(0, 1, N)
    coords[:, 1] = np.linspace(0, 1, N)
    coords[:, 2] = np.linspace(0, 1, N)
    coords = coords @ cell.lattice_vectors()
    pyscfval = cell.eval_gto("PBC"+eval_str, coords, kpts=orbitals.kpts)
    if len(pyscfval[0]) == 10:
        for p in pyscfval:
            p[4] = p[[4, 7, 9]].sum(axis=0)
        pyscfval = [p[:5] for p in pyscfval]
    orbval = orbitals.eval_gto(eval_str, coords)
    return orbval, pyscfval, coords
    

if __name__ == "__main__":
    # Plot
    import matplotlib.pyplot as plt
    #mol = pyscf.pbc.gto.M(atom="Li 0. 0. 0.; Li 3.3 3.3 3.3", basis="ccecp-ccpvdz", unit="B", a=np.eye(3)*2*3.3)
    mol = pyscf.pbc.gto.M(atom="N 0. 0. 0.; N 3.3 3.3 3.3", basis="ccecp-ccpvdz", unit="B", a=np.eye(3)*2*3.3)
    #mol = pyscf.gto.M(atom="Na 0. 0. 0.; Na 3.3 3.3 3.3", basis="ccecp-ccpvdz", unit="B")
    mol.precision = 1e-12
    mol.build()
    for at, basis in mol._basis.items():
        print(at)
        for bas in basis:
            print(bas[0])
            for b in bas[1:]:
                    print(b)

    labels = mol.ao_labels()
    kpts = np.pi * np.mgrid[:2,:2,:2].reshape(3, -1).T @ np.linalg.inv(mol.lattice_vectors())
    orbitals = pbcgto.PeriodicAtomicOrbitalEvaluator(mol, eval_gto_precision=1e-11, kpts=kpts)
    #orbitals = gto.AtomicOrbitalEvaluator(mol)#, eval_gto_precision=1e-8)

    orbval, pyscfval, coords = pbc_against_pyscf(mol, orbitals)
    #orbval = orbval[np.newaxis]
    #pyscfval = pyscfval[np.newaxis]

    diff = np.real(orbval - pyscfval)
    
    print(diff.shape)

    print("max diff", np.amax(np.abs(diff)))
    print("norm", np.linalg.norm((diff)))
    print("sum diff", np.sum((diff), axis=(0, 2, 3)))
    print("sum abs diff", np.sum(np.abs(diff), axis=(0, 2, 3)))
    
    ao = 12
    d = 0
    #for k in range(0, 1):
        #plt.plot(pyscfval[k][d, :, ao], label=f"pyscf{k}")
        #plt.plot(orbval[k, d, :, ao], ls=":", label=f"orb{k}")
    k = 0
    ratio = np.zeros_like(orbval[k, d])
    ind = np.where(pyscfval[k][d] > 1e-10)
    print("inds")
    for _i in ind:
        print(_i.shape)

    ratio[ind] = orbval[k, d][ind] / pyscfval[k][d][ind]
    v = max(np.amax(np.abs(w[k][d])) for w in [diff])
    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(ratio.T, vmin=-2, vmax=2, cmap="PRGn")
    axs[1].imshow(diff[k, d].T, vmin=-v, vmax=v, cmap="PRGn")
    axs[0].set_title("ratio")
    axs[1].set_title("diff")
    #plt.title(f"ao={labels[ao]}")
    #plt.legend()
    plt.savefig("imshow_pyscf_orbs.pdf", bbox_inches="tight")
    plt.show()