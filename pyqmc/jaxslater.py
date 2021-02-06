import numpy as np
from pyqmc import pbc
from jax.ops import index, index_add, index_update
import jax.numpy as jnp
from jax import jit, device_put


@jit
def sherman_morrison_row(e, inv, vec):
    """
    :param e: electron index
    :type e: int
    :param inv: inverse matrix for the spin associated with e and masked like vec (i.e. inverse[s][mask])
    :type inv: array (nmask, ns, ns), where `ns` is the number of electrons in this spin channel
    :param vec: new row for the Slater matrix
    :type vec: (nmask, ns) array
    """
    tmp = jnp.einsum("ek,ekj->ej", vec, inv)
    ratio = tmp[:, e]
    inv_ratio = inv[:, :, e] / ratio[:, np.newaxis]
    invnew = inv - jnp.einsum("ki,kj->kij", inv_ratio, tmp)
    invnew = index_update(invnew, index[:, :, e], inv_ratio)
    return ratio, invnew


def _testrow_gradient_kernel(eeff, mograd, inverse):
    return jnp.einsum("dij,ij->di", mograd, inverse[:, :, eeff])


_gldict = {"laplacian": slice(1), "gradient_laplacian": slice(4)}


def _aostack_mol(ao, gl):
    return jnp.concatenate(
        [ao[_gldict[gl]], jnp.sum(ao[[4, 7, 9]], axis=0, keepdims=True)], axis=0
    )


def get_wrapphase_real(x):
    return (-1) ** jnp.round(x / jnp.pi)


def get_real_phase(x):
    return jnp.sign(x)


class JaxPySCFSlater:
    """A wave function object has a state defined by a reference configuration of electrons.
    The functions recompute() and updateinternals() change the state of the object, and 
    the rest compute and return values from that state. """

    def __init__(self, mol, mf, twist=None):
        """
        Inputs:
          supercell: object returned by get_supercell(cell, S)
          mf: scf object of primitive cell calculation. scf calculation must include k points that fold onto the gamma point of the supercell
          twist: (3,) array, twisted boundary condition in fractional coordinates, i.e. as coefficients of the reciprocal lattice vectors of the supercell. Integer values are equivalent to zero.
        """
        self.parameters = {"det_coeff": np.array([1.0])}
        self.real_tol = 1e4
        self._coefflookup = ("mo_coeff_alpha", "mo_coeff_beta")

        if hasattr(mol, "a"):
            raise NotImplementedError
        self._init_mol(mol, mf)
        self.pbc_str = "PBC" if hasattr(mol, "a") else ""
        self._aostack = _aostack_pbc if hasattr(mol, "a") else _aostack_mol

        self.dtype = complex if self.iscomplex else float
        if self.iscomplex:
            raise NotImplementedError
        else:
            self.get_phase = get_real_phase
            self.get_wrapphase = get_wrapphase_real

    def _init_mol(self, mol, mf):
        from pyscf import scf

        for s, lookup in enumerate(self._coefflookup):
            if len(mf.mo_occ.shape) == 2:
                self.parameters[lookup] = device_put(
                    mf.mo_coeff[s][:, np.asarray(mf.mo_occ[s] > 0.9)]
                )
            else:
                minocc = (0.9, 1.1)[s]
                self.parameters[lookup] = device_put(
                    mf.mo_coeff[:, np.asarray(mf.mo_occ > minocc)]
                )
        self._nelec = tuple(mol.nelec)
        self._mol = mol
        self.iscomplex = bool(sum(map(np.iscomplexobj, self.parameters.values())))
        self.evaluate_orbitals = self._evaluate_orbitals_mol
        self.evaluate_mos = self._evaluate_mos_mol

    def _evaluate_orbitals_mol(self, configs, mask=None, eval_str="GTOval_sph"):
        mycoords = configs.configs if mask is None else configs.configs[mask]
        mycoords = mycoords.reshape((-1, mycoords.shape[-1]))
        return self._mol.eval_gto(eval_str, mycoords)

    def _evaluate_mos_mol(self, ao, s):
        return jnp.dot(ao, self.parameters[self._coefflookup[s]])

    def recompute(self, configs):
        """This computes the value from scratch. Returns the logarithm of the wave function as
        (phase,logdet). If the wf is real, phase will be +/- 1."""
        nconf, nelec, ndim = configs.configs.shape
        aos = self.evaluate_orbitals(configs)
        aos_shape = (1, nconf, nelec, -1)
        aos = np.reshape(aos, aos_shape)
        self._aovals = aos
        self._dets = []
        self._inverse = []
        for s in [0, 1]:
            i0, i1 = s * self._nelec[0], self._nelec[0] + s * self._nelec[1]
            ne = self._nelec[s]
            mo = self.evaluate_mos(aos[:, :, i0:i1], s).reshape(nconf, ne, ne)
            phase, mag = np.linalg.slogdet(mo)
            self._dets.append((phase, mag))
            self._inverse.append(jnp.linalg.inv(mo))  # inverses are jax arrays

        return self.value()

    def updateinternals(self, e, epos, mask=None):
        s = int(e >= self._nelec[0])
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        eeff = e - s * self._nelec[0]
        aos = self.evaluate_orbitals(epos, mask=mask)
        self._aovals[:, mask, e, :] = np.asarray(aos)  # (kpt, config, ao)
        ### gpu kernel
        mo = self.evaluate_mos(aos, s)
        ratio, newinv = sherman_morrison_row(eeff, self._inverse[s][mask, :, :], mo)
        #####
        self._inverse[s] = index_update(self._inverse[s], index[mask, :, :], newinv)
        self._updateval(ratio, s, mask)

    def _updateval(self, ratio, s, mask):
        self._dets[s][0][mask] *= self.get_phase(ratio)
        self._dets[s][1][mask] += np.log(np.abs(ratio))

    ### not state-changing functions

    def value(self):
        """Return logarithm of the wave function as noted in recompute()"""
        return (
            self._dets[0][0] * self._dets[1][0],
            self._dets[0][1]
            + self._dets[1][1]
            + np.log(np.abs(self.parameters["det_coeff"][0])),
        )

    def _testrow(self, e, vec, mask=None, spin=None):
        """vec is a nconfig,nmo vector which replaces row e"""
        s = int(e >= self._nelec[0]) if spin is None else spin
        elec = e - s * self._nelec[0]
        if mask is None:
            return jnp.einsum("i...j,ij...->i...", vec, self._inverse[s][:, :, elec])

        return jnp.einsum("i...j,ij...->i...", vec, self._inverse[s][mask][:, :, elec])

    def testvalue(self, e, epos, mask=None):
        """ return the ratio between the current wave function and the wave function if 
        electron e's position is replaced by epos"""
        s = int(e >= self._nelec[0])
        nmask = epos.configs.shape[0] if mask is None else np.sum(mask)
        if nmask == 0:
            return np.zeros((0, epos.configs.shape[1]))
        aos = self.evaluate_orbitals(epos, mask)
        mo = self.evaluate_mos(aos, s)
        mo = mo.reshape(nmask, *epos.configs.shape[1:-1], self._nelec[s])
        return self._testrow(e, mo, mask)

    def gradient(self, e, epos):
        """ Compute the gradient of the log wave function 
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        s = int(e >= self._nelec[0])
        aograd = self.evaluate_orbitals(epos, eval_str="GTOval_sph_deriv1")
        mograd = self.evaluate_mos(aograd, s)
        ratios = np.asarray([self._testrow(e, x) for x in mograd])
        return ratios[1:] / ratios[:1]

    def laplacian(self, e, epos):
        s = int(e >= self._nelec[0])
        ao = self.evaluate_orbitals(epos, eval_str="GTOval_sph_deriv2")
        mo = self.evaluate_mos(self._aostack(ao, "laplacian"), s)
        ratios = np.asarray([self._testrow(e, x) for x in mo])
        return ratios[1] / ratios[0]

    def gradient_laplacian(self, e, epos):
        s = int(e >= self._nelec[0])
        ao = self.evaluate_orbitals(epos, eval_str="GTOval_sph_deriv2")
        mo = self.evaluate_mos(self._aostack(ao, "gradient_laplacian"), s)
        ratios = np.asarray([self._testrow(e, x) for x in mo])
        return ratios[1:-1] / ratios[:1], ratios[-1] / ratios[0]
