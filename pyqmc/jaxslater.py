import numpy as np
from pyqmc import pbc
from jax.ops import index, index_add, index_update
import jax.numpy as jnp
from jax import jit, device_put


### ON CPU ###
# configs
# _aovals


### ON GPU ###
# _inverse
# parameters mo_coeff


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


def _testrow_kernel(eeff, vec, inverse):
    return jnp.einsum("i...j,ij...->i...", vec, inverse[:, :, eeff])


def _testrow_gradient_kernel(eeff, mograd, inverse):
    return jnp.einsum("dij,ij->di", mograd, inverse[:, :, eeff])


def evaluate_mos_mol_kernel(ao, mo_coeff):
    return jnp.dot(ao, mo_coeff)


def evaluate_mos_pbc_kernel(aos, mo_coeff):
    mo = [jnp.dot(ao, param) for ao, param in zip(aos, mo_coeff)]
    return jnp.concatenate(mo, axis=-1)


def updateinternals_mol_kernel(e, aos, mo_coeff, inverse):
    mo = evaluate_mos_mol_kernel(jnp.asarray(aos), jnp.asarray(mo_coeff))
    ratio, newinverse = sherman_morrison_row(e, jnp.asarray(inverse), mo)
    return ratio, newinverse


def updateinternals_pbc_kernel(e, aos, mo_coeff, inverse):
    mo = evaluate_mos_pbc_kernel(jnp.asarray(aos), jnp.asarray(mo_coeff))
    ratio, newinverse = sherman_morrison_row(e, jnp.asarray(inverse), mo)
    return ratio, newinverse


def testvalue_mol_kernel(aos, mo_coeff, eeff, inverse):
    mo = evaluate_mos_mol_kernel(aos, mo_coeff)
    mo = mo.reshape(inverse.shape[0], -1, inverse.shape[1])
    ratios = _testrow_kernel(eeff, mo, inverse)
    return jnp.squeeze(ratios)


def testvalue_pbc_kernel(aos, mo_coeff, eeff, inverse):
    mo = evaluate_mos_pbc_kernel(ao, mo_coeff)
    mo = mo.reshape(inverse.shape[0], -1, inverse.shape[1])
    ratios = _testrow_kernel(eeff, mo, inverse)
    return jnp.squeeze(ratios)


def evaluate_orbitals_pbc(
    configs, cell, latvecs, S, kpts, get_wrapphase, mask=None, eval_str="GTOval_sph"
):
    mycoords = configs.configs
    configswrap = configs.wrap
    if mask is not None:
        mycoords = mycoords[mask]
        configswrap = configswrap[mask]
    mycoords = mycoords.reshape((-1, mycoords.shape[-1]))
    configswrap = configswrap.reshape((-1, configswrap.shape[-1]))

    # wrap supercell positions into primitive cell
    prim_coords, prim_wrap = pbc.enforce_pbc(latvecs, mycoords)

    # evaluate AOs for all electron positions
    ao = cell.eval_gto("PBC" + eval_str, prim_coords, kpts=kpts)
    return evaluate_orbitals_pbc_kernel(
        ao, prim_wrap, configswrap, latvecs, S, kpts, get_wrapphase
    )


def evaluate_orbitals_pbc_kernel(
    ao, prim_wrap, origwrap, lattice_vectors, S, kpts, phasefunc
):
    wrap = prim_wrap + jnp.dot(origwrap, S)
    kdotR = jnp.einsum("ij,kj,lk->il", kpts, lattice_vectors, wrap)
    wrap_phase = phasefunc(kdotR)
    # return [ao[k] * wrap_phase[k][:, np.newaxis] for k in range(len(kpts))]
    return [ao * wp[:, jnp.newaxis] for ao, wp in zip(aos, wrap_phase)]


# _gldict = {"laplacian": np.s_[:1], "gradient_laplacian": np.s_[0:4]}
_gldict = {"laplacian": slice(1), "gradient_laplacian": slice(4)}


def _aostack_mol(ao, gl):
    return jnp.concatenate(
        [ao[_gldict[gl]], jnp.sum(ao[[4, 7, 9]], axis=0, keepdims=True)], axis=0
    )


def _aostack_pbc(ao, gl):
    return [_aostack_mol(ak, gl) for ak in ao]


def get_wrapphase_real(x):
    return (-1) ** jnp.round(x / jnp.pi)


def get_wrapphase_complex(x):
    return jnp.exp(1j * x)


def get_real_phase(x):
    return jnp.sign(x)


def get_complex_phase(x):
    return x / jnp.abs(x)


def get_kinds(cell, mf, kpts, tol=1e-6):
    """Given a list of kpts, return inds such that mf.kpts[inds] is a list of kpts equivalent to the input list"""
    kdiffs = mf.kpts[np.newaxis] - kpts[:, np.newaxis]
    frac_kdiffs = np.dot(kdiffs, cell.lattice_vectors().T) / (2 * np.pi)
    kdiffs = np.mod(frac_kdiffs + 0.5, 1) - 0.5
    return np.nonzero(np.linalg.norm(kdiffs, axis=-1) < tol)[1]


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
            self._init_pbc(mol, mf, twist)
        else:
            self._init_mol(mol, mf)
        self.pbc_str = "PBC" if hasattr(mol, "a") else ""
        self._aostack = _aostack_pbc if hasattr(mol, "a") else _aostack_mol

        self.dtype = complex if self.iscomplex else float
        if self.iscomplex:
            self.get_phase = get_complex_phase
            self.get_wrapphase = get_wrapphase_complex
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

    def _init_pbc(self, cell, mf, twist):
        from pyscf.pbc import scf
        from pyqmc.supercell import get_supercell_kpts

        # Make sure supercell has attributes S and original_cell
        for attribute in ["original_cell", "S", "scale"]:
            if not hasattr(cell, attribute):
                print('Warning: supercell is missing attribute "%s"' % attribute)
                print("setting original_cell=supercell and S=np.eye(3)")
                cell.original_cell = cell
                cell.S = np.eye(3)
                cell.scale = 1
        self.supercell = cell
        self._cell = cell.original_cell
        self.primitive_lattice_vectors = self._cell.lattice_vectors()

        # Define kpts
        if twist is None:
            twist = np.zeros(3)
        else:
            twist = np.dot(np.linalg.inv(cell.a), np.mod(twist, 1.0)) * 2 * np.pi
        self.kinds = get_kinds(self._cell, mf, get_supercell_kpts(cell) + twist)
        self._kpts = mf.kpts[self.kinds]
        assert len(self.kinds) == len(self._kpts), (self._kpts, mf.kpts)
        self.nk = len(self._kpts)

        # Define parameters
        self.param_split = {}
        for s, lookup in enumerate(self._coefflookup):
            mclist = []
            for kind in self.kinds:
                if len(mf.mo_coeff[0][0].shape) == 2:
                    mca = mf.mo_coeff[s][kind][:, np.asarray(mf.mo_occ[s][kind] > 0.9)]
                else:
                    minocc = (0.9, 1.1)[s]
                    mca = mf.mo_coeff[kind][:, np.asarray(mf.mo_occ[kind] > minocc)]
                if np.all(mca.imag < self.real_tol * np.finfo(float).eps):
                    mca = np.real(mca)
                mclist.append(mca / np.sqrt(self.nk))
            self.param_split[lookup] = np.cumsum([m.shape[1] for m in mclist])
            self.parameters[lookup] = np.concatenate(mclist, axis=-1)

        self.iscomplex = bool(sum(map(np.iscomplexobj, self.parameters.values())))
        self.iscomplex = self.iscomplex or np.linalg.norm(self._kpts) > 1e-12

        # Define nelec
        if len(mf.mo_coeff[0][0].shape) == 2:
            # Then indices are (spin, kpt, basis, mo)
            self._nelec = [int(np.sum([o[k] for k in self.kinds])) for o in mf.mo_occ]
        elif len(mf.mo_coeff[0][0].shape) == 1:
            # Then indices are (kpt, basis, mo)
            self._nelec = [
                int(np.sum([mf.mo_occ[k] > t for k in self.kinds])) for t in (0.9, 1.1)
            ]
        else:
            print("Warning: PySCFSlater not expecting scf object of type", type(mf))
            scale = self.supercell.scale
            self._nelec = [int(np.round(n * scale)) for n in self._cell.nelec]
        self._nelec = tuple(self._nelec)

        self.evaluate_orbitals = self._evaluate_orbitals_pbc
        self.evaluate_mos = self._evaluate_mos_pbc

    def _evaluate_orbitals_mol(self, configs, mask=None, eval_str="GTOval_sph"):
        mycoords = configs.configs if mask is None else configs.configs[mask]
        mycoords = mycoords.reshape((-1, mycoords.shape[-1]))
        return self._mol.eval_gto(eval_str, mycoords)

    def _evaluate_mos_mol(self, ao, s):
        return evaluate_mos_mol_kernel(ao, self.parameters[self._coefflookup[s]])

    def _evaluate_orbitals_pbc(self, configs, mask=None, eval_str="GTOval_sph"):
        return evaluate_orbitals_pbc(
            configs,
            self._cell,
            self.primitive_lattice_vectors,
            self.supercell.S,
            self._kpts,
            self.get_wrapphase,
            mask=mask,
            eval_str=eval_str,
        )

    def _evaluate_mos_pbc(self, aos, s):
        """
        Evaluate MOs for spin s given aos
        """
        return evaluate_mos_pbc_kernel(aos, self.params_split_by_k[s])

    def recompute(self, configs):
        """This computes the value from scratch. Returns the logarithm of the wave function as
        (phase,logdet). If the wf is real, phase will be +/- 1."""
        nconf, nelec, ndim = configs.configs.shape
        aos = self.evaluate_orbitals(configs)
        if hasattr(self, "nk"):
            aos_shape = (self.nk, nconf, nelec, -1)
            self.params_split_by_k = [
                jnp.split(self.parameters[c], self.param_split[c], axis=-1)
                for c in self._coefflookup
            ]
        else:
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
        ###
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

    def _testcol(self, i, s, vec):
        """vec is a nconfig,nmo vector which replaces column i"""
        return jnp.einsum("ij...,ij->i...", vec, self._inverse[s][:, i, :])

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

    def testvalue_many(self, e, epos, mask=None):
        """ return the ratio between the current wave function and the wave function if 
        an electron's position is replaced by epos for each electron"""
        s = (e >= self._nelec[0]).astype(int)
        nmask = epos.configs.shape[0] if mask is None else np.sum(mask)
        if nmask == 0:
            return np.zeros((0, epos.configs.shape[1]))

        aos = self.evaluate_orbitals(epos, mask)
        ratios = np.zeros((epos.configs.shape[0], e.shape[0]), dtype=self.dtype)
        for spin in [0, 1]:
            ind = s == spin
            mo = self.evaluate_mos(aos, spin)
            mo = mo.reshape(nmask, *epos.configs.shape[1:-1], self._nelec[spin])
            ratios[:, ind] = self._testrow(e[ind], mo, mask=mask, spin=spin)
        return ratios

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

    def pgradient(self):
        d = {"det_coeff": np.zeros(self._aovals.shape[-3])}
        for parm in ["mo_coeff_alpha", "mo_coeff_beta"]:
            s = int("beta" in parm)
            # Get AOs for our spin channel only
            i0, i1 = s * self._nelec[0], self._nelec[0] + s * self._nelec[1]
            ao = self._aovals[:, :, i0:i1, :]  # (kpt, config, electron, ao)
            pgrad_shape = (ao.shape[-3],) + self.parameters[parm].shape
            pgrad = np.zeros(pgrad_shape, dtype=self.dtype)  # (nconf, coeff)
            # Compute derivatives w.r.t. MO coefficients
            if ao.shape[0] > 1:  # multiple kpts
                split_sizes = np.diff([0] + list(self.param_split[parm]))
                k = np.repeat(np.arange(self.nk), split_sizes)
                for i in range(self._nelec[s]):  # MO loop
                    pgrad[:, :, i] = self._testcol(i, s, ao[k[i]])
            else:
                ao = ao[0]
                for i in range(self._nelec[s]):  # MO loop
                    pgrad[:, :, i] = self._testcol(i, s, ao)
            d[parm] = np.asarray(pgrad)
        return d
