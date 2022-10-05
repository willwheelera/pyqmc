import numpy as np
import kptpyqmc.gpu as gpu
import kptpyqmc.determinant_tools as determinant_tools
import kptpyqmc.orbitals


def sherman_morrison_row(e, inv, vec):
    tmp = np.einsum("ek,ekj->ej", vec, inv)
    ratio = tmp[:, e]
    inv_ratio = inv[:, :, e] / ratio[:, np.newaxis]
    invnew = inv - np.einsum("ki,kj->kij", inv_ratio, tmp)
    invnew[:, :, e] = inv_ratio
    return ratio, invnew


def get_complex_phase(x):
    return x / np.abs(x)


def get_e_ind_for_k_matrix(k_indices, nelec, e, s):
    e_ind = int(np.nonzero(np.nonzero(k_indices == k_indices[e])[0] == e)[0])
    if s == 1:
        e_ind -= nelec[k_indices[e]][0]
    return e_ind


class JoinParameters:
    """
    This class provides a dict-like interface that actually references
    other dictionaries in the background.
    If keys collide, then the first dictionary that matches the key will be returned.
    However, some bad things may happen if you have colliding keys.
    """

    def __init__(self, dicts):
        self.data = {}
        self.data = dicts

    def find_i(self, idx):
        for i, d in enumerate(self.data):
            if idx in d:
                return i

    def __setitem__(self, idx, value):
        i = self.find_i(idx)
        self.data[i][idx] = value

    def __getitem__(self, idx):
        i = self.find_i(idx)
        return self.data[i][idx]

    def __delitem__(self, idx):
        i = self.find_i(idx)
        del self.data[i][idx]

    def __iter__(self):
        for d in self.data:
            yield from d.keys()

    def __len__(self):
        return sum(len(i) for i in self.data)

    def items(self):
        for d in self.data:
            yield from d.items()

    def __repr__(self):
        return self.data.__repr__()

    def keys(self):
        for d in self.data:
            yield from d.keys()

    def values(self):
        for d in self.data:
            yield from d.values()


def sherman_morrison_ms(e, inv, vec):
    tmp = np.einsum("edk,edkj->edj", vec, inv)
    ratio = tmp[:, :, e]
    inv_ratio = inv[:, :, :, e] / ratio[:, :, np.newaxis]
    invnew = inv - np.einsum("kdi,kdj->kdij", inv_ratio, tmp)
    invnew[:, :, :, e] = inv_ratio
    return ratio, invnew


class Slater:
    """
    A multi-determinant wave function object initialized
    via an SCF calculation.

    How to use with hci

    .. code-block:: python

        cisolver = pyscf.hci.SCI(mol)
        cisolver.select_cutoff=0.1
        nmo = mf.mo_coeff.shape[1]
        nelec = mol.nelec
        h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
        h2 = pyscf.ao2mo.full(mol, mf.mo_coeff)
        e, civec = cisolver.kernel(h1, h2, nmo, nelec, verbose=4)
        cisolver.ci = civec[0]
        wf = pyqmc.multislater.MultiSlater(mol, mf, cisolver, tol=0.1)


    """

    def __init__(self, mol, mf, mc=None, tol=None, twist=None, determinants=None):
        """
        determinants should be a list of tuples, for example
        [ (1.0, [0,1],[0,1]),
          (-0.2, [0,2],[0,2]) ]
        would be a two-determinant wave function with a doubles excitation in the second one.

        determinants overrides any information in mc, if passed.
        """
        self._mol = mol
        self._nelec = mol.nelec

        self.myparameters = {}
        (
            self.myparameters["det_coeff"],
            self._det_occup,
            self._det_map,
            self._nelec_k,
            self.orbitals,
        ) = kptpyqmc.orbitals.choose_evaluator_from_pyscf(
            mol, mf, mc, twist=twist, determinants=determinants
        )

        self.parameters = self.orbitals.parameters

        self.iscomplex = self.orbitals.iscomplex or bool(
            sum(map(gpu.cp.iscomplexobj, self.parameters.values()))
        )
        self.dtype = complex if self.iscomplex else float
        self.get_phase = get_complex_phase if self.iscomplex else gpu.cp.sign

    def recompute(self, configs):
        """This computes the value from scratch. Returns the logarithm of the wave function as
        (phase,logdet). If the wf is real, phase will be +/- 1."""

        nconf, nelec, ndim = configs.configs.shape
        self.e_ind_for_k_matrix = np.array(
            [
                get_e_ind_for_k_matrix(
                    configs.k_indices, self._nelec_k, e, int(e >= self._nelec[0])
                )
                for e in range(sum(self._nelec))
            ]
        )
        aos = self.orbitals.aos("GTOval_sph", configs)
        self._aovals = [ao[0] for ao in aos]
        self._dets = []
        self._inverse = []
        for s in [0, 1]:
            spin_slice = lambda n, s: slice(n[0] * s, n[0] + n[1] * s)
            aos_ = [a[0, :, spin_slice(n, s), :] for a, n in zip(aos, self._nelec_k)]
            mo = self.orbitals.mos(aos_, s)  # list for each k
            self._dets.append(
                [gpu.cp.asarray(np.linalg.slogdet(m)) for m in mo]
            )  # spin, kpt, (sign, val), nconf
            self._inverse.append(
                [gpu.cp.linalg.inv(m) for m in mo]
            )  # spin, kpt, nconf, nelec, nelec
        self._dets = gpu.cp.asarray(self._dets)
        return self.value()

    def updateinternals(self, e, epos, mask=None):
        """Update any internals given that electron e moved to epos. mask is a Boolean array
        which allows us to update only certain walkers"""

        s = int(e >= self._nelec[0])
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        eeff = e - s * self._nelec[0]
        ki = epos.k_indices
        # How do we index the electron in the kpt?
        # a[k_indices == i] extracts a list. where does position e map to?
        # np.nonzero(np.nonzero(k_indices == i)[0] == e)[0]
        e_ind = self.e_ind_for_k_matrix[e]
        ao = self.orbitals.aos("GTOval_sph", epos, mask)
        self._aovals[ki][mask, e_ind] = ao[0][0, :, 0]
        mo = self.orbitals.mos_single_elec(ao, s, ki)[0][..., 0, :]

        det_ratio, self._inverse[s][ki][mask] = sherman_morrison_row(
            e_ind, self._inverse[s][ki][mask], mo
        )

        self._updateval(det_ratio, s, ki, mask)

    def value(self):
        """Return logarithm of the wave function as noted in recompute()"""
        phase = np.prod(self._dets[:, :, 0], axis=(0, 1))
        val = np.sum(self._dets[:, :, 1].real, axis=(0, 1))
        return phase, val

    def _updateval(self, ratio, s, ki, mask):
        self._dets[s][ki][0, mask] *= self.get_phase(ratio)
        self._dets[s][ki][1, mask] += gpu.cp.log(gpu.cp.abs(ratio))

    def _testrow(self, e, ki, vec, mask=None, spin=None):
        e_ind = self.e_ind_for_k_matrix[e]
        s = int(e >= self._nelec[0]) if spin is None else spin
        if mask is None:
            mask = [True] * vec.shape[0]

        return gpu.cp.einsum(
            "i...j,ij...->i...",
            vec,
            self._inverse[s][ki][mask][..., e_ind],
        )

    def _testrowderiv(self, e, ki, vec, spin=None):
        """vec is a nconfig,nmo vector which replaces row e"""
        e_ind = self.e_ind_for_k_matrix[e]
        s = int(e >= self._nelec[0]) if spin is None else spin

        return gpu.cp.einsum(
            "ei...j,ij...->ei...",
            vec,
            self._inverse[s][ki][..., e_ind],
        )

    def _testcol(self, i, s, ki, vec):
        """vec is a nconfig,nmo vector which replaces column i
        of spin s in determinant det"""

        return gpu.cp.einsum(
            "ij...,ij->i...", vec, self._inverse[s][ki][:, i, :], optimize="greedy"
        )

    def gradient(self, e, epos):
        """Compute the gradient of the log wave function
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        s = int(e >= self._nelec[0])
        ao = self.orbitals.aos("GTOval_sph_deriv1", epos)
        mo = self.orbitals.mos_single_elec(ao, s, epos.k_indices)[..., 0, :]

        ratios = self._testrowderiv(e, epos.k_indices, mo)
        return gpu.asnumpy(ratios[1:] / ratios[0])

    def gradient_value(self, e, epos):
        """Compute the gradient of the log wave function
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        s = int(e >= self._nelec[0])
        ao = self.orbitals.aos("GTOval_sph_deriv1", epos)
        mo = self.orbitals.mos_single_elec(ao, s, epos.k_indices)[..., 0, :]

        ratios = gpu.asnumpy(self._testrowderiv(e, epos.k_indices, mo))
        return ratios[1:] / ratios[0], ratios[0]

    def laplacian(self, e, epos):
        """ Compute the laplacian Psi/ Psi. """
        s = int(e >= self._nelec[0])
        ao = self.orbitals.aos("GTOval_sph_deriv2", epos)
        ao = gpu.cp.concatenate(
            [ao[0][0:1, ...], ao[0][[4, 7, 9], ...].sum(axis=0, keepdims=True)], axis=0
        )[np.newaxis]
        mo = self.orbitals.mos_single_elec(ao, s, epos.k_indices)[..., 0, :]
        ratios = self._testrowderiv(e, epos.k_indices, mo)
        return gpu.asnumpy(ratios[1] / ratios[0])

    def gradient_laplacian(self, e, epos):
        s = int(e >= self._nelec[0])
        ao = self.orbitals.aos("GTOval_sph_deriv2", epos)
        ao = gpu.cp.concatenate(
            [ao[0][0:4, ...], ao[0][[4, 7, 9], ...].sum(axis=0, keepdims=True)], axis=0
        )[np.newaxis]
        mo = self.orbitals.mos_single_elec(ao, s, epos.k_indices)[..., 0, :]
        ratios = self._testrowderiv(e, epos.k_indices, mo)
        ratios = gpu.asnumpy(ratios / ratios[:1])
        return ratios[1:-1], ratios[-1]

    def testvalue(self, e, epos, mask=None):
        """return the ratio between the current wave function and the wave function if
        electron e's position is replaced by epos"""
        s = int(e >= self._nelec[0])
        ao = self.orbitals.aos("GTOval_sph", epos, mask)
        mo = self.orbitals.mos_single_elec(ao, s, epos.k_indices)[0]
        mo = gpu.cp.asarray(mo)
        if len(epos.configs.shape) == 2:
            mo = mo[..., 0, :]
        return gpu.asnumpy(self._testrow(e, epos.k_indices, mo, mask))

    def testvalue_many(self, e, epos, mask=None):
        """return the ratio between the current wave function and the wave function if
        electron e's position is replaced by epos for each electron"""
        s = (e >= self._nelec[0]).astype(int)
        ao = self.orbitals.aos("GTOval_sph", epos, mask)
        ratios = gpu.cp.zeros((epos.configs.shape[0], e.shape[0]), dtype=self.dtype)
        for spin in [0, 1]:
            ind = s == spin
            mo = self.orbitals.mos(ao, spin)
            mo_vals = mo[..., self._det_occup[spin]]
            ratios[:, ind] = self._testrow(e[ind], mo_vals, mask, spin=spin)

        return gpu.asnumpy(ratios)

    def pgradient(self):
        """Compute the parameter gradient of Psi.
        Returns :math:`\partial_p \Psi/\Psi` as a dictionary of numpy arrays,
        which correspond to the parameter dictionary.

        The wave function is given by ci Di, with an implicit sum

        We have two sets of parameters:

        Determinant coefficients:
        di psi/psi = Dui Ddi/psi

        Orbital coefficients:
        dj psi/psi = ci dj (Dui Ddi)/psi

        Let's suppose that j corresponds to an up orbital coefficient. Then
        dj (Dui Ddi) = (dj Dui)/Dui Dui Ddi/psi = (dj Dui)/Dui di psi/psi
        where di psi/psi is the derivative defined above.
        """
        d = {}

        # Det coeff
        curr_val = self.value()
        d["det_coeff"] = (
            self._dets[0][0, :, self._det_map[0]]
            * self._dets[1][0, :, self._det_map[1]]
            * gpu.cp.exp(
                self._dets[0][1, :, self._det_map[0]]
                + self._dets[1][1, :, self._det_map[1]]
                - gpu.cp.array(curr_val[1])
            )
            / gpu.cp.array(curr_val[0])
        ).T

        for s, parm in zip([0, 1], ["mo_coeff_alpha", "mo_coeff_beta"]):
            ao = self._aovals[
                :, :, s * self._nelec[0] : self._nelec[s] + s * self._nelec[0], :
            ]

            split, aos = self.orbitals.pgradient(ao, s)
            mos = gpu.cp.split(gpu.cp.arange(split[-1]), gpu.asnumpy(split).astype(int))
            # Compute dj Diu/Diu
            nao = aos[0].shape[-1]
            nconf = aos[0].shape[0]
            nmo = int(split[-1])
            deriv = gpu.cp.zeros(
                (len(self._det_occup[s]), nconf, nao, nmo), dtype=curr_val[0].dtype
            )
            for det, occ in enumerate(self._det_occup[s]):
                for ao, mo in zip(aos, mos):
                    for i in mo:
                        if i in occ:
                            col = occ.index(i)
                            deriv[det, :, :, i] = self._testcol(det, col, s, ao)

            # now we reduce over determinants
            d[parm] = gpu.cp.zeros(deriv.shape[1:], dtype=curr_val[0].dtype)
            for di, coeff in enumerate(self.parameters["det_coeff"]):
                whichdet = self._det_map[s][di]
                d[parm] += (
                    deriv[whichdet]
                    * coeff
                    * d["det_coeff"][:, di, np.newaxis, np.newaxis]
                )
        for k, v in d.items():
            d[k] = gpu.asnumpy(v)
        return d
