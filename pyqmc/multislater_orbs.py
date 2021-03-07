import numpy as np
from pyqmc.slater import sherman_morrison_row, get_complex_phase
import pyqmc.determinant_tools as determinant_tools
import pyqmc.orbitals

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
        for d1 in self.data:
            for d2 in self.data:
                for k in d1.keys():
                    if k in d2:
                        raise ValueError("JoinParameters initialized with conflicting keys")


    def find_i(self,idx):
        for i,d in enumerate(self.data):
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
    ratio = np.einsum("idj,idj->id", vec, inv[:, :, :, e])
    tmp = np.einsum("edk,edkj->edj", vec, inv)
    invnew = (
        inv
        - np.einsum("kdi,kdj->kdij", inv[:, :, :, e], tmp)
        / ratio[:, :, np.newaxis, np.newaxis]
    )
    invnew[:, :, :, e] = inv[:, :, :, e] / ratio[:, :, np.newaxis]
    return ratio, invnew




class MultiSlater:
    """
    A multi-determinant wave function object initialized
    via an SCF calculation. Methods and structure are very similar
    to the PySCFSlaterUHF class.

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

    def __init__(self, mol, mf, mc=None, tol=None, twist=None):
        self.tol = -1 if tol is None else tol
        self._mol = mol
        if hasattr(mc, "nelecas"):
            # In case nelecas overrode the information from the molecule object.
            self._nelec = (mc.nelecas[0] + mc.ncore, mc.nelecas[1] + mc.ncore)
        else:
            self._nelec = mol.nelec

        self.myparameters={}
        if mc is None:
            self.myparameters["det_coeff"], self._det_occup, self._det_map = determinant_tools.determinants_from_mean_field(mf)
        else:
            self.myparameters["det_coeff"], self._det_occup, self._det_map = determinant_tools.interpret_ci(mc, self.tol)

        self.orbitals = pyqmc.orbitals.choose_evaluator_from_pyscf(mol, mf, mc, self._det_occup)
        self.parameters=JoinParameters([self.myparameters,self.orbitals.parameters])

        self.iscomplex = bool(sum(map(np.iscomplexobj, self.parameters.values())))
        self.get_phase = get_complex_phase if self.iscomplex else np.sign


    def recompute(self, configs):
        """This computes the value from scratch. Returns the logarithm of the wave function as
        (phase,logdet). If the wf is real, phase will be +/- 1."""

        nconf, nelec, ndim = configs.configs.shape

        aos = self.orbitals.aos('GTOval_sph', configs).reshape(nconf, nelec, -1)
        self._aovals = aos.reshape(-1,nconf,nelec, aos.shape[-1])

        self._dets = []
        self._inverse = []
        for s in [0, 1]:
            begin = self._nelec[0]*s
            end = self._nelec[0] + self._nelec[1] * s
            mo = self.orbitals.mos(self._aovals[:,:,begin:end, :],s)
            mo_vals = np.swapaxes(mo[:, :, self._det_occup[s]], 1, 2)
            self._dets.append(
                np.array(np.linalg.slogdet(mo_vals))
            )  # Spin, (sign, val), nconf, [ndet_up, ndet_dn]
            self._inverse.append(
                np.linalg.inv(mo_vals)
            )  # spin, Nconf, [ndet_up, ndet_dn], nelec, nelec
        return self.value()

    def updateinternals(self, e, epos, mask=None):
        """Update any internals given that electron e moved to epos. mask is a Boolean array 
        which allows us to update only certain walkers"""

        s = int(e >= self._nelec[0])
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        eeff = e - s * self._nelec[0]
        ao = self.orbitals.aos("GTOval_sph",epos,mask)
        self._aovals[:,mask, e, :] = ao
        mo = self.orbitals.mos(ao,s)

        mo_vals = mo[:, self._det_occup[s]]
        det_ratio, self._inverse[s][mask, :, :, :] = sherman_morrison_ms(
            eeff, self._inverse[s][mask, :, :, :], mo_vals
        )

        self._updateval(det_ratio, s, mask)

    def value(self):
        """Return logarithm of the wave function as noted in recompute()"""
        updets = self._dets[0][:, :, self._det_map[0]]
        dndets = self._dets[1][:, :, self._det_map[1]]
        return determinant_tools.compute_value(updets,dndets, self.parameters["det_coeff"])

    def _updateval(self, ratio, s, mask):
        self._dets[s][0, mask, :] *= self.get_phase(ratio)
        self._dets[s][1, mask, :] += np.log(np.abs(ratio))

    def _testrow(self, e, vec, mask=None, spin=None):
        """vec is a nconfig,nmo vector which replaces row e"""
        s = int(e >= self._nelec[0]) if spin is None else spin
        if mask is None:
            mask = [True] * vec.shape[0]

        ratios = np.einsum(
            "i...dj,idj...->i...d",
            vec,
            self._inverse[s][mask][..., e - s * self._nelec[0]],
        )

        upref = np.amax(self._dets[0][1])
        dnref = np.amax(self._dets[1][1])
        
        det_array = (
            self._dets[0][0, :, self._det_map[0]][:, mask]
            * self._dets[1][0, :, self._det_map[1]][:, mask]
            * np.exp(
                self._dets[0][1, :, self._det_map[0]][:, mask]
                + self._dets[1][1, :, self._det_map[1]][:, mask]-upref-dnref
            )
        )
        numer = np.einsum(
            "i...d,d,di->i...",
            ratios[..., self._det_map[s]],
            self.parameters["det_coeff"],
            det_array,
        )
        denom = np.einsum(
            "d,di->i...",
            self.parameters["det_coeff"],
            det_array,
        )
        #curr_val = self.value()
        
        if len(numer.shape) == 2:
            denom = denom[:, np.newaxis]
        return numer / denom

    def _testcol(self, det, i, s, vec):
        """vec is a nconfig,nmo vector which replaces column i 
        of spin s in determinant det"""

        return np.einsum(
            "ij...,ij->i...", vec, self._inverse[s][:, det, i, :], optimize="greedy"
        )

    def gradient(self, e, epos):
        """ Compute the gradient of the log wave function 
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        s = int(e >= self._nelec[0])
        aograd = self.orbitals.aos('GTOval_sph_deriv1', epos)
        mograd = self.orbitals.mos(aograd, s)

        mograd_vals = mograd[:, :, self._det_occup[s]]

        ratios = np.asarray([self._testrow(e, x) for x in mograd_vals])
        return ratios[1:] / ratios[0]

    def laplacian(self, e, epos):
        """ Compute the laplacian Psi/ Psi. """
        s = int(e >= self._nelec[0])
        ao = self.orbitals.aos("GTOval_sph_deriv2", epos)
        ao_val = ao[:,0,:,:]
        ao_lap = np.sum(ao[:,[4,7,9],:,:],axis=1)
        mos = [self.orbitals.mos(x,s)[...,self._det_occup[s]] for x in [ao_val, ao_lap]]
        ratios = [self._testrow(e,mo) for mo in mos]
        return ratios[1]/ratios[0]

    def gradient_laplacian(self, e, epos):
        s = int(e >= self._nelec[0])
        ao = self.orbitals.aos("GTOval_sph_deriv2", epos)
        ao = np.concatenate([ao[:,0:4,...], ao[:,[4,7,9],...].sum(axis=1,keepdims=True)],axis=1)
        mo = self.orbitals.mos(ao,s)
        mo_vals = mo[:, :, self._det_occup[s]]
        ratios = np.asarray([self._testrow(e, x) for x in mo_vals])
        return ratios[1:-1] / ratios[:1], ratios[-1] / ratios[0]

    def testvalue(self, e, epos, mask=None):
        """ return the ratio between the current wave function and the wave function if 
        electron e's position is replaced by epos"""
        s = int(e >= self._nelec[0])
        ao = self.orbitals.aos('GTOval_sph',epos, mask)
        mo = self.orbitals.mos(ao, s)
        mo_vals = mo[..., self._det_occup[s]]
        return self._testrow(e, mo_vals, mask)

    def testvalue_many(self, e, epos, mask=None):
        """ return the ratio between the current wave function and the wave function if 
        electron e's position is replaced by epos for each electron"""
        s = (e >= self._nelec[0]).astype(int)
        ao = self.orbitals.aos('GTOval_sph', epos, mask)

        ratios = np.zeros((epos.configs.shape[0], e.shape[0]))
        for spin in [0, 1]:
            ind = s == spin
            #mo = ao.dot(self.parameters[self._coefflookup[spin]])
            mo = self.orbitals.mos(ao, spin)
            mo_vals = mo[..., self._det_occup[spin]]
            ratios[:, ind] = self._testrow(e[ind], mo_vals, mask, spin=spin)

        return ratios

    def pgradient(self):
        r"""Compute the parameter gradient of Psi. 
        Returns $$d_p \Psi/\Psi$$ as a dictionary of numpy arrays,
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
            * np.exp(
                self._dets[0][1, :, self._det_map[0]]
                + self._dets[1][1, :, self._det_map[1]]
                - curr_val[1]
            )/curr_val[0]
        ).T

        for s,parm in zip([0,1],["mo_coeff_alpha", "mo_coeff_beta"]):
            ao = self._aovals[
                :, :, s * self._nelec[0] : self._nelec[s] + s * self._nelec[0], :
            ]

            split, aos = self.orbitals.pgradient(ao, s)
            mos = np.split(range(split[-1]), split)
            # Compute dj Diu/Diu
            nao = aos[0].shape[-1]
            nconf = aos[0].shape[0]
            nmo = split[-1]
            deriv = np.zeros((len(self._det_occup[s]), nconf, nao, nmo))
            for det, occ in enumerate(self._det_occup[s]):
                for ao, mo in zip(aos, mos):
                    for i in mo:
                        if i in occ:
                            col = occ.index(i)
                            deriv[det, :, :, i]= self._testcol(det, col, s, ao) 

            # now we reduce over determinants
            d[parm] = np.zeros(deriv.shape[1:])
            for di,coeff in enumerate(self.parameters['det_coeff']):
                whichdet = self._det_map[s][di]
                d[parm] += deriv[whichdet]*coeff*d["det_coeff"][:,di, np.newaxis, np.newaxis]

        return d
