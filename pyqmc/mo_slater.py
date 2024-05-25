# MIT License
# 
# Copyright (c) 2019-2024 The PyQMC Developers
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import numpy as np
import pyqmc.gpu as gpu
import pyqmc.determinant_tools as determinant_tools
import pyqmc.orbitals
import pyqmc.pyscftools
import pyqmc.slater as slater
import pyqmc.symmetry_basis as symmetry_basis
import warnings


def get_full_mo_coeff(mf, mc=None):
    try:
        mf = mf.to_uhf()
    except TypeError:
        mf = mf.to_uhf(mf)
    
    if hasattr(mc, "mo_coeff"):
            # assume no kpts for mc calculation
        _mo_coeff = mc.mo_coeff
        if len(_mo_coeff.shape) == 2: # restricted spin: create up and down copies
            _mo_coeff = [_mo_coeff, _mo_coeff]
        if periodic:
            _mo_coeff = [m[np.newaxis] for m in _mo_coeff] # add kpt dimension
    else:
        _mo_coeff = mf.mo_coeff
    return _mo_coeff
        

class MOSlater(slater.Slater):
    r"""
    A multi-determinant wave function object initialized
    via an SCF calculation.
    This stores orbital parameters by MO, 
   

    """

    def __init__(self, mol, mf, mc=None, tol=None, twist=0, determinants=None, eval_gto_precision=None):
        """
        """
        if hasattr(mf, "kpts") and np.linalg.norm(mf.kpts) > 0:
            raise NotImplementedError("MO slater is not yet implemented for k points")

        self.tol = -1 if tol is None else tol
        self._mol = mol
        if hasattr(mc, "nelecas"):
            # In case nelecas overrode the information from the molecule object.
            ncore = mc.ncore
            if not hasattr(ncore, "__len__"):
                ncore = [ncore, ncore]
            self._nelec = (mc.nelecas[0] + ncore[0], mc.nelecas[1] + ncore[1])
        else:
            self._nelec = mol.nelec

        self.eval_gto_precision = eval_gto_precision
        self.parameters = {}
        (
            self.parameters["det_coeff"],
            self._det_occup,
            self._det_map,
            self.orbitals,
        ) = pyqmc.pyscftools.orbital_evaluator_from_pyscf(
            mol, mf, mc, twist=twist, determinants=determinants, tol=self.tol, eval_gto_precision=self.eval_gto_precision
        )

        ovlp = mf.get_ovlp()
        self.mo0 = get_full_mo_coeff(mf, mc)
        for s, alpha in enumerate(["mo_coeff_alpha", "mo_coeff_beta"]):
            nmo = self.orbitals.parameters[alpha].shape[1] # how many mos we store
            self.parameters[alpha] = np.eye(len(ovlp), nmo)
                
        iscomplex = self.orbitals.mo_dtype == complex or bool(
            sum(map(gpu.cp.iscomplexobj, self.parameters.values()))
        )
        self.dtype = complex if iscomplex else float
        self.get_phase = slater.get_complex_phase if iscomplex else gpu.cp.sign

    def set_orbital_mos(self):
        for s, alpha in enumerate(["mo_coeff_alpha", "mo_coeff_beta"]):
            self.orbitals.parameters[alpha] = self.mo0[s] @ self.parameters[alpha]

    def recompute(self, configs):
        """This computes the value from scratch. Returns the logarithm of the wave function as
        (phase,logdet). If the wf is real, phase will be +/- 1."""
        self.set_orbital_mos()
        return super().recompute(configs)

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

        M_{ac} = sum_b B_{ab} m_{bc}
        dM_{ac} / dm_{bc} = B_{ab}
        dPsi/dm_{bc} = sum_a dPsi/dM_{ac} dM_{ac}/dm{bc}
                     = sum_a dPsi/dM_{ac} B_{ab}
        """
        d = super().pgradient()
        for s, alpha in enumerate(["mo_coeff_alpha", "mo_coeff_beta"]):
            d[alpha] = np.einsum("ab,nac->nbc", self.mo0[s], d[alpha])
        return d

