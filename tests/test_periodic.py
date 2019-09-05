import numpy as np
from pyqmc.distance import MinimalImageDistance, RawDistance
from pyqmc.pbc import enforce_pbc
import pyqmc
import pandas as pd
from pyscf.pbc import gto, scf
from pyqmc.reblock import optimally_reblocked
import time

def test_cubic(kind=0, nk=(1,1,1)): 
    L = 2 
    mol = gto.M(
        atom = '''H     {0}      {0}      {0}                
                  H     {1}      {1}      {1}'''.format(0.0, L/2),
        basis='sto-3g',
        a = np.eye(3)*L,
        spin=0,
        unit='bohr',
    )
    kpts = mol.make_kpts(nk)
    mf = scf.KUKS(mol, kpts)
    mf.xc = "pbe"
    #mf = mf.density_fit()
    mf = mf.run()
    runtest(mol, mf, kind)

def test_RKS(kind=0, nk=(1,1,1)): 
    L = 2 
    mol = gto.M(
        atom = '''He     {0}      {0}      {0}'''.format(0.0),
        basis='sto-3g',
        a = np.eye(3)*L,
        unit='bohr',
    )
    kpts = mol.make_kpts(nk)
    mf = scf.KRKS(mol, kpts)
    mf.xc = "pbe"
    #mf = mf.density_fit()
    mf = mf.run()
    runtest(mol, mf, kind)

def test_noncubic(kind=0, nk=(1,1,1)): 
    L = 3 
    mol = gto.M(
        atom = '''H     {0}      {0}      {0}                
                  H     {1}      {1}      {1}'''.format(0.0, L/4),
        basis='sto-3g',
        a = (np.ones((3,3))-np.eye(3))*L/2,
        spin=0,
        unit='bohr',
    )
    kpts = mol.make_kpts(nk)
    mf = scf.KUKS(mol, kpts)
    mf.xc = "pbe"
    #mf = mf.density_fit()
    mf = mf.run()
    runtest(mol, mf, kind)

def runtest(mol, mf, kind):
    kpt = mf.kpts[kind]
    wf = pyqmc.PySCFSlaterUHF(mol, mf, twist=np.dot(kpt,mol.a.T/np.pi)) 

    #####################################
    ## evaluate KE in PySCF
    #####################################
    ke_mat = mol.pbc_intor('int1e_kin', hermi=1, kpts=np.array(kpt))
    dm = mf.make_rdm1() 
    if len(dm.shape) == 4:
        dm = np.sum(dm, axis=0)
    pyscfke = np.real(np.einsum('ij,ji->',ke_mat, dm[kind]))
    print('PySCF kinetic energy: {0}'.format(pyscfke))

    #####################################
    ## evaluate KE integral with VMC
    #####################################
    coords = pyqmc.initial_guess(mol, 1200, .7)
    warmup = 10
    start = time.time()
    df, coords = pyqmc.vmc(
        wf, coords, nsteps=30+warmup, tstep=1, accumulators={"energy": pyqmc.accumulators.EnergyAccumulator(mol)}, verbose=False,
    )
    print("VMC time", time.time()-start)
    df = pd.DataFrame(df)
    dfke = df["energyke"][warmup:]
    vmcke, err = dfke.mean(), dfke.std()/np.sqrt(len(dfke))
    print('VMC kinetic energy: {0} $\pm$ {1}'.format(vmcke, err))
    
    assert np.abs(vmcke-pyscfke) < 5 * err, \
        "energy diff not within 5 sigma ({0:.6f}): energies \n{1} \n{2}".format(5 * err, vmcke, pyscfke)


if __name__=="__main__":
    kind = 0
    nk = [2,2,2]
    #test_cubic(kind, nk)
    #test_RKS(kind, nk)
    test_noncubic(kind, nk)