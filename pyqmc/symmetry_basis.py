# 0. Determine symmetry of structure
# 1. Define symmetry operators on R^3
# 2. Estimate <a|g|b> for orbitals a, b and symm op g 
#    a. for each sampled point r generate all symmetry copies gr
#    b. evaluate a(gr), b(gr) (all orbs) for all g [and keep index by  g]
# 3. 
#    do these need to be orthogonal?  no, use S^-1; have all orbitals, can do in accumulator with the corresponding row i of S^-1.
#    a. constructing a representation on AO basis. that means 
#    b. p(r) = \sum_i c_i a_i(r),
#    c. p(gr) = \sum_i c_i a_i(gr) = \sum_i c_i [\sum_j g_{ij} a_j(r) ] = \sum_j [\sum_i c_i g_{ij}] a_j(r)
#    d. want a_j(gr) = \sum_k g_{jk} a_k(r)
#           so project <a_i| g | a_j> = \sum_k g_{jk} <a_i|a_k> = g_jk S_{ki}
#           so g_{jk} = <a_i | g | a_j> S^{-1}
#    e. do invariant atoms get exactly 1?

#   General numerical approach for full AO rep: 
#       generate 2l+1 random points r, and gr for all g
#       eval a_m(gr) for all m, for all g, for all r
#       a(gr) = c_0 a_0(r) + c_1 a_1(r) + ... + c_m a_m(r)
#       a_n(gr) = \sum_m C_{nm}^g a_m(r)
#       solve linear system (should work for full AO rep)
#       do the orbs need to be orthogonal?  no. need S to eval <a_k | g | a_n> = S C.T, a_n(gr) = C_{nm} a_m(r)
#   will be exact

# want basis transformation from AO space to irrep space
# - can project each basis (AO) vec onto each irrep
# - need characters for projection 
# 
# Tr(G) = \sum_\Omega \chi^\Omega(G)
# \Omega is over all component irreps in the representation (can repeat)
# Orthogonality: \sum_G \chi^{\Gamma}(G) \chi^{\Gamma'}(G) = |G| \delta_{\Gamma\Gamma'}
# Tr(G)^2 = \sum_{ij} \chi^{\Omega_i}(G) \chi^{\Omega_j}(G)
# \sum_G Tr(G)^2 = \sum_{ij} \chi^{\Omega_i}(G) \chi^{\Omega_j}(G) |G| \delta_{\Omega_i\Omega_j}
# if \Gamma_i appears n times, this contributes n^2 to the sum (assuming over all ij, not just pairs)
# so \sum_G Tr(G)^2 = |G| \sum_{i} m_{\Gamma_i}^2 ## multiplicity
# size of irreps \sum_{i} dim(\Omega_i) = dim(rep) = n_AO
# size of irreps \sum_{i} dim(\Gamma_i) m_{\Gamma_i} = dim(rep) = n_AO
# for any i, \sum_G \chi^{\Gamma_i}(G)^2 = |G| ## N_irrep equations, N_irrep*|G| unknowns 
# but solutions restricted to integers

# 1. For one AO, apply all group operations. The set of nonzero AO coeffs over all the Gs makes a smaller rep
# 
import numpy as np
import scipy
import pyqmc.api as pyq
from pyscf import gto, scf


def generate_tags_aos(mol, xyz_rep):
    """

    """
    nG = len(xyz_rep)
    ao_labels = mol.ao_labels()
    nao = len(ao_labels)
    nsamples = nao // nG + 1
    nelec = sum(mol.nelec)
    print(nao, nsamples, nelec, nG)

    # sample points to solve for AO representation
    r = pyq.initial_guess(mol, (nsamples+1) // nelec + 1).configs.reshape(-1, 3)[:nsamples]
    gr = np.einsum("gij,kj->kgi", xyz_rep, r)
    aos = mol.eval_gto("GTOval_sph", gr.reshape(-1, 3)).reshape((nsamples, nG, nao))
    groupmap = compute_group_map(xyz_rep)

    # each tag is invariant under G and has its own representation
    # search_ao_label(tag) gives the indices for that tag
    tags = make_tags(ao_labels) # e.g. "H 2p"
    return tags, aos, groupmap

def id_mo_irreps(mol, xyz_rep, ovlp, mo_coeff0, character):
    """
    mol: the pyscf mol or cell
    xyz_rep: group representation on R^3 (|G|, 3, 3)
    character: character table (n_irrep, |G|)
    mo_coeff0: starting/input MOs
    ovlp: from mf.get_ovlp()
    returns:
        irrep_indicator (n_irrep, nao) boolean array indexing MOs
    """
    tags, aos, k = generate_tags_aos(mol, xyz_rep)
    nao = len(ovlp)
    nG = character.shape[1]
    SV = ovlp @ mo_coeff0
    mo_rep_diag = np.zeros((nG, nao)) # diagonal of mo representation of G
    print("ovlp", ovlp.shape)
    for tag in tags:
        inds = mol.search_ao_label(tag)
        tag_ao_rep = _solve_coefficients(aos[:, :, inds], k)
        # For MO basis
        SVinds = SV[inds]
        mc0inds = mo_coeff0[inds]
        mo_rep_diag += np.einsum("im,gij,jm->gm", mc0inds.conj(), tag_ao_rep, SVinds)
    dotprod = np.einsum("rg,gj->rj", character, mo_rep_diag) * character[:, :1]/ nG
    irrep_indicator = dotprod > 1e-12

    return irrep_indicator

def id_ao_irreps(mol, xyz_rep, ovlp, mo_coeff0, character):
    """
    returns:
        irrep_indicator (n_irrep, nao) boolean array indexing MOs
        basis (nao, nao) columns are basis vectors
    """
    tags, aos, k = generate_tags_aos(mol, xyz_rep)
    nao = len(ovlp)
    print("ovlp", ovlp.shape)
    ao_irrep_basis = np.zeros((nao, nao))
    irrep_indicator = np.zeros((len(character), nao), dtype=bool)
    for tag in tags:
        inds = mol.search_ao_label(tag)
        tag_ao_rep = _solve_coefficients(aos[:, :, inds], k)
        # For AO basis
        # project AO basis onto irreps. In AO basis, the elements are the identity matrix
        # projection P = \sum_g \chi_g C_g v; columns are the projected vectors in AO basis
        # Acting with the group elements is tag_ao_rep @ I = tag_ao_rep
        tag_ovlp = ovlp[inds][:, inds]
        dotprod = np.einsum("rg,gij,jk->rik", character, tag_ao_rep, tag_ovlp)
        dotprod *= character[:, :1, np.newaxis]/ character.shape[1]
        # projection will have zero eigenvalues, SVD to remove
        # we don't want parameters in AO basis, want a good basis -> use U, discard V 
        U, S, Vh = np.linalg.svd(dotprod)
        select = S > 1e-8
        tmp = ao_irrep_basis[inds]
        counter = 0
        for i, (u, sel) in enumerate(zip(U, select)): # for each irrep
            sel_inds = inds[counter:counter+sel.sum()]
            counter += sel.sum()
            irrep_indicator[i, sel_inds] = 1
            tmp[:, sel_inds] = u[:, sel]
        ao_irrep_basis[inds] = tmp
    #T = np.linalg.inv(ovlp)
    #ao_irrep_basis = T @ ao_irrep_basis

    return irrep_indicator, ao_irrep_basis
    

def recompute_mo_coeffs(self, mo_coeff0, occ, irrep_indicator, params):
    # This goes in the irrep_orbitals object
    if not hasattr(self, "mo_coeff"):
        self.mo_coeff = np.zeros((len(mo_coeff0), sum(occ)))
    for i, ind in enumerate(irrep_indicator):
        self.mo_coeff[:, ind[occ]] = mo_coeff0[:, ind] @ params[i] # params[i] (irrep_inds_i, occ_inds_i)

def generate_initial_params(occ, irrep_indicator):
    # This goes in the irrep_orbitals constructor 
    # nrows: number of mos in irrep = ind.sum()
    # cols: occupied mos in irrep = occ[ind]
    params = [np.eye(ind.sum())[:, occ[ind]] for ind in irrep_indicator]


def _solve_coefficients(aos, k):
    """
    Parameters: 
        aos: subset of aos shape (nsamples, nG, nao)
        k: group table. g_{k[i, j]} = g_i @ g_j
    Returns:
        C: (nG, nao, nao) group representation on this subset of aos
    
    eq: a_n(g_i g_j r) = sum C_{nm}^i a_m(g_j r)
    solve C_{mn}^i group op i. multiple data points by group op j. k=k[i, j]
    aos[:, k[i], n] = sum C_{nm}^i aos[:, :, m]
    fit n, i: aos[:, :, m] @(m) C_{nm}^i     = aos[:, k[i], n]
              (nsam, nG, nao), (nao, nG, nao), (nsam, nG, nG, nao)
    lstsq: solve a @ x = b for x. a (neqns, nparams), b (neqns, nfits), c (nparams, nfits)
    neqns = nsamples * nG
    nparams = naos
    nfits = nG * naos
    """
    nsamples, nG, nao = aos.shape
    # simpler test
    if False:
        a = aos[:, 0] # just samples, identity op
        b = aos.reshape(nsamples, nG*nao)
        x, res, rank, sing = np.linalg.lstsq(a, b, rcond=None)
        print("res", res)
        # solve A_{lm} C_{m(gn)} = B_{l(gn)}
        # want C^g_{nm}
        C = np.moveaxis(x.reshape(nao, nG, nao), 0, 2) 
        test_rep(C, k); quit()
        return C


    a = aos.reshape(-1, nao)
    b = aos[:, k.T, :].reshape(nsamples * nG, nG * nao)
    x, res, rank, sing = np.linalg.lstsq(a, b, rcond=None)
    C = np.moveaxis(x.reshape(nao, nG, nao), 0, 2)
    return C
    

def generate_regular_rep(xyz_rep):
    r = np.random.randn(3)
    r = r / np.linalg.norm(r) # unit vector - doesn't matter
    gr = np.einsum("gij,j->gi", xyz_rep, r)
    ggr = np.einsum("hki,gi->hgk", xyz_rep, gr)
    ops = (np.linalg.norm(ggr[:, :, np.newaxis] - gr, axis=-1) < 1e-12).astype(float)
    return ops
    

def compute_group_map(ops):
    # ops is (|G|, n, n) n can be whatever, usually 3
    # apply group op i and group op j ... equivalent to which index?
    # apply is matrix multiply
    nG = len(ops)
    product_array = np.einsum("ilm,jmn->ijln", ops, ops)
    # asking: composing elements i, j, which element k is the same matrix
    difference = product_array[:, :, np.newaxis] - ops[np.newaxis, np.newaxis] # ijkln 
    indicator = np.linalg.norm(difference, axis=(-2, -1)) < 1e-12
    assert np.all(np.linalg.norm(indicator, axis=-1) > 0)
    k = np.argmax(indicator, axis=-1)
    return k

def make_tags(ao_labels):
    tags = []
    for ao_label in ao_labels:
        l = ao_label.split()
        tag = l[1] + " " + l[2][:2]
        tags.append(tag)
    tags = list(set(tags))
    tags.sort()
    return tags

def organize_block_diagonal(A, tol=1e-2):
    # For a single matrix that is permuted block diagonal, permute the basis to make it block diagonal
    from matplotlib.animation import ArtistAnimation 
    import matplotlib.pyplot as plt
    B = A.copy()
    current_col = 0
    end_permutation = np.arange(len(A))
    blocksizes = []
    artists = []
    #fig, ax = plt.subplots()
    v = 2
    #ax.imshow(B, vmax=v, vmin=-v, cmap="PRGn")
    for i in range(len(A)):
        newblock = np.all(np.abs(B[i, :current_col]) <tol)
        current_col += newblock

        permutation, current_col, blocksize = _arrange_row(B, i, current_col, tol)
        blocksize += newblock
        B = B[permutation][:, permutation]
        end_permutation = end_permutation[permutation]

        #im = ax.imshow(B, vmax=v, vmin=-v, cmap="PRGn", animated=True)
        #artists.append([im])

        if blocksize==0 and newblock:
            print("row", i)
            print(B[i])
            print(np.around(B[i], 2))
            print(permutation)
            print(B[permutation][:, permutation][i])
            quit()
        if newblock:
            blocksizes.append(blocksize)
        else:
            blocksizes[-1] += blocksize

    #ani = ArtistAnimation(fig, artists, interval=200, blit=True, repeat_delay=1000)
    #plt.show()

    return B, end_permutation, blocksizes

def _arrange_row(B, i, aftercol, tol=1e-10):
    r = B[i, aftercol:]
    c = B[aftercol:, i]
    r = np.abs(r) > tol
    c = np.abs(c) > tol
    v = np.logical_or(r, c)

    nonzero = np.nonzero(v)[0] + aftercol
    zero = np.nonzero(v==0)[0] + aftercol
    before = np.arange(aftercol)
    n = len(nonzero)
    return np.concatenate([before, nonzero, zero]), aftercol + n, n

###########################################################
def test_irrep_indicator(mol, mo_coeff, irrep_indicator, xyz_rep):
    nG = len(xyz_rep)
    nconf = 1
    r = pyq.initial_guess(mol, nconf).configs.reshape(-1, 3)
    gr = np.einsum("gij,kj->gki", xyz_rep, r)
    aos = mol.eval_gto("GTOval_sph", gr.reshape(-1, 3)).reshape((nG, len(r), -1))
    mos = aos @ mo_coeff

    print(irrep_indicator.sum(axis=0))
    for i, ind in enumerate(irrep_indicator):
        print(i, ind.shape, ind.sum())
        vals = mos[:, 0, ind]
        print(np.around(vals/vals[0], 3))


def test_rep(C, k):
    # a_n(gr) = sum_m C_nm a_m(r)
    # so a_n(hgr) = sum_m C_nm a_m(gr) = sum_lm C_nm C_ml a_l(r)
    # i.e. C_nl = sum_m C_nm C_ml (hg = h compose g)
    # C is (nG, nao, nao)
    # k is (nG, nG)
    diffs = []
    for i in range(len(k)):
        prod = np.einsum("ij,gjk->gik", C[i], C)
        d = prod - C[k[i]]
        #print(np.around(np.stack([prod, C[k[i]]], axis=1).reshape(len(k)* 2, -1), 1))
        #print(np.around(d.reshape(len(k), -1), 2))
        diff = np.linalg.norm(d, axis=(1, 2)) # k[i] is k_ij group element that is C_i C_j
        diffs.append(diff)
    print(np.around(diff, 3))

def test_group_map():
    klein = np.zeros((4, 2, 2))
    klein[0] = np.diag([1, 1])
    klein[1] = np.diag([-1, 1])
    klein[2] = np.diag([1, -1])
    klein[3] = np.diag([-1, -1])
    compute_group_map(klein)


def test_block_diagonal():
    A = np.random.random()
    B = np.random.random((2,2))
    C = np.random.random((2,2))
    D = np.random.random((3,3))
    M = scipy.linalg.block_diag(A, B, C, D)
    p = np.random.permutation(len(M))
    M_ = M[p][:, p]
    out, q = organize_block_diagonal(M_)
    print((M>0).astype(int))
    print((out>0).astype(int))
    print((M_[q][:, q]>0).astype(int))

    A = np.diag(np.arange(8))
    out, q = organize_block_diagonal(A)
    print(out)

