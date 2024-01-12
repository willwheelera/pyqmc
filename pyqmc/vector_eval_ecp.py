import numpy as np
import copy
import scipy.spatial.transform
from pyqmc.eval_ecp import get_P_l, get_v_l, ecp_mask


def ecp(mol, configs, wf, threshold, naip=None):
    """
    :returns: ECP value, summed over all the electrons and atoms.
    """
    nconf, nelec = configs.configs.shape[0:2]
    ecp_tot = np.zeros(nconf, dtype=wf.dtype)
    if mol._ecp != {}:
        for atom in mol._atom:
            if atom[0] in mol._ecp.keys():
                for e in range(nelec):
                    ecp_tot += ecp_ea(mol, configs, wf, e, atom, threshold, naip)[
                        "total"
                    ]
    return ecp_tot


def ecp_ea(mol, configs, wf, e, atom, threshold, naip=None):
    """
    :returns: the ECP value between electron e and atom at, local+nonlocal.
    TODO: update documentation
    """
    nconf = configs.configs.shape[0]
    ecp_val = np.zeros(nconf, dtype=wf.dtype)

    at_name, apos = atom
    apos = np.asarray(apos)

    r_ea_vec = configs.dist.dist_i(apos, configs.configs[:, e, :]).reshape((-1, 3))
    r_ea = np.linalg.norm(r_ea_vec, axis=-1)

    l_list, v_l = get_v_l(mol, at_name, r_ea)
    mask, prob = ecp_mask(v_l, threshold)
    masked_v_l = v_l[mask]
    masked_v_l[:, :-1] /= prob[mask, np.newaxis]

    # Use masked objects internally
    r_ea = r_ea[mask]
    r_ea_vec = r_ea_vec[mask]
    P_l, r_ea_i = get_P_l(r_ea, r_ea_vec, l_list, naip)

    # Note: epos_rot is not just apos+r_ea_i because of the boundary;
    # positions of the samples are relative to the electron, not atom.
    epos_rot = np.repeat(
        configs.configs[:, e, :][:, np.newaxis, :], P_l.shape[1], axis=1
    )
    epos_rot[mask] = (configs.configs[mask, e, :] - r_ea_vec)[:, np.newaxis] + r_ea_i

    epos = configs.make_irreducible(e, epos_rot, mask)
    ratio = wf.testvalue(e, epos, mask)[0]

    # Compute local and non-local parts
    ecp_val[mask] = np.einsum("ij,ik,ijk->i", ratio, masked_v_l, P_l)
    ecp_val += v_l[:, -1]  # local part
    return {
        "total": ecp_val,
        "v_l": masked_v_l,
        "local": v_l[:, -1],
        "P_l": P_l,
        "ratio": ratio,
        "epos": epos,
        "mask": mask,
    }

