import numpy as np
import pyqmc.energy as energy
from pyqmc.mc import limdrift


def collect_energy_overlaps(wfs, configs, pgrad):
    r""" Like collect_overlap but just for the energy overlap, assuming that 
    configs are distributed according to 
    
    .. math:: \rho \propto \sum_i |\Psi_i|^2

    The key 'energy_overlap' is

    `energy_overlap`:

    .. math:: \langle \Psi_i | H | \Psi_j \rangle = \left\langle \frac{\Psi_i^* \Psi_j}{\rho} \frac{H \Psi_j}{\Psi_j} \right\rangle_{\rho}
    """
    phase, log_vals = [np.array(x) for x in zip(*[wf.value() for wf in wfs])]
    log_vals = np.real(log_vals)  # should already be real
    ref = np.max(log_vals, axis=0)
    save_dat = {}
    denominator = np.sum(np.exp(2 * (log_vals - ref)), axis=0)
    normalized_values = phase * np.exp(log_vals - ref)

    # don't recompute coulomb part for each wf - especially ewald for periodic
    energy_f = pgrad.enacc(configs, wfs[-1])
    coulombpart = energy_f["total"] - (energy_f["ecp"] + energy_f["ke"])

    wf_enacc = WFEnergyAccumulator(pgrad.enacc.mol, pgrad.enacc.threshold)
    energies = [wf_enacc(configs, wf)["total"] + coulombpart for wf in wfs[:-1]]
    energies.append(energy_f["total"])

    save_dat["energy_overlap"] = np.einsum(  # shape (wf, wf)
        "ik,jk->ij",
        normalized_values.conj(),
        normalized_values / denominator * np.asarray(energies),
    ) / len(ref)
    save_dat["normalization"] = np.sum(
        np.exp(2 * (log_vals - ref)) / denominator, axis=1
    )

    return save_dat


class WFEnergyAccumulator:
    """returns just ecp and kinetic energy of each configuration in a dictionary. 
  Keys and their meanings can be found in energy.energy """

    def __init__(self, mol, threshold=10):
        self.mol = mol
        self.threshold = threshold

    def __call__(self, configs, wf):
        ecp_val = energy.get_ecp(self.mol, configs, wf, self.threshold)
        ke = energy.kinetic(configs, wf)
        return {"ke": ke, "ecp": ecp_val, "total": ke + ecp_val}

    def avg(self, configs, wf):
        return {k: np.mean(it, axis=0) for k, it in self(configs, wf).items()}

    def keys(self):
        return self.shapes.keys()

    def shapes(self):
        return {"ke": (), "ecp": (), "total": ()}


def evaluate(return_data, warmup):
    avg_data = {}
    for k, it in return_data.items():
        avg_data[k] = np.average(it[warmup:], axis=0)

    N = avg_data["normalization"]
    Nij = np.sqrt(np.outer(N, N))
    Hij = avg_data["energy_overlap"] / Nij
    return {"N": N, "H": Hij}


def sample_overlap_worker(wfs, configs, pgrad, nsteps, tstep=0.5, collect=[]):
    r""" Run nstep Metropolis steps to sample a distribution proportional to 
    :math:`\sum_i |\Psi_i|^2`, where :math:`\Psi_i` = wfs[i]
    """
    nconf, nelec, _ = configs.configs.shape
    for wf in wfs:
        wf.recompute(configs)
    block_avg = {}
    block_avg["acceptance"] = np.zeros(nsteps)
    for n in range(nsteps):
        for e in range(nelec):  # a sweep
            # Propose move
            grads = [np.real(wf.gradient(e, configs.electron(e)).T) for wf in wfs]
            grad = limdrift(np.mean(grads, axis=0))
            gauss = np.random.normal(scale=np.sqrt(tstep), size=(nconf, 3))
            newcoorde = configs.configs[:, e, :] + gauss + grad * tstep
            newcoorde = configs.make_irreducible(e, newcoorde)

            # Compute reverse move
            grads = [np.real(wf.gradient(e, newcoorde).T) for wf in wfs]
            new_grad = limdrift(np.mean(grads, axis=0))
            forward = np.sum(gauss ** 2, axis=1)
            backward = np.sum((gauss + tstep * (grad + new_grad)) ** 2, axis=1)

            # Acceptance
            t_prob = np.exp(1 / (2 * tstep) * (forward - backward))
            wf_ratios = np.array(
                [np.abs(wf.testvalue(e, newcoorde)) ** 2 for wf in wfs]
            )
            log_values = np.real(np.array([wf.value()[1] for wf in wfs]))
            weights = np.exp(2 * (log_values - log_values[0]))

            ratio = t_prob * np.sum(wf_ratios * weights, axis=0) / weights.sum(axis=0)
            accept = ratio > np.random.rand(nconf)
            block_avg["acceptance"][n] += accept.mean() / nelec

            # Update wave function
            configs.move(e, newcoorde, accept)
            for wf in wfs:
                wf.updateinternals(e, newcoorde, mask=accept)

        # Collect rolling average
        for func in collect:
            save_dat = func(wfs, configs, pgrad)
            for k, it in save_dat.items():
                if k not in block_avg:
                    block_avg[k] = np.zeros((*it.shape,), dtype=it.dtype)
                block_avg[k] += save_dat[k] / nsteps

    return block_avg, configs


def sample_overlap(
    wfs, configs, pgrad, nblocks=10, nsteps=10, collect=(collect_energy_overlaps,)
):
    r"""
    Sample 

    .. math:: \rho(R) = \sum_i |\Psi_i(R)|^2

    `pgrad` is expected to be a gradient generator. returns data as follows:

    """

    return_data = {}
    for block in range(nblocks):
        block_avg, configs = sample_overlap_worker(
            wfs, configs, pgrad, nsteps, collect=collect
        )
        # Blocks stored
        for k, it in block_avg.items():
            if k not in return_data:
                return_data[k] = np.zeros((nblocks, *it.shape), dtype=it.dtype)
            return_data[k][block, ...] = it.copy()
    return return_data, configs
