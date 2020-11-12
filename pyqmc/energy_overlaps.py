import numpy as np
import pyqmc.energy as energy


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
    energy_f = pgrad.enacc(configs, wf)
    coulombpart = energy_f["total"] - (energy_f["ecp"] + energy_f["ke"])

    wf_enacc = WFEnergyAccumulator(pgrad.enacc.mol, pgrad.enacc.threshold)
    energies = [wf_enacc(configs, wf)["total"] + coulombpart for wf in wfs[:-1]]
    energies.append(energyf["total"])

    save_dat["energy_overlap"] = np.einsum(  # shape (wf, wf)
        "ik,jk->ij",
        normalized_values.conj(),
        normalized_values / denominator * np.asarray(energy_f),
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
        ecp_val = energy.get_ecp(mol, configs, wf, threshold)
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
