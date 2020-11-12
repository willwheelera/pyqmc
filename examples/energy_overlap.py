import numpy as np
import pyqmc
import excited_state as es
from copy import deepcopy
import pyqmc.energy_overlaps as eo

savefiles = {
    "mf": "test.chk",
    "linemin": "linemin.hdf5",
    "excited1": "excited1.hdf5",
    "excited2": "excited2.hdf5",
    "linemin_vmc": "linemin_vmc.hdf5",
    "excited1_vmc": "excited1_vmc.hdf5",
    "excited2_vmc": "excited2_vmc.hdf5",
    "linemin_final": "linemin_final.hdf5",
    "excited1_final": "excited1_final.hdf5",
    "excited2_final": "excited2_final.hdf5",
}


def load_wfs(sys):
    wfs = []
    for key in ["linemin", "excited1", "excited2"]:
        wf = deepcopy(sys["wf"])
        pyqmc.read_wf(wf, savefiles[key])
        wfs.append(wf)
    return wfs


def compute_energy_overlaps():
    sys = es.pyqmc_from_hdf(savefiles["mf"])
    configs = pyqmc.initial_guess(sys["mol"], 1000)
    wfs = load_wfs(sys)
    # warmup
    _, configs = eo.sample_overlap(
        wfs, configs, sys["pgrad"], nblocks=1, nsteps=10, collect=[]
    )
    # evaluating
    block_avg, configs = eo.sample_overlap(
        wfs, configs, sys["pgrad"], nblocks=10, nsteps=10
    )
    data = eo.evaluate(block_avg, 0)
    print(data["H"])
    return data


def plot_H(data):
    import matplotlib.pyplot as plt

    v = np.amax(np.abs(data["H"]))
    im = plt.imshow(data["H"], vmin=-v, vmax=v, cmap="PRGn")
    plt.colorbar(im)
    plt.show()


if __name__ == "__main__":
    data = compute_energy_overlaps()
    plot_H(data)
