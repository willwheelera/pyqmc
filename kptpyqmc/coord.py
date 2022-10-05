import numpy as np
import kptpyqmc.distance as distance
import kptpyqmc.pbc as pbc
import copy


class KptElectron:
    """
    Represents the coordinates of a test electron position, for many walkers and
    potentially several different points.

    configs is a 2D or 3D vector with elements [config, point, dimension]
    wrap is same shape as configs
    lvec and dist will most likely be references to the parent object
    """

    def __init__(self, epos, k_indices, lattice_vectors, dist, wrap=None):
        self.configs = epos
        self.k_indices = k_indices
        self.wrap = wrap if wrap is not None else np.zeros_like(epos)
        self.lvecs = lattice_vectors
        self.dist = dist


def build_kpt_indices(nk, nup, ndown):
    rangenk = list(range(nk))
    parts = []
    for nelec in [nup, ndown]:
        q_elec, r_elec = np.divmod(nelec, nk)
        lens = np.ones(nk, dtype=int) * int(q_elec)
        if r_elec > 0:
            inds = np.random.choice(np.arange(nk), r_elec)
            lens[inds] += 1
        parts.extend([[i] * l for i, l in enumerate(lens)])
    return np.concatenate(parts)


class KptConfigs:
    def __init__(self, configs, lattice_vectors, wrap=None, k_indices=None, nk=None, nup=None):
        """
        kpts define the qmc simulation cell
        lattice_vectors: are for the primitive cell
        wrap: store to apply phase, since we aren't tracking position outside primitive cell
        """
        configs, wrap_ = pbc.enforce_pbc(lattice_vectors, configs)
        self.configs = configs
        self.wrap = wrap_
        if wrap is not None:
            self.wrap += wrap
        if nup is None:
            nup = int(np.ceil(configs.shape[1] / 2))
        ndown = configs.shape[1] - nup
        if k_indices is None:
            if nk is None:
                raise ValueError("k_indices and nk can't both be None")
            k_indices = build_kpt_indices(nk, nup, ndown)
        self.k_indices = k_indices
        self.lvecs = lattice_vectors
        self.dist = distance.MinimalImageDistance(lattice_vectors)

    def electron(self, e):
        return KptElectron(
            self.configs[:, e],
            self.k_indices[e],
            self.lvecs,
            self.dist,
            wrap=self.wrap[:, e],
        )

    def mask(self, mask):
        return KptConfigs(
            self.configs[mask], self.lvecs, k_indices=self.k_indices, wrap=self.wrap[mask]
        )

    def make_irreducible(self, e, vec, mask=None):
        """
        Input: a (nconfig, 3) vector or a (nconfig, N, 3) vector
        Output: A Kpt Electron
        """
        if mask is None:
            mask = np.ones(vec.shape[0:-1], dtype=bool)
        epos_, wrap_ = pbc.enforce_pbc(self.lvecs, vec[mask])
        epos = vec.copy()
        epos[mask] = epos_
        wrap = self.wrap[:, e, :].copy()
        if len(vec.shape) == 3:
            wrap = np.repeat(self.wrap[:, e][:, np.newaxis], vec.shape[1], axis=1)
        wrap[mask] += wrap_
        return KptElectron(
            epos, self.k_indices[e], self.lvecs, wrap=wrap, dist=self.dist
        )

    def move(self, e, new, accept):
        """
        Change coordinates of one electron
        Args:
          e: int, electron index
          new: KptConfigs with (nconfig, 3) new coordinates
          accept: (nconfig,) boolean for which configs to update
        """
        self.configs[accept, e, :] = new.configs[accept, :]
        self.wrap[accept, e, :] = new.wrap[accept, :]

    def move_all(self, new, accept):
        """
        Change coordinates of all electrons
        Args:
          new: KptConfigs with configs.shape new coordinates
          accept: (nconfig,) boolean for which configs to update
        """
        self.configs[accept] = new.configs[accept]
        self.wrap[accept] = new.wrap[accept]

    def resample(self, newinds):
        """
        Resample configs by new indices (e.g. for DMC branching)
        Args:
          newinds: (nconfigs,) array of indices
        """
        self.configs = self.configs[newinds]
        self.wrap = self.wrap[newinds]

    def split(self, npartitions):
        """
        Split configs into npartitions new configs objects for parallelization
        Args:
          npartitions: int, number of partitions to divide configs into
        Returns:
          configslist: list of new configs objects
        """
        clist = np.array_split(self.configs, npartitions)
        wlist = np.array_split(self.wrap, npartitions)
        return [KptConfigs(c, self.lvecs, w, self.k_indices) for c, w in zip(clist, wlist)]

    def join(self, configslist, axis=0):
        """
        Merge configs into this object to collect from parallelization
        Args:
          configslist: list of KptConfigs objects
        """
        self.configs = np.concatenate([c.configs for c in configslist], axis=axis)
        self.wrap = np.concatenate([c.wrap for c in configslist], axis=axis)

    def copy(self):
        return copy.deepcopy(self)

    def reshape(self, shape):
        self.configs = self.configs.reshape(shape)
        self.wrap = self.wrap.reshape(shape)

    def initialize_hdf(self, hdf):
        hdf.create_dataset(
            "configs",
            self.configs.shape,
            chunks=True,
            maxshape=(None, *self.configs.shape[1:]),
        )
        hdf.create_dataset(
            "wrap", self.wrap.shape, chunks=True, maxshape=(None, *self.wrap.shape[1:])
        )
        hdf.create_dataset(
            "k_indices",
            self.k_indices.shape,
            chunks=True,
            maxshape=(None, *self.k_indices.shape[1:]),
        )

    def to_hdf(self, hdf):
        hdf["configs"].resize(self.configs.shape)
        hdf["configs"].resize(self.wrap.shape)
        hdf["configs"][...] = self.configs
        hdf["wrap"][...] = self.wrap
        hdf["k_indices"][...] = self.k_indices

    def load_hdf(self, hdf):
        # The ... seems to be necessary to avoid changing the dtype and screwing up
        # pyscf's calls.
        self.configs[...] = hdf["configs"][()]
        self.wrap[...] = hdf["wrap"][()]
        self.k_indices[...] = hdf["k_indices"][()]
    
    def select(self, ki):
        s = self.k_indices == ki
        return KptConfigs(
            self.configs[:, s], self.lvecs, k_indices=self.k_indices[s], wrap=self.wrap[:, s]
        )
        
