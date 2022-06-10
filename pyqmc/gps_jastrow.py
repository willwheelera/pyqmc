import numpy as np
from pyqmc import mc

# things to change: figure if there is a way to deal with epos with len(epos.shape)>2 without if statement
# when masking, only do relevant calculations. understand masking better.

# TODO rename Xtraining to Xsupport


class GPSJastrow:
    def __init__(self, mol, start_alpha, start_sigma):
        self.n_training = start_alpha.size
        self._mol = mol
        self.iscomplex = False
        self.parameters = {}
        # self.parameters["Xtraining"] = mc.initial_guess(mol,n_training).configs
        self.parameters["Xtraining"] = np.array(
            [[[0, 0, 0], [0, 0, 1.54 / 0.529177]], [[0, 0, 1.54 / 0.529177], [0, 0, 0]]]
        )
        self.parameters["sigma"] = start_sigma
        self.parameters["alpha"] = start_alpha

    def recompute(self, configs):
        self._configscurrent = configs
        self.nconfig, self.nelec = configs.configs.shape[:-1]
        e_partial = np.zeros(shape=(self.nconfig, self.n_training, self.nelec))

        # is there faster way than looping over e <- profile before pursing this
        for e in range(self.nelec):
            e_partial[:, :, e] = self._compute_partial_e(configs.configs[:, e, :])
        self._sum_over_e = e_partial.sum(axis=-1)
        self._e_partial = e_partial
        return self.value()

    # return phase and log(value) tuple
    def value(self):
        return (
            np.ones(self.nconfig),
            self._compute_value_from_sum_e(self._sum_over_e),
        )

    def _compute_partial_e(self, epos):
        y = epos[..., np.newaxis, np.newaxis, :] - self.parameters["Xtraining"]
        return np.einsum("...kl,...kl->...", y, y)

    def _compute_value_from_sum_e(self, sum_e):
        alpha, sigma = self.parameters["alpha"], self.parameters["sigma"]
        return np.dot(np.exp(-sum_e / (2.0 * sigma)), alpha)

    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        if saved_values is None:
            partial_e = self._compute_partial_e(epos.configs[mask])
            new_sum = self._sum_over_e[mask] + partial_e - self._e_partial[mask, :, e]
        else:
            partial_e, new_sum = saved_values
        self._sum_over_e[mask] = new_sum
        self._e_partial[:, :, e][mask] = partial_e
        self._configscurrent.move(e, epos, mask)

    def testvalue(self, e, epos, mask=None):
        if mask is None:
            mask = [True] * epos.configs.shape[0]

        old_means = self.value()[1][mask]
        new_partial_e = self._compute_partial_e(epos.configs[mask])
        prior_e_sum = self._sum_over_e[mask]
        if len(epos.configs.shape) > 2:
            e_sum2 = (
                prior_e_sum[:, np.newaxis, :]
                + new_partial_e
                - self._e_partial[mask, np.newaxis, :, e]
            )
            old_means = old_means[:, np.newaxis]
        else:
            e_sum2 = prior_e_sum + new_partial_e - self._e_partial[mask, :, e]
        means2 = self._compute_value_from_sum_e(e_sum2)
        return np.exp(means2 - old_means), (new_partial_e, e_sum2)

    def gradient_value(self, e, epos):
        return (self.gradient(e, epos), *self.testvalue(e, epos))

    def gradient(self, e, epos):
        # TODO very confusing to use same variable names in gradient() and laplacian() to mean different things
        partial_e = self._compute_partial_e(epos.configs)
        gradsum = (
            self.parameters["Xtraining"].sum(axis=1)
            - epos.configs[:, np.newaxis] * self.nelec
        )
        gradsum = np.transpose(gradsum, axes=(2, 0, 1))
        term1 = gradsum / self.parameters["sigma"]
        e_sum = self._sum_over_e + partial_e - self._e_partial[:, :, e]
        grads = np.sum(
            self.parameters["alpha"]
            * term1
            * np.exp(-e_sum / (2.0 * self.parameters["sigma"])),
            axis=-1,
        )
        return grads

    # first simplify the math, then the function
    def gradient_laplacian(self, e, epos):
        # Why is this variable called gradsum??
        gradsum = (
            self.parameters["Xtraining"].sum(axis=1)
            - epos.configs[:, np.newaxis] * self.nelec
        )
        partial_e = self._compute_partial_e(epos.configs)
        e_sum = self._sum_over_e + partial_e - self._e_partial[:, :, e]
        term1grad = gradsum / self.parameters["sigma"]
        term1lap = np.sum(term1grad ** 2, axis=-1)
        term2lap = -3 * self.nelec / self.parameters["sigma"]

        grads = np.einsum(
            "s,csd,cs->cd",
            self.parameters["alpha"],
            term1grad,
            np.exp(-e_sum / (2 * self.parameters["sigma"])),
        )
        laps = np.einsum(
            "s,cs,cs->c",
            self.parameters["alpha"],
            term1lap + term2lap,
            np.exp(-e_sum / (2 * self.parameters["sigma"])),
        )
        laps += np.sum(gradient ** 2, axis=-1)
        return gradient, laps

    def laplacian(self, e, epos):
        return self.gradient_laplacian(e, epos)[1]

    def pgradient(self):
        configs = self._configscurrent.configs
        alphader = np.exp(-0.5 * self._sum_over_e / self.parameters["sigma"])
        # print(self._sum_over_e.shape,"sumovere")
        # nc,ns,ne,3
        dersum = (
            self.parameters["Xtraining"] * self.nelec
            - configs.sum(axis=1)[:, np.newaxis, np.newaxis]
        )
        alphaderalpha = alphader * self.parameters["alpha"] / self.parameters["sigma"]
        Xder = np.einsum("csed,cs->csed", dersum, alphaderalpha)
        sigmader = np.einsum(
            "cs,cs->c",
            alphaderalpha / (2 * self.parameters["sigma"]),
            self._sum_over_e,
        )
        return {"alpha": alphader, "Xtraining": Xder, "sigma": sigmader}
