# Copyright (C) 2025 Enzo Busseti
#
# This file is part of GIIDE, Gaussian IID Embedding.
#
# GIIDE is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# GIIDE is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# GIIDE. If not, see <https://www.gnu.org/licenses/>.

import cvxpy as cp
import numpy as np
import scipy as sp

def gaussian_marginalizer(
        samples: np.array, n_nodes: int = 10, qnorm_extrapolate: float = 100.,
        slopes_regularizer: float = 1e-2, reject_slope_below: float = 1e-4,
        ):
    """Fit piece-wise linear transformer."""

    # we only want already normalized data (b/c of slope regularization)
    assert np.isclose(np.mean(samples), 0.)
    assert np.isclose(np.mean(samples**2), 1.)

    # temporary
    (qnorm, q), _ = sp.stats.probplot(samples, dist="norm")

    # this is main difference, use equispace in G space instead of quantiles
    nodes = np.linspace(np.min(qnorm),np.max(qnorm), n_nodes+1)
    nodes[0] = -qnorm_extrapolate
    nodes[-1] = qnorm_extrapolate

    # temporary plumbing, use CP
    a_s = cp.Variable(len(nodes)-1)
    b_s = cp.Variable(len(nodes)-1)

    if len(np.unique(samples)) == len(samples):
        counts = np.ones(len(samples))
    else:
        # we collapse the duplicates using approximate integral of
        # normal PDF (just mean of the qnorm's)

        q, counts = np.unique(np.sort(samples), return_counts=True)

        # this is inefficient, but hopefully it's done sparingly
        indexes = [0] + list(np.cumsum(counts))
        qnorm_orig = qnorm
        qnorm = []
        for i in range(len(indexes)-1):
            qnorm.append(np.mean(qnorm_orig[indexes[i]:indexes[i+1]]))
        qnorm = np.array(qnorm)

    # of course this can be simplified
    objective = 0.
    for i in range(len(nodes)-1):
        mask = (qnorm<=nodes[i+1])&(qnorm>nodes[i])
        if np.sum(mask) == 0:
            continue
        my_qnorm = qnorm[mask]
        my_q = q[mask]
        my_counts = counts[mask]
        objective += cp.sum_squares(
            cp.multiply(a_s[i] * my_qnorm + b_s[i] - my_q, my_counts))

    for _ in range(10):

        # this should be hopefully quite invariant
        objective += slopes_regularizer * cp.sum_squares(
            cp.diff(a_s)) * (len(qnorm) / len(nodes))

        # continuity at the nodes
        constraints = [
            a_s[i] * nodes[i+1] + b_s[i] == a_s[i+1] * nodes[i+1] + b_s[i+1]
            for i in range(len(nodes) - 2)
        ]

        # this causes program to be a QP instead of LS, most likely not needed
        # yes, especially now that we loop outside increasing the regularizer
        constraints += [a_s >= 0] # [a_s >= 1e-4] # do we want tiny slope instead?

        # defaults to OSQP, should be fine
        cp.Problem(cp.Minimize(objective), constraints).solve()

        # compute node values
        node_values = [a_s.value[i] * nodes[i] + b_s.value[i]
            for i in range(len(nodes) - 1)]
        node_values.append(a_s.value[-1] * nodes[-1] + b_s.value[-1])
        node_values = np.array(node_values)

        # compute slopes
        slopes = np.diff(node_values) / np.diff(nodes)

        if np.min(slopes) >= reject_slope_below:
            break

        print("One slope is too small, retrying with double regularizer")
        slopes_regularizer *= 2.

    return nodes, node_values
