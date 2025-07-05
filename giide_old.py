# Copyright (C) 2025 Enzo Busseti
#
# This file is part of GIIDE (Gaussian IID Embedding).
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


import numpy as np
import pandas as pd
import scipy as sp
from scipy.interpolate import RBFInterpolator

class GaussianTransformer:
    """Transform to Gaussian."""
    def __init__(self, raw_data: pd.Series):
        ###############
        # the ECDF part is easy to extract; it's a np.sort, np.unique,
        # np.cumsum; really so a sort and then a pass O(n) with very simple ops
        ###############
        ecdf = sp.stats.ecdf(raw_data).cdf
        self.q, p = ecdf.quantiles, ecdf.probabilities
        # p_corr = (p - p[0]/2)
        p1 = np.concatenate([[0], p])
        p2 = (p1[1:] + p1[:-1]) / 2
        # same as
        # p2 = p - 0.5 / len(raw_data)
        ###############
        # The ppf is sp.special function ndtri, Scipy contains direct C
        # implementation of it.
        # This op can be greatly sped up since we know input is sorted
        # and we only care about change (integral of derivative of PDF) with
        # often constant step
        ###############
        self.q_norm = sp.stats.norm.ppf(p2)
        # plt.scatter(q_norm, q); plt.show()

    ###############
    # Using interpolate for simplicity; it works (invertible) with steps in
    # the ECDF (non-unique obs); fwd pass is really already done above no need
    # to recompute; bwd pass requires bisection search and lin interpolate b/w
    # 2 values, O(log2(n)), where n is input dataset len, look at np
    # implementation not sure if already uses sorted inputs
    ###############
    
    def transf(self, input_data: pd.Series):
        return pd.Series(
            np.interp(input_data, self.q, self.q_norm), index=input_data.index)
        # plt.scatter(transf, raw_data); plt.show()
    
    def transf_back(self, input_data: pd.Series):
        return pd.Series(
            np.interp(input_data, self.q_norm, self.q), index=input_data.index)

    # def transf_back(self, input_data: pd.Series):
    #     return pd.Series(
    #         RBFInterpolator(
    #             self.q_norm.reshape(len(self.q_norm), 1),
    #             self.q.reshape(len(self.q), 1),
    #             neighbors = 10,
    #             kernel = 'linear',
    #             smoothing = 1.,
    #             # kernel='gaussian',
    #             # epsilon=10,
    #             # smoothing=0.0,
    #         )(input_data.values.reshape(len(input_data), 1)).flatten(),
    #     index=input_data.index)

# TEMPORARY PLUMBING

def nonlin(input):
    transformers = []
    output = pd.DataFrame(dtype=float)
    for column in input.columns:
        t = GaussianTransformer(input[column])
        transformers.append(t)
        output[column] = t.transf(input[column])

    print(output.describe())
    return output

def lin(input):
    #############
    # This should be redone using QR; also pySPQR to handle a little
    # bit sparsity too (can't guarantee sparsity holds, but worth adding it)
    # Also there should be logic to handle zero singular values / zeros on
    # the diagonal or R, reducing dimensionality of embedding
    #############
    u,s,v = np.linalg.svd(input, full_matrices=False)
    print('SINGULAR VALUES', s)
    loss = 2 * (s[0] - s[-1]) / (s[0] + s[-1])
    print('LOSS', loss)
    return pd.DataFrame(input @ v.T, index=input.index), loss

def transform(input_dataframe, max_iters=100, target_loss=1e-3):
    #############
    # No need to recompute Gaussian quantiles each time; most of the times
    # they are identical.
    #############
    result = pd.DataFrame(input_dataframe, copy=True)
    losses = []
    for i in range(max_iters):
        print('ITER', i)
        result, loss = lin(nonlin(result))
        losses.append(loss)
        if loss < target_loss:
            break
    return result

