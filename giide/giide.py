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

from abc import ABC, abstractmethod
import logging

import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Layer(ABC):
    """Base class for GIIDE layer."""
    
    @abstractmethod
    def fit_transform(self, data: np.array) -> np.array:
        """Fit the layer with a 2D array and (forward) transform it."""
        pass

    @abstractmethod
    def transform(self, data: np.array) -> np.array:
        """Forward transform a 2D array."""
        pass

    @abstractmethod
    def transform_back(self, data: np.array) -> np.array:
        """Backward transform a 2D array."""
        pass

    # TODO: also abstract methods for fw/bw differentiation

class RandomRotation(Layer):
    """Random rotation."""

    # placeholder
    rotation = np.array([[]])

    # TODO: implement some way to seed random draw

    # TODO: Figure out if O(n) Householder is enough, probably it is?

    @staticmethod
    def random_rotation(size):
        a = np.random.randn(size,size),
        q,_ = np.linalg.qr(a)
        return q[0]
    
    def fit_transform(self, data: np.array) -> np.array:
        """Fit the layer with a 2D array and (forward) transform it."""
        self.rotation = self.random_rotation(size=data.shape[1])
        return self.transform(data=data)

    def transform(self, data: np.array) -> np.array:
        """Forward transform a 2D array."""
        data @= self.rotation
        return data

    def transform_back(self, data: np.array) -> np.array:
        """Backward transform a 2D array."""
        data @= self.rotation.T
        return data

class SVD(Layer):
    """SVD of input data."""

    # placeholder
    v = np.array([[]])
    s = np.array([])

    def fit_transform(self, data: np.array) -> np.array:
        """Fit the layer with a 2D array and (forward) transform it."""
        u, s, v = np.linalg.svd(data, full_matrices=False)
        self.v = v
        self.s = s / np.sqrt(len(data))
        return u * np.sqrt(len(data))

    def transform(self, data: np.array) -> np.array:
        """Forward transform a 2D array."""
        data @= self.v.T
        data /= self.s
        return data

    def transform_back(self, data: np.array) -> np.array:
        """Backward transform a 2D array."""
        data *= self.s
        data @= self.v
        return data

class RandomHouseholder(Layer):
    """Random Householder reflection."""

    # placeholder
    reflection_vector = np.array([])

    # TODO: implement some way to seed random draw

    @staticmethod
    def random_reflection(size):
        reflection_vector = np.random.randn(size)
        reflection_vector /= np.linalg.norm(reflection_vector)
        return reflection_vector
    
    def fit_transform(self, data: np.array) -> np.array:
        """Fit the layer with a 2D array and (forward) transform it."""
        self.reflection_vector = self.random_reflection(size=data.shape[1])
        return self.transform(data=data)

    def transform(self, data: np.array) -> np.array:
        """Forward transform a 2D array."""
        # TODO this can be probably be written in more efficient numpy
        data -= np.outer(
            data @ self.reflection_vector, 2 * self.reflection_vector)
        return data

    def transform_back(self, data: np.array) -> np.array:
        """Backward transform a 2D array."""
        return self.transform(data)


def compute_normal_quantiles(nobs: int) -> np.array:
    """Compute normal quantiles for non-duplicate inputs."""
    p = (0.5 + np.arange(float(nobs))) / float(nobs)
    return sp.stats.norm.ppf(p)

class FullGaussianMarginal(Layer):
    """Full transformation of the marginals to normals.

    .. warning::

        This will give incorrect results if any column has duplicate values.

    .. warning::

        This will not extrapolate outside the ranges of the input data columns.

    """

    # placeholder
    qs = None

    def __init__(self, cache: dict):
        """Initialize with shared cache dict."""
        self.cache = cache

    def fit_transform(self, data: np.array) -> np.array:
        """Fit the layer with a 2D array and (forward) transform it."""
        self.qs = np.sort(data, axis=0)
        if not 'q_norm' in self.cache:
            self.cache['q_norm'] = compute_normal_quantiles(len(data))
        # this can be made a little more efficient using np.argsort
        return self.transform(data)

    def transform(self, data: np.array) -> np.array:
        """Forward transform a 2D array."""
        for column in range(data.shape[1]):
            data[:, column] = np.interp(
                data[:, column], self.qs[:, column], self.cache['q_norm'])
        return data

    def transform_back(self, data: np.array) -> np.array:
        """Backward transform a 2D array."""
        for column in range(data.shape[1]):
            data[:, column] = np.interp(
                data[:, column], self.cache['q_norm'], self.qs[:, column])
        return data

class QuantilesGaussianMarginal(FullGaussianMarginal):
    """Quantiles transformation of the marginals to normals."""

    # _quantiles = np.linspace(0., 1., 101)[1:-1]
    # _qnorm_extrapolate = 10

    def __init__(
            self, n_quantiles: int = 100, extrapolate: float = 100., **kwargs):
        self._n_quantiles = n_quantiles
        if self._n_quantiles == -1:
            self._quantiles = None
        else:
            self._quantiles = np.linspace(0., 1., self._n_quantiles + 1)[1:-1]
        self._qnorm_extrapolate = extrapolate
        super().__init__(**kwargs)

    def fit_transform(self, data: np.array) -> np.array:
        """Fit the layer with a 2D array and (forward) transform it."""

        if self._quantiles is None:
            self.qs = np.sort(data, axis=0)
        else:
            self.qs = np.quantile(data, self._quantiles, axis=0)

        breakpoint()

        if not 'q_norm_raw' in self.cache:
            _raw_qnorm = compute_normal_quantiles(len(data))
            if self._quantiles is None:
                self.cache['q_norm_orig'] = _raw_qnorm
            else:
                self.cache['q_norm_orig'] = np.quantile(
                    _raw_qnorm, self._quantiles)
            self.cache['q_norm'] = np.copy(self.cache['q_norm_orig'])

            # extrapolation
            self.cache['q_norm'][0] = min(
                -self._qnorm_extrapolate, self.cache['q_norm'][0])
            self.cache['q_norm'][-1] = max(
                self._qnorm_extrapolate, self.cache['q_norm'][-1])

        q_norm_orig = self.cache['q_norm_orig']
        q_norm = self.cache['q_norm']

        # extrapolation
        slope_left = (self.qs[1] - self.qs[0]) / (
            q_norm_orig[1] - q_norm_orig[0])
        slope_right = (self.qs[-1] - self.qs[-2]) / (
            q_norm_orig[-1] - q_norm_orig[-2])

        self.qs[0, :] = self.qs[1] - slope_left * (q_norm[1] - q_norm[0])
        self.qs[-1, :] = self.qs[-2] + slope_right * (q_norm[-1] - q_norm[-2])
        
        return self.transform(data)



class RobustFullGaussianMarginal(Layer):
    """Full transformation of the marginals to normals, robust to duplicates.

    .. warning::

        This will not extrapolate outside the ranges of the input data columns.

    """

    # placeholders
    qs = ()
    q_norms = ()

    def fit_transform(self, data: np.array) -> np.array:
        """Fit the layer with a 2D array and (forward) transform it."""
        
        self.qs = []
        self.q_norms = []

        for column in range(data.shape[1]):
            ecdf = sp.stats.ecdf(data[:, column]).cdf
            q, p = ecdf.quantiles, ecdf.probabilities
            p1 = np.concatenate([[0], p])
            p2 = (p1[1:] + p1[:-1]) / 2
            q_norm = sp.stats.norm.ppf(p2)
            self.qs.append(q)
            self.q_norms.append(q_norm)

        return self.transform(data)

    def transform(self, data: np.array) -> np.array:
        """Forward transform a 2D array."""
        for column in range(data.shape[1]):
            data[:, column] = np.interp(
                data[:, column], self.qs[column], self.q_norms[column])
        return data

    def transform_back(self, data: np.array) -> np.array:
        """Backward transform a 2D array."""
        for column in range(data.shape[1]):
            data[:, column] = np.interp(
                data[:, column], self.q_norms[column], self.qs[column])
        return data

class GIIDE(Layer):
    """GIIDE model.
    
    :param n_layers: Number of stacked Gaussian marginalizers and random
        rotation layers.
    :param robust: Use Gaussian marginals computation robust to duplicate
        values, but less efficient. **This option will probably be removed.**
        Default false.
    :param n_quantiles: Number of quantiles used for piece-wise linear
        transformation. Pass -1 to use full Gaussian marginalization.
        Default 10. 
    :param extrapolate: Extend the piece-wise linear transformation linearly
        around the edges up to this Gaussian tail value. If smaller than the
        observed value (e.g., 0) extrapolation is disabled. Default 100.
    :param n_householders: Number of Householder reflections for each rotation.
        If -1, use instead a dense rotation matrix (*i.e.*, as many HRs as the
        size of the matrix). Default 5.
    """

    def __init__(
            self, n_layers: int, robust: bool = False, n_quantiles: int = 10,
            extrapolate: float = 100., svd=False, n_householders: int = 1):
        self._n_layers = n_layers
        self._robust = robust
        self._n_quantiles = n_quantiles
        self._extrapolate = extrapolate
        self._q_norm = None
        self._layers = []
        self._n_householders = n_householders
        self.cache = {}
        self._svd = svd

    def fit_transform(self, data: np.array) -> np.array:
        """Fit the model with a 2D array and (forward) transform it."""

        data = np.copy(data)

        for _ in tqdm(range(self._n_layers)):

            if self._robust:
                layer = RobustFullGaussianMarginal()
            else:
                # if self._n_quantiles == -1:
                #     # logger.warning("Extrapolation disabled for full GM.")
                #     layer = FullGaussianMarginal(cache=self.cache)
                # else:
                layer = QuantilesGaussianMarginal(
                    cache=self.cache, n_quantiles=self._n_quantiles,
                    extrapolate=self._extrapolate)
            data = layer.fit_transform(data)
            self._layers.append(layer)

            if self._svd:
                layer = SVD()
                data = layer.fit_transform(data)
                self._layers.append(layer)

            for _ in range(self._n_householders):
                layer = RandomHouseholder()
                data = layer.fit_transform(data)
                self._layers.append(layer)
            
            if self._n_householders == -1:
                layer = RandomRotation()
                data = layer.fit_transform(data)
                self._layers.append(layer)

        return data

    def transform(self, data: np.array) -> np.array:
        """Forward transform a 2D array."""
        data = np.copy(data)
        for layer in tqdm(self._layers):
            data = layer.transform(data)
        return data

    def transform_back(self, data: np.array) -> np.array:
        """Backward transform a 2D array."""
        data = np.copy(data)
        for layer in tqdm(self._layers[::-1]):
            data = layer.transform_back(data)
        return data