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

from unittest import TestCase

import numpy as np
import pandas as pd

from giide.giide import (
    RandomRotation, FullGaussianMarginal, RobustFullGaussianMarginal,
    compute_normal_quantiles, GIIDE, RandomHouseholder, SVD,
    QuantilesGaussianMarginal)

class TestGiide(TestCase):
    """Unit tests for GIIDE."""

    def _test_layer( # pylint: disable=dangerous-default-value
            self, layer_class: type, ntrainobs: int = 100, ntestobs: int = 100,
            init_kwargs: dict ={}):
        """Base test case for layer."""
        layer = layer_class(**init_kwargs)
        if ntrainobs % 2 != 0:
            raise SyntaxError("Pass even ntrainobs.")
        data = np.random.randn(ntrainobs//2, 3)

        # we add duplicates
        data = np.concat([data,data], axis=0)

        data_transf = layer.fit_transform(np.copy(data))
        self.assertEqual(data_transf.shape, data.shape)
        data_transf_back = layer.transform_back(np.copy(data_transf))
        self.assertTrue(np.allclose(data, data_transf_back))

        data2 = np.random.randn(ntestobs, 3)

        # checking .99 quantile of abs diffs if very small, for layers that
        # don't do extrapolation
        self.assertLess(
          np.quantile(
            np.abs(
              data2 - layer.transform_back(layer.transform(np.copy(data2)))
            ), .95), 1e-12)


    def test_layer_random_rotation(self):
        """Test random rotation layer."""
        self._test_layer(RandomRotation)

    def test_layer_SVD_rotation(self):
        """Test SVD layer."""
        self._test_layer(SVD)

    def test_layer_random_householder(self):
        """Test random Householder reflection layer."""
        self._test_layer(RandomHouseholder)

    def test_layer_full_gaussian_marginals(self):
        """Test FGM layer."""

        self._test_layer(
            FullGaussianMarginal, ntrainobs=10000, ntestobs=100,
            init_kwargs={'cache': {}})

    def test_layer_quantiles_gaussian_marginals(self):
        """Test QGM layer."""

        self._test_layer(
            QuantilesGaussianMarginal, init_kwargs={
                'n_quantiles': 10, 'cache': {}})


    def test_layer_robust_full_gaussian_marginals(self):
        """Test RFGM layer."""

        self._test_layer(RobustFullGaussianMarginal, ntrainobs=10000,
            ntestobs=100)

    def test_simple_giide(self):
        self._test_layer(
            GIIDE, ntrainobs=10000, ntestobs=100, init_kwargs={'n_layers': 3})