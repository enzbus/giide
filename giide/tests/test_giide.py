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

from giide.giide import RandomRotation

class TestGiide(TestCase):
    """Unit tests for GIIDE."""

    def _test_layer(self, layer_class: type):
        """Base test case for layer."""
        layer = layer_class()
        data = np.random.randn(100, 3)

        data_transf = layer.fit_transform(data)
        self.assertEqual(data_transf.shape, data.shape)
        data_transf_back = layer.transform_back(data_transf)
        self.assertTrue(np.allclose(data, data_transf_back))

        data2 = np.random.randn(100, 3)

        self.assertTrue(
            np.allclose(data2, layer.transform_back(layer.transform(data2))))

    def test_layer_random_rotation(self):
        """Test random rotation layer."""
        self._test_layer(RandomRotation)