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

import numpy as np
import pandas as pd

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
    """Layer implementing a random rotation."""

    # placeholder
    _rotation = np.array([[]])

    # TODO: implement some way to seed random draw

    @staticmethod
    def random_rotation(size):
        a = np.random.randn(size,size),
        q,_ = np.linalg.qr(a)
        return q[0]
    
    def fit_transform(self, data: np.array) -> np.array:
        """Fit the layer with a 2D array and (forward) transform it."""
        self._rotation = self.random_rotation(size=data.shape[1])
        return self.transform(data=data)

    def transform(self, data: np.array) -> np.array:
        """Forward transform a 2D array."""
        return data @ self._rotation

    def transform_back(self, data: np.array) -> np.array:
        """Backward transform a data frame."""
        _ = data @ self._rotation.T
        return _
