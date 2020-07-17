r"""
'MatrixOracle' is the abstract base class for all matrix oracles. A
matric oracle provides a means of observing an underlying matrix by
means of an 'observe_entries' method, as well as a 'shape' method
that gives the dimension of the matrix.

Subclasses only have to implement 'shape' and the helper method '_observe_entries'.
"""
import numpy as np
from typing import List, Tuple

from abc import ABC, abstractmethod


class MatrixOracle(ABC):
    @abstractmethod
    def shape(self) -> Tuple[int, int]:  # pragma: no cover
        r"""
        Returns the shape of the underlying matrix.
        """
        raise NotImplementedError

    def observe_entries(
        self,
        matrix_indices: List[Tuple[int, int]]
    ) -> np.array:
        rows, cols = zip(*matrix_indices)
        vals = self._observe_entries(rows, cols)
        if 'X_observed' not in self.__dict__:  # TODO: This is duplicated code
            self.X_observed = np.zeros(shape=self.shape()) + np.nan
        self.X_observed[(rows, cols)] = vals
        return vals

    @abstractmethod
    def _observe_entries(
        self,
        rows: List[int],
        cols: List[int]
    ) -> np.array:   # pragma: no cover
        r"""
        Returns the one-dimensional np.array of values found at the requested entries
        (as specified by 'rows' and 'cols').
        """
        raise NotImplementedError

    def observed_matrix(self) -> np.array:
        r"""
        Returns the currently observed matrix.
        """
        if 'X_observed' not in self.__dict__:  # TODO: This is duplicated code
            self.X_observed = np.zeros(shape=self.shape()) + np.nan
        return self.X_observed


class OracleWithAPrescribedMatrix(MatrixOracle):
    def __init__(self, X: np.array):
        self.X = X

    def shape(self) -> Tuple[int, int]:
        return self.X.shape

    def _observe_entries(
        self,
        rows: List[int],
        cols: List[int]
    ) -> np.array:
        return self.X[(rows, cols)]


class CompassOracle(MatrixOracle):
    def __init__(self):
        raise NotImplementedError

    def shape(self) -> Tuple[int, int]:
        raise NotImplementedError

    def _observe_entries(
        self,
        rows: List[int],
        cols: List[int]
    ) -> np.array:
        raise NotImplementedError
