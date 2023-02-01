"""Estimate pi using various methods."""
import abc

import numpy as np


class PiEstimator(abc.ABC):
    """Abstract base class for pi estimators."""

    @abc.abstractmethod
    def estimate(self):
        """Estimate pi."""


class MonteCarloPiEstimator(PiEstimator):
    """Estimate pi using the Monte Carlo method."""

    def __init__(self, n):
        """Initialize the Monte Carlo estimator.

        Parameters
        ----------
        n: int
            Number of random points to use.
        """
        self.n = n

    def estimate(self):
        """Estimate pi using the Monte Carlo method."""
        data = np.random.uniform(-0.5, 0.5, size=(self.n, 2))
        inside = len(np.argwhere(np.linalg.norm(data, axis=1) < 0.5))
        return 4 * inside / self.n


class BaileyBorweinPlouffeEstimator(PiEstimator):
    """Estimate pi using the Bailey-Borwein-Plouffe formula."""

    def __init__(self, n):
        """Initialize the Bailey-Borwein-Plouffe formula estimator.

        Parameters
        ----------
        n: int
            Number of points to use.
        """
        self.n = n

    def estimate(self):
        """Estimate pi using the Bailey-Borwein-Plouffe formula."""
        data = np.arange(self.n)
        pre_factor = 1 / (16**data)
        pi = (
            4 / (8 * data + 1)
            - 2 / (8 * data + 4)
            - 1 / (8 * data + 5)
            - 1 / (8 * data + 6)
        )
        pi = pre_factor * pi
        return float(np.sum(pi))


class LeibnizPiEstimator(PiEstimator):
    """Estimate pi using the Leibniz formula."""

    def __init__(self, n):
        """Initialize the Leibniz formula estimator.

        Parameters
        ----------
        n: int
            Number of points to use.
        """
        self.n = n

    def estimate(self):
        """Estimate pi using the Leibniz formula."""
        data = np.arange(self.n)
        pi = 4 * (-1) ** data / (2 * data + 1)
        return float(np.sum(pi))
