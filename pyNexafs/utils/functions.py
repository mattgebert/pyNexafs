"""
A general collection of utility functions for peak fitting.
"""

import numpy as np


def gaussian(x, A, mu, sigma):
    """
    Gaussian function.

    Parameters
    ----------
    x : array-like
        The independent variable.
    A : float
        The amplitude of the Gaussian.
    mu : float
        The mean of the Gaussian.
    sigma : float
        The standard deviation of the Gaussian.

    Returns
    -------
    array-like
        The Gaussian function evaluated at x.
    """
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def lorentzian(x, A, mu, gamma):
    """
    Lorentzian function.

    Parameters
    ----------
    x : array-like
        The independent variable.
    A : float
        The amplitude of the Lorentzian.
    mu : float
        The mean of the Lorentzian.
    gamma : float
        The half-width at half-maximum of the Lorentzian.

    Returns
    -------
    array-like
        The Lorentzian function evaluated at x.
    """
    return A * (gamma**2 / ((x - mu) ** 2 + gamma**2))


def pseudo_voigt(x, A, mu, sigma, eta):
    """
    Pseudo-Voigt function.

    Parameters
    ----------
    x : array-like
        The independent variable.
    A : float
        The amplitude of the Pseudo-Voigt.
    mu : float
        The mean of the Pseudo-Voigt.
    sigma : float
        The standard deviation of the Gaussian component.
    eta : float
        The mixing parameter between Gaussian and Lorentzian components.

    Returns
    -------
    array-like
        The Pseudo-Voigt function evaluated at x.
    """
    return eta * lorentzian(x, A, mu, sigma) + (1 - eta) * gaussian(x, A, mu, sigma)


if __name__ == "__main__":
    # Plot the functions for testing
    import matplotlib.pyplot as plt

    x = np.linspace(-10, 10, 1000)
    A, mu, sigma, eta = 1, 0, 1, 0.5
    plt.plot(x, gaussian(x, A, mu, sigma), label="Gaussian")
    plt.plot(x, lorentzian(x, A, mu, sigma), label="Lorentzian")
    plt.plot(x, pseudo_voigt(x, A, mu, sigma, eta), label="Pseudo-Voigt")
    plt.legend()
    plt.show()
