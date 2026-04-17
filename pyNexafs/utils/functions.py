"""
A general collection of utility functions for peak fitting.
"""
import numpy as np

def gaussian(x, A, mu, sigma):
    """Gaussian function."""
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma**2))

def lorentzian(x, A, mu, gamma):
    """Lorentzian function."""
    return A * (gamma**2 / ((x - mu) ** 2 + gamma**2))

def pseudo_voigt(x, A, mu, sigma, eta):
    """Pseudo-Voigt function."""
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