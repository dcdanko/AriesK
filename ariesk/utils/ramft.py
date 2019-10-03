# cython: profile=True

from math import gcd
import numpy as np


def memoize(func):
    """Memoize the input function."""
    tbl = {}

    def helper(args):
        if args not in tbl:
            tbl[args] = func(args)
        return tbl[args]
    return helper


@memoize
def phi(n):
    """Return the Euler's totient of n."""
    amount = 0
    for k in range(1, n + 1):
        if gcd(n, k) == 1:
            amount += 1
    return amount


def ram_sum(n, q):
    """Return the ramanujan sum of (n, q)."""
    c_q = [
        np.exp(2 * 1j * np.pi * n * (p / q))
        for p in range(1, q + 1) if gcd(p, q) == 1
    ]
    c_q = sum(c_q)
    return np.real(c_q)


def build_rs_matrix(N):
    """Return the ram sum matrix with normalization."""
    def inner(q, j):
        """Return the normalized ram sum for the coordinates."""
        return (1 / (phi(q) * N)) * ram_sum(1 + (j - 1) % q, q)

    return np.array([
        [inner(q, j) for j in range(1, N + 1)] for q in range(1, N + 1)
    ])
