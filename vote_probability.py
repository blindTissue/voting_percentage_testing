from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp


def alpha_from_counts(counts, kappa: float) -> np.ndarray:
    """Convert 3-way calibration counts into alpha = kappa * direction."""
    counts_array = np.asarray(counts, dtype=float)
    if counts_array.shape != (3,):
        raise ValueError("counts must contain exactly three categories")
    if np.any(counts_array < 0):
        raise ValueError("counts must be non-negative")
    total = counts_array.sum()
    if total <= 0:
        raise ValueError("at least one calibration count must be positive")
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    return kappa * counts_array / total


def alpha_from_mean(u1: float, u2: float, kappa: float) -> np.ndarray:
    """Convert direct mean shares into alpha = kappa * (u1, u2, 1-u1-u2)."""
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if u1 <= 0 or u2 <= 0:
        raise ValueError("u1 and u2 must be positive")
    other = 1.0 - u1 - u2
    if other <= 1e-12:
        raise ValueError("u1 + u2 must be less than 1")
    return kappa * np.array([u1, u2, other], dtype=float)


def _validate_probability_vector(p) -> np.ndarray:
    p_array = np.asarray(p, dtype=float)
    if p_array.shape != (3,):
        raise ValueError("probability vector must contain exactly three categories")
    if np.any(p_array <= 0):
        raise ValueError("all probabilities must be positive")
    total = p_array.sum()
    if not np.isclose(total, 1.0):
        p_array = p_array / total
    return p_array


def _validate_alpha(alpha) -> np.ndarray:
    alpha_array = np.asarray(alpha, dtype=float)
    if alpha_array.shape != (3,):
        raise ValueError("alpha must contain exactly three categories")
    if np.any(alpha_array <= 0):
        raise ValueError("all alpha values must be positive")
    return alpha_array


def _validate_weights(weights, expected_length: int) -> np.ndarray:
    weights_array = np.asarray(weights, dtype=float)
    if weights_array.shape != (expected_length,):
        raise ValueError("weights length must match the number of components")
    if np.any(weights_array < 0):
        raise ValueError("weights must be non-negative")
    total = weights_array.sum()
    if total <= 0:
        raise ValueError("at least one mixture weight must be positive")
    return weights_array / total


def _log_comb(n: int, k) -> np.ndarray:
    k_array = np.asarray(k)
    return gammaln(n + 1) - gammaln(k_array + 1) - gammaln(n - k_array + 1)


def _binom_logpmf(k, n: int, p: float) -> np.ndarray:
    k_array = np.asarray(k)
    return _log_comb(n, k_array) + k_array * np.log(p) + (n - k_array) * np.log1p(-p)


def _betabinom_logpmf(k, n: int, a: float, b: float) -> np.ndarray:
    k_array = np.asarray(k)
    return (
        _log_comb(n, k_array)
        + gammaln(k_array + a)
        + gammaln(n - k_array + b)
        - gammaln(n + a + b)
        + gammaln(a + b)
        - gammaln(a)
        - gammaln(b)
    )


def collision_probability_multinomial(n_a: int, n_b: int, p_a, p_b) -> float:
    """Probability two multinomial districts match on categories 1 and 2."""
    p_a = _validate_probability_vector(p_a)
    p_b = _validate_probability_vector(p_b)
    x_max = min(n_a, n_b)
    p_a_y_given_not_x = p_a[1] / (1.0 - p_a[0])
    p_b_y_given_not_x = p_b[1] / (1.0 - p_b[0])

    log_terms = []
    for x in range(x_max + 1):
        y = np.arange(min(n_a - x, n_b - x) + 1)
        log_x = _binom_logpmf(x, n_a, p_a[0]) + _binom_logpmf(x, n_b, p_b[0])
        log_y = _binom_logpmf(y, n_a - x, p_a_y_given_not_x) + _binom_logpmf(
            y, n_b - x, p_b_y_given_not_x
        )
        log_terms.append(log_x + logsumexp(log_y))

    return float(np.exp(logsumexp(log_terms)))


def collision_probability_dirichlet_multinomial(n_a: int, n_b: int, alpha_a, alpha_b) -> float:
    """Probability two Dirichlet-Multinomial districts match on categories 1 and 2."""
    alpha_a = _validate_alpha(alpha_a)
    alpha_b = _validate_alpha(alpha_b)
    x_max = min(n_a, n_b)

    log_terms = []
    for x in range(x_max + 1):
        y = np.arange(min(n_a - x, n_b - x) + 1)
        log_x = _betabinom_logpmf(x, n_a, alpha_a[0], alpha_a[1] + alpha_a[2]) + _betabinom_logpmf(
            x, n_b, alpha_b[0], alpha_b[1] + alpha_b[2]
        )
        log_y = _betabinom_logpmf(y, n_a - x, alpha_a[1], alpha_a[2]) + _betabinom_logpmf(
            y, n_b - x, alpha_b[1], alpha_b[2]
        )
        log_terms.append(log_x + logsumexp(log_y))

    return float(np.exp(logsumexp(log_terms)))


def collision_probability_dirichlet_mixture(
    n_a: int,
    n_b: int,
    weights_a,
    alphas_a,
    weights_b,
    alphas_b,
) -> float:
    """Probability two mixture-of-Dirichlet-Multinomial districts match on categories 1 and 2."""
    alphas_a = [_validate_alpha(alpha) for alpha in alphas_a]
    alphas_b = [_validate_alpha(alpha) for alpha in alphas_b]
    weights_a = _validate_weights(weights_a, len(alphas_a))
    weights_b = _validate_weights(weights_b, len(alphas_b))

    total = 0.0
    for weight_a, alpha_a in zip(weights_a, alphas_a):
        for weight_b, alpha_b in zip(weights_b, alphas_b):
            total += weight_a * weight_b * collision_probability_dirichlet_multinomial(
                n_a, n_b, alpha_a, alpha_b
            )
    return float(total)
