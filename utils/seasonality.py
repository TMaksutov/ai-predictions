from typing import List

import numpy as np


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(max_lag + 1)
    x = x - np.nanmean(x)
    n = x.size
    # Compute autocovariances via FFT for speed
    fft_len = 1
    while fft_len < 2 * n:
        fft_len <<= 1
    fx = np.fft.rfft(x, n=fft_len)
    acov = np.fft.irfft(fx * np.conj(fx))[: n]
    acov = acov / max(acov[0], 1e-12)
    acf_vals = acov[: max_lag + 1]
    return np.real(acf_vals)


def _periodogram_top_periods(x: np.ndarray, max_period: int, top_n: int) -> List[int]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return []
    x = x - np.nanmean(x)
    # Power spectrum
    fx = np.fft.rfft(x)
    power = np.abs(fx) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    # Ignore zero frequency (trend)
    if power.size <= 1:
        return []
    power[0] = 0.0
    # Convert to periods (in samples = days)
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = np.where(freqs > 0, (1.0 / freqs), np.inf)
    candidates = []
    for i in range(1, len(freqs)):
        p = periods[i]
        if not np.isfinite(p):
            continue
        if p <= 1 or p > max_period:
            continue
        candidates.append((power[i], int(round(p))))
    if not candidates:
        return []
    # Aggregate duplicate integer periods by max power
    from collections import defaultdict

    max_power_by_p = defaultdict(float)
    for pw, p in candidates:
        if pw > max_power_by_p[p]:
            max_power_by_p[p] = pw
    ranked = sorted(max_power_by_p.items(), key=lambda kv: kv[1], reverse=True)
    return [p for p, _ in ranked[: top_n]]


def detect_seasonal_periods(y: np.ndarray, max_period: int = 400, top_n: int = 3, min_cycles: int = 3) -> List[int]:
    """
    Return up to top_n candidate integer-day periods detected in y using FFT periodogram
    cross-validated by ACF peaks. Requires at least min_cycles cycles in the series length.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 16:
        return []

    # Initial candidates from periodogram
    period_candidates = _periodogram_top_periods(y, max_period=max_period, top_n=top_n * 3)
    if not period_candidates:
        return []

    # Validate with ACF peaks
    acf_vals = _acf(y, max_lag=min(max_period, max(1, int(n // 2))))
    selected: List[int] = []
    for p in period_candidates:
        if p <= 1 or p > max_period:
            continue
        # Need enough cycles
        if n < p * max(1, int(min_cycles)):
            continue
        # ACF support: local prominence at lag p
        if p < len(acf_vals) - 1:
            left = acf_vals[p - 1] if p - 1 >= 0 else 0.0
            mid = acf_vals[p]
            right = acf_vals[p + 1]
            if mid > left and mid > right and mid > 0.1:
                selected.append(int(p))
        if len(selected) >= top_n:
            break

    # Deduplicate near-harmonics (e.g., prefer 365 over 182 if both appear)
    final_periods: List[int] = []
    for p in selected:
        if any(abs(p - q) <= 2 or (max(p, q) % min(p, q) == 0) for q in final_periods if q > 0):
            continue
        final_periods.append(p)

    return final_periods[: top_n]


