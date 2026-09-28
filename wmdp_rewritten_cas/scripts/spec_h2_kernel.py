#!/usr/bin/env python3
"""Vectorised information-theoretic primitives for Spec H2.

Pure numpy. No I/O, no globals, no model imports. Every quantity is computed from
a 4x4x4 joint count table over (F, Y, L); the batched forms build B such tables in
one bincount so that bootstrap and permutation workloads stay in numpy.

Estimator identity with Spec H is a contract: the plugin entropy and the
Miller-Madow correction (occupied_cells - 1) / (2 n ln 2) are the same quantities
computed by ``spec_h_common.entropy``. ``tests/test_spec_h2.py`` asserts this to
1e-12 on random inputs. Spec H2 Amendment 1 Ruling 6 forbids the Counter-over-
tuples path of ``spec_h_common`` inside any inner loop, so it is used only as the
test oracle.
"""
from __future__ import annotations

import numpy as np

LETTERS = "ABCD"
LN2 = float(np.log(2.0))
RATIO_EPSILON = 1e-12


def argmax_label(probabilities: np.ndarray) -> np.ndarray:
    """Four-way argmax with ties broken toward the lowest letter index.

    ``np.argmax`` already returns the first maximal index, which is the rule used
    by ``spec_h_full.label`` (``key=(p, -i)``). Amendment 1 Ruling 6 makes the tie
    rule normative, so it is stated explicitly here rather than inherited.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError("probabilities must have shape (n, 4)")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    return probabilities.argmax(axis=1).astype(np.int8)


def count_argmax_ties(probabilities: np.ndarray) -> int:
    """Rows whose maximum is attained by more than one letter."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    maxima = probabilities.max(axis=1, keepdims=True)
    return int(((probabilities == maxima).sum(axis=1) > 1).sum())


def _as_codes(f: np.ndarray, y: np.ndarray, l: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    l = np.asarray(l, dtype=np.int64)
    if not (f.shape == y.shape == l.shape) or f.ndim != 1 or f.size == 0:
        raise ValueError("f, y, l must be non-empty aligned 1-D arrays")
    for name, arr in (("f", f), ("y", y), ("l", l)):
        if arr.min() < 0 or arr.max() > 3:
            raise ValueError(f"{name} must contain four-way labels in [0, 3]")
    return f * 16 + y * 4 + l


def joint3(f: np.ndarray, y: np.ndarray, l: np.ndarray) -> np.ndarray:
    """(4, 4, 4) joint counts indexed [f, y, l]."""
    codes = _as_codes(f, y, l)
    return np.bincount(codes, minlength=64).reshape(4, 4, 4)


def joint3_batched(f: np.ndarray, y: np.ndarray, l: np.ndarray,
                   index: np.ndarray) -> np.ndarray:
    """(B, 4, 4, 4) joint counts for B item-index selections in one bincount.

    ``index`` has shape (B, m); each row selects m items (with replacement for a
    bootstrap, as a permutation for a null).
    """
    codes = _as_codes(f, y, l)
    index = np.asarray(index, dtype=np.int64)
    if index.ndim != 2:
        raise ValueError("index must have shape (B, m)")
    batch = index.shape[0]
    selected = codes[index] + 64 * np.arange(batch, dtype=np.int64)[:, None]
    return np.bincount(selected.ravel(), minlength=64 * batch).reshape(batch, 4, 4, 4)


def joint3_batched_l(f: np.ndarray, y: np.ndarray, l_batch: np.ndarray) -> np.ndarray:
    """(B, 4, 4, 4) joint counts for B alternative L vectors over fixed (F, Y).

    Used by the mu permutation null (L shuffled item-wise) and by the
    accuracy-matched null-model draws.
    """
    f = np.asarray(f, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    l_batch = np.asarray(l_batch, dtype=np.int64)
    if l_batch.ndim != 2 or l_batch.shape[1] != f.size:
        raise ValueError("l_batch must have shape (B, n)")
    batch = l_batch.shape[0]
    codes = (f * 16 + y * 4)[None, :] + l_batch
    selected = codes + 64 * np.arange(batch, dtype=np.int64)[:, None]
    return np.bincount(selected.ravel(), minlength=64 * batch).reshape(batch, 4, 4, 4)


def _entropy(counts: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Plugin entropy in bits and the Miller-Madow correction, over a batch.

    ``counts`` has shape (B, ...); all trailing axes are the alphabet.
    """
    flat = counts.reshape(counts.shape[0], -1).astype(np.float64)
    p = flat / float(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0.0, p * np.log2(np.where(p > 0.0, p, 1.0)), 0.0)
    plugin = -terms.sum(axis=1)
    occupied = (flat > 0.0).sum(axis=1)
    correction = (occupied - 1.0) / (2.0 * float(n) * LN2)
    return plugin, correction


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray,
                 epsilon: float = RATIO_EPSILON) -> np.ndarray:
    """Element-wise ratio; NaN where the denominator is within epsilon of zero.

    NaN is the in-array encoding of "undefined"; it is converted to a JSON null at
    the serialisation boundary. Negative values are never clipped.
    """
    out = np.full(numerator.shape, np.nan, dtype=np.float64)
    usable = np.abs(denominator) > epsilon
    out[usable] = numerator[usable] / denominator[usable]
    return out


def stats_from_joint(counts: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """All Spec H2 scalars from a batch of (4, 4, 4) joints indexed [f, y, l].

    Returns arrays of shape (B,). Keys carry an explicit ``_plugin`` or ``_mm``
    suffix; both estimators are always computed and stored (spec section 0).

    ``ratio_min`` and ``ceiling`` are the Amendment 1 Ruling 3 exploratory
    quantities: the original ratio is structurally bounded by
    min(1, I(L;Y) / I(F;Y)) because mu <= min(I(F;Y), I(L;Y)) (Nakkiran Def. 1).
    """
    counts = np.asarray(counts)
    if counts.ndim == 3:
        counts = counts[None, ...]
    if counts.shape[1:] != (4, 4, 4):
        raise ValueError("counts must have shape (B, 4, 4, 4) or (4, 4, 4)")

    h_f, c_f = _entropy(counts.sum(axis=(2, 3)), n)
    h_y, c_y = _entropy(counts.sum(axis=(1, 3)), n)
    h_l, c_l = _entropy(counts.sum(axis=(1, 2)), n)
    h_fy, c_fy = _entropy(counts.sum(axis=3), n)
    h_fl, c_fl = _entropy(counts.sum(axis=2), n)
    h_yl, c_yl = _entropy(counts.sum(axis=1), n)
    h_fyl, c_fyl = _entropy(counts, n)

    i_fy_plugin = h_f + h_y - h_fy
    i_fy_mm = i_fy_plugin + (c_f + c_y - c_fy)
    i_ly_plugin = h_l + h_y - h_yl
    i_ly_mm = i_ly_plugin + (c_l + c_y - c_yl)
    cmi_plugin = h_fl + h_yl - h_l - h_fyl
    cmi_mm = cmi_plugin + (c_fl + c_yl - c_l - c_fyl)

    out: dict[str, np.ndarray] = {}
    for suffix, i_fy, i_ly, cmi in (
        ("plugin", i_fy_plugin, i_ly_plugin, cmi_plugin),
        ("mm", i_fy_mm, i_ly_mm, cmi_mm),
    ):
        mu = i_fy - cmi
        out[f"i_fy_{suffix}"] = i_fy
        out[f"i_ly_{suffix}"] = i_ly
        out[f"cmi_{suffix}"] = cmi
        out[f"mu_{suffix}"] = mu
        out[f"ratio_{suffix}"] = _safe_divide(mu, i_fy)
        out[f"ratio_min_{suffix}"] = _safe_divide(mu, np.minimum(i_fy, i_ly))
        out[f"ceiling_{suffix}"] = np.minimum(1.0, _safe_divide(i_ly, i_fy))
    out["mm_correction_i_fy"] = c_f + c_y - c_fy
    out["mm_correction_cmi"] = c_fl + c_yl - c_l - c_fyl
    return out


def mi2_batched(f: np.ndarray, y_batch: np.ndarray) -> dict[str, np.ndarray]:
    """I(F;Y) over B alternative Y vectors; the 4x4 path used by the I_t null."""
    f = np.asarray(f, dtype=np.int64)
    y_batch = np.asarray(y_batch, dtype=np.int64)
    if y_batch.ndim != 2 or y_batch.shape[1] != f.size:
        raise ValueError("y_batch must have shape (B, n)")
    batch, n = y_batch.shape
    codes = f[None, :] * 4 + y_batch + 16 * np.arange(batch, dtype=np.int64)[:, None]
    joint = np.bincount(codes.ravel(), minlength=16 * batch).reshape(batch, 4, 4)
    h_f, c_f = _entropy(joint.sum(axis=2), n)
    h_y, c_y = _entropy(joint.sum(axis=1), n)
    h_fy, c_fy = _entropy(joint, n)
    plugin = h_f + h_y - h_fy
    return {"i_fy_plugin": plugin, "i_fy_mm": plugin + (c_f + c_y - c_fy)}


def symmetric_null_draws(y: np.ndarray, p_correct: float, draws: int,
                         rng: np.random.Generator) -> np.ndarray:
    """(draws, n) accuracy-matched random references (Nakkiran footnote 5).

    Each entry equals Y with probability ``p_correct``, else is uniform over the
    other three letters. ``(y + 1 + k) % 4`` for k uniform on {0, 1, 2} enumerates
    exactly the three labels other than y, so the channel is symmetric.
    """
    if not 0.25 <= p_correct <= 1.0:
        raise ValueError("four-way matched-null p must lie in [0.25, 1]")
    y = np.asarray(y, dtype=np.int64)
    uniform = rng.random((draws, y.size))
    alternative = rng.integers(0, 3, size=(draws, y.size))
    return np.where(uniform < p_correct, y[None, :],
                    (y[None, :] + 1 + alternative) % 4).astype(np.int64)


def permutation_batch(n: int, batch: int, rng: np.random.Generator) -> np.ndarray:
    """(batch, n) independent permutations of range(n)."""
    return np.argsort(rng.random((batch, n)), axis=1)


def bootstrap_index(n: int, batch: int, rng: np.random.Generator) -> np.ndarray:
    """(batch, n) question-level resample indices, drawn with replacement."""
    return rng.integers(0, n, size=(batch, n))


def percentile_ci(values: np.ndarray, low: float = 2.5,
                  high: float = 97.5) -> tuple[float | None, float | None, int]:
    """Percentile interval over the finite entries; never clipped at zero."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None, 0
    return (float(np.percentile(finite, low)),
            float(np.percentile(finite, high)),
            int(finite.size))
