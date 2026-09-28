#!/usr/bin/env python3
"""Pure-CPU information-decomposition primitives for Spec H."""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Hashable, Iterable, Sequence


class SpecHError(ValueError):
    """A scientific contract or input invariant failed."""


def _rows(*columns: Sequence[Hashable]) -> list[tuple[Hashable, ...]]:
    if not columns or len({len(column) for column in columns}) != 1:
        raise SpecHError("entropy columns must be non-empty and aligned")
    return list(zip(*columns))


def entropy(*columns: Sequence[Hashable], miller_madow: bool = False) -> tuple[float, float]:
    rows = _rows(*columns)
    counts = Counter(rows)
    n = len(rows)
    plugin = -sum((count / n) * math.log2(count / n) for count in counts.values())
    correction = (len(counts) - 1) / (2 * n * math.log(2)) if miller_madow else 0.0
    return plugin + correction, correction


def mutual_information(x: Sequence[Hashable], y: Sequence[Hashable],
                       miller_madow: bool = False) -> tuple[float, float]:
    hx, cx = entropy(x, miller_madow=miller_madow)
    hy, cy = entropy(y, miller_madow=miller_madow)
    hxy, cxy = entropy(x, y, miller_madow=miller_madow)
    return hx + hy - hxy, cx + cy - cxy


def conditional_mutual_information(x: Sequence[Hashable], y: Sequence[Hashable],
                                   z: Sequence[Hashable],
                                   miller_madow: bool = False) -> tuple[float, float]:
    hxz, cxz = entropy(x, z, miller_madow=miller_madow)
    hyz, cyz = entropy(y, z, miller_madow=miller_madow)
    hz, cz = entropy(z, miller_madow=miller_madow)
    hxyz, cxyz = entropy(x, y, z, miller_madow=miller_madow)
    return hxz + hyz - hz - hxyz, cxz + cyz - cz - cxyz


def decomposition(f: Sequence[int], reference: Sequence[int], gold: Sequence[int],
                  miller_madow: bool = False, epsilon: float = 1e-12) -> dict[str, float | None]:
    raw, raw_correction = mutual_information(f, gold, miller_madow=miller_madow)
    conditional, conditional_correction = conditional_mutual_information(
        f, gold, reference, miller_madow=miller_madow
    )
    mu = raw - conditional
    return {
        "i_f_y": raw,
        "i_f_y_given_l": conditional,
        "mu": mu,
        "ratio": None if abs(raw) <= epsilon else mu / raw,
        "raw_mm_correction": raw_correction,
        "conditional_mm_correction": conditional_correction,
        "mu_mm_correction": raw_correction - conditional_correction,
    }


def argmax_label(probabilities: Sequence[float]) -> int:
    if len(probabilities) != 4 or any(not math.isfinite(float(x)) or float(x) < 0 for x in probabilities):
        raise SpecHError("each prediction must contain four finite non-negative values")
    total = sum(float(x) for x in probabilities)
    if abs(total - 1.0) > 1e-5:
        raise SpecHError(f"probabilities must sum to one; observed {total}")
    return max(range(4), key=lambda index: (float(probabilities[index]), -index))


def validate_alignment(*tables: Sequence[dict]) -> list[str]:
    if not tables:
        raise SpecHError("at least one prediction table is required")
    id_lists = [[str(row["question_id"]) for row in table] for table in tables]
    if any(len(ids) != len(set(ids)) for ids in id_lists):
        raise SpecHError("duplicate question IDs in a prediction table")
    baseline = id_lists[0]
    for ids in id_lists[1:]:
        if ids != baseline:
            raise SpecHError("question IDs are absent, reordered, or misaligned")
    gold_lists = [[int(row["gold"]) for row in table] for table in tables]
    if any(golds != gold_lists[0] for golds in gold_lists[1:]):
        raise SpecHError("gold labels disagree across aligned tables")
    return baseline


def symmetric_random_predictions(gold: Sequence[int], p_correct: float, rng: random.Random) -> list[int]:
    if not 0.25 <= p_correct <= 1.0:
        raise SpecHError("four-way matched-null p must lie in [0.25, 1]")
    result = []
    for label in gold:
        if rng.random() < p_correct:
            result.append(int(label))
        else:
            alternatives = [value for value in range(4) if value != int(label)]
            result.append(alternatives[rng.randrange(3)])
    return result


def expected_symmetric_null_mi(gold: Sequence[int], p_correct: float) -> float:
    """Exact MI of the four-way symmetric channel under the empirical Y distribution."""
    n = len(gold)
    if not n:
        raise SpecHError("gold labels are empty")
    py = [sum(int(y == label) for y in gold) / n for label in range(4)]
    joint = [[0.0] * 4 for _ in range(4)]
    for y in range(4):
        for prediction in range(4):
            conditional = p_correct if prediction == y else (1 - p_correct) / 3
            joint[prediction][y] = py[y] * conditional
    pf = [sum(joint[f][y] for y in range(4)) for f in range(4)]
    return sum(
        joint[f][y] * math.log2(joint[f][y] / (pf[f] * py[y]))
        for f in range(4) for y in range(4) if joint[f][y] > 0 and pf[f] > 0 and py[y] > 0
    )


def solve_null_p(gold: Sequence[int], target_mi: float, tolerance: float = 5e-4) -> tuple[float, float]:
    maximum = expected_symmetric_null_mi(gold, 1.0)
    if target_mi < -tolerance or target_mi > maximum + tolerance:
        raise SpecHError(f"target MI {target_mi} is outside symmetric-null range [0,{maximum}]")
    low, high = 0.25, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        achieved = expected_symmetric_null_mi(gold, middle)
        if achieved < target_mi:
            low = middle
        else:
            high = middle
    p = (low + high) / 2
    achieved = expected_symmetric_null_mi(gold, p)
    if abs(achieved - target_mi) > tolerance:
        raise SpecHError("matched-null solver failed requested tolerance")
    return p, achieved


def bootstrap_decomposition(f: Sequence[int], reference: Sequence[int], gold: Sequence[int],
                            resamples: int, seed: int, miller_madow: bool) -> dict[str, list[float]]:
    if resamples < 1000:
        raise SpecHError("Spec H requires at least 1000 bootstrap resamples")
    validate_lengths = {len(f), len(reference), len(gold)}
    if len(validate_lengths) != 1 or not gold:
        raise SpecHError("bootstrap arrays must be non-empty and aligned")
    rng = random.Random(seed)
    collected: dict[str, list[float]] = {key: [] for key in ("i_f_y", "i_f_y_given_l", "mu", "ratio")}
    n = len(gold)
    for _ in range(resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        value = decomposition([f[i] for i in indices], [reference[i] for i in indices],
                              [gold[i] for i in indices], miller_madow=miller_madow)
        for key in collected:
            if value[key] is not None and math.isfinite(float(value[key])):
                collected[key].append(float(value[key]))
    result = {}
    for key, values in collected.items():
        values.sort()
        if not values:
            result[key] = []
            continue
        lo = values[max(0, int(0.025 * len(values)) - 1)]
        hi = values[min(len(values) - 1, int(0.975 * len(values)))]
        result[key] = [lo, hi]
    return result


def tier_values(f, reference, gold, miller_madow=True):
    raw, raw_c = mutual_information(f, gold, miller_madow=miller_madow)
    l_correct = [int(l == y) for l, y in zip(reference, gold)]
    c2, c2c = conditional_mutual_information(f, gold, l_correct, miller_madow=miller_madow)
    c3, c3c = conditional_mutual_information(f, gold, reference, miller_madow=miller_madow)
    def formed(cond, correction):
        mu=raw-cond
        return {"i_f_y_given_l":cond,"mu":mu,"ratio":None if abs(raw)<=1e-12 else mu/raw,
                "conditional_mm_correction":correction}
    return {"tier_1":{"i_f_y":raw,"raw_mm_correction":raw_c},
            "tier_2":formed(c2,c2c),"tier_3":formed(c3,c3c)}
