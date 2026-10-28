"""Deterministic student engagement pattern and level labeling module.

Maps decoded Viterbi state sequences (s_1, s_2, s_3, s_4) to canonical engagement
patterns (Consistent, Declining, Improving, Sporadic) and overall engagement levels
(High, Medium, Low) according to project rules.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class PatternClassification:
    """Classified pattern and engagement level for a student trajectory."""

    pattern: str  # Consistent | Declining | Improving | Sporadic
    level: str  # High | Medium | Low
    mean_state: float
    net_change: int
    n_reversals: int


def classify_trajectory_pattern(
    path: Union[List[int], np.ndarray],
    k: int = 3,
) -> PatternClassification:
    """Classify 4-period state sequence into engagement pattern and level.

    Rules:
    1. Sporadic:
       >= 2 direction changes (up-down or down-up) across consecutive transitions.
    2. Consistent:
       State does not change, OR changes by at most one level once without returning
       (i.e., monotonic transition with total variation <= 1).
    3. Declining:
       Net drop of >= 1 state level from period 1 to period 4 (s_4 <= s_1 - 1)
       with no recovery in the final period (s_4 <= s_3).
    4. Improving:
       Net rise of >= 1 state level from period 1 to period 4 (s_4 >= s_1 + 1)
       with no drop in the final period (s_4 >= s_3).
    5. Fallback:
       Any remaining fluctuating path is classified as Sporadic.

    Overall LEVEL:
    Derived from mean state across 4 periods:
    - For K=3:
      Low: mean < 1.67
      Medium: 1.67 <= mean <= 2.33
      High: mean > 2.33
    - For K=2:
      Low: mean < 1.50
      High: mean >= 1.50
    - For K=4:
      Low: mean < 1.75
      Medium: 1.75 <= mean <= 3.25
      High: mean > 3.25
    """
    p = [int(s) for s in path]
    if len(p) != 4:
        raise ValueError(f"Expected 4-period state path, got length {len(p)}: {p}")

    s1, s2, s3, s4 = p
    mean_state = float(np.mean(p))
    net_change = s4 - s1

    # Transition step differences: d1, d2, d3
    d1 = s2 - s1
    d2 = s3 - s2
    d3 = s4 - s3
    diffs = [d1, d2, d3]

    # Count direction reversals (up-to-down or down-to-up)
    non_zero_diffs = [d for d in diffs if d != 0]
    n_reversals = 0
    if len(non_zero_diffs) >= 2:
        for i in range(len(non_zero_diffs) - 1):
            if (non_zero_diffs[i] > 0 and non_zero_diffs[i + 1] < 0) or (
                non_zero_diffs[i] < 0 and non_zero_diffs[i + 1] > 0
            ):
                n_reversals += 1

    # 1. Check Sporadic by reversal count
    if n_reversals >= 2:
        pattern = "Sporadic"
    # 2. Check Consistent:
    # (a) Exactly constant
    elif s1 == s2 == s3 == s4:
        pattern = "Consistent"
    # (b) Changes by at most one level once without returning:
    # monotonic single step: abs(net_change) <= 1 and sum of abs(diffs) <= 1
    elif abs(net_change) <= 1 and (abs(d1) + abs(d2) + abs(d3)) <= 1:
        pattern = "Consistent"
    # 3. Check Declining: net drop >= 1 and no recovery in last period
    elif s4 <= s1 - 1 and s4 <= s3:
        pattern = "Declining"
    # 4. Check Improving: net rise >= 1 and no drop in last period
    elif s4 >= s1 + 1 and s4 >= s3:
        pattern = "Improving"
    else:
        # Fluctuations that didn't meet other criteria (e.g. [2, 3, 2, 2] with return)
        pattern = "Sporadic"

    # Overall LEVEL
    if k == 2:
        if mean_state < 1.50:
            level = "Low"
        else:
            level = "High"
    elif k == 3:
        if mean_state < 1.67:
            level = "Low"
        elif mean_state <= 2.33:
            level = "Medium"
        else:
            level = "High"
    elif k == 4:
        if mean_state < 1.75:
            level = "Low"
        elif mean_state <= 3.25:
            level = "Medium"
        else:
            level = "High"
    else:
        # Generic quantile thresholds
        if mean_state <= (1.0 + (k - 1) / 3.0):
            level = "Low"
        elif mean_state <= (1.0 + 2.0 * (k - 1) / 3.0):
            level = "Medium"
        else:
            level = "High"

    return PatternClassification(
        pattern=pattern,
        level=level,
        mean_state=mean_state,
        net_change=net_change,
        n_reversals=n_reversals,
    )


def classify_all_trajectories(
    viterbi_paths: np.ndarray,
    k: int = 3,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """Classify an array of shape (N, 4) into pattern labels and level labels.

    Returns:
    - patterns: list of pattern strings
    - levels: list of level strings
    - summary_df: DataFrame with student-level classification details
    """
    patterns = []
    levels = []
    records = []

    for idx, path in enumerate(viterbi_paths):
        res = classify_trajectory_pattern(path, k=k)
        patterns.append(res.pattern)
        levels.append(res.level)
        records.append(
            {
                "student_idx": idx,
                "state_p1": int(path[0]),
                "state_p2": int(path[1]),
                "state_p3": int(path[2]),
                "state_p4": int(path[3]),
                "pattern": res.pattern,
                "level": res.level,
                "mean_state": res.mean_state,
                "net_change": res.net_change,
                "n_reversals": res.n_reversals,
            }
        )

    summary_df = pd.DataFrame(records)
    return patterns, levels, summary_df
