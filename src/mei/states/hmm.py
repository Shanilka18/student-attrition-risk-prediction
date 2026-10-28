"""Hidden Markov Model for student engagement state sequence analysis.

Fits GaussianHMM (covariance_type="diag") over longitudinal 4-period sequences
of the 7 canonical indicators. Selects K in [2, 4] using BIC and cross-validated
log-likelihood with a simplicity preference rule (BIC diff < 10 -> simpler K).
States are deterministically ordered by mean composite engagement.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

CANONICAL_INDICATORS = [
    "resource_access",
    "study_time",
    "review_time",
    "quiz_practice",
    "assignment_engagement",
    "study_consistency",
    "participation",
]


@dataclass
class HMMModelSelectionResult:
    """Results from evaluating GaussianHMM across K in [2, 4]."""

    best_model: Any
    best_k: int
    best_bic: float
    model_table: pd.DataFrame
    scaler: StandardScaler
    indicator_names: List[str]
    state_names: Dict[int, str]
    transition_matrix: np.ndarray  # (K, K)
    start_probabilities: np.ndarray  # (K,)
    emission_means: np.ndarray  # (K, 7)
    expected_dwell_times: Dict[int, float]
    converged: bool


def prepare_sequence_data(
    df: pd.DataFrame,
    indicators: Optional[List[str]] = None,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, List[int], StandardScaler, List[str]]:
    """Format longitudinal survey DataFrame into hmmlearn sequence array.

    Expected input: DataFrame with columns `<indicator>_p<1..4>` for 7 indicators.
    Output:
    - X_stacked: 2D array of shape (N * 4, 7)
    - lengths: list of [4] * N
    - scaler: fitted StandardScaler
    - indicator_names: canonical indicator list
    """
    if indicators is None:
        indicators = CANONICAL_INDICATORS

    n_students = len(df)
    n_indicators = len(indicators)
    n_periods = 4

    # Build (N, 4, 7) tensor
    tensor = np.zeros((n_students, n_periods, n_indicators), dtype=float)
    for ind_idx, ind in enumerate(indicators):
        for p_idx, p in enumerate([1, 2, 3, 4]):
            col = f"{ind}_p{p}"
            if col not in df.columns:
                raise ValueError(f"Missing required indicator column: {col}")
            tensor[:, p_idx, ind_idx] = df[col].values

    # Reshape to (N * 4, 7)
    X_stacked = tensor.reshape(n_students * n_periods, n_indicators)

    if scaler is None:
        scaler = StandardScaler()
        if fit_scaler:
            X_scaled = scaler.fit_transform(X_stacked)
        else:
            X_scaled = X_stacked
    else:
        if fit_scaler:
            X_scaled = scaler.fit_transform(X_stacked)
        else:
            X_scaled = scaler.transform(X_stacked)

    lengths = [n_periods] * n_students
    return X_scaled, lengths, scaler, indicators


def _compute_hmm_bic(
    log_likelihood: float,
    k: int,
    n_features: int,
    n_samples: int,
) -> float:
    """Compute Bayesian Information Criterion for GaussianHMM with diagonal covariance.

    Parameters:
    - Initial probabilities: K - 1
    - Transition matrix: K * (K - 1)
    - Emission means: K * n_features
    - Emission diagonal covariances: K * n_features
    Total parameters p = (K - 1) + K(K - 1) + 2 * K * n_features = K^2 - 1 + 2 * K * n_features.
    BIC = -2 * ln(L) + p * ln(N_observations).
    """
    p = (k**2 - 1) + (2 * k * n_features)
    n_obs = n_samples * 4
    bic = -2.0 * log_likelihood + p * np.log(n_obs)
    return float(bic)


def _order_hmm_states(model: Any) -> Any:
    """Deterministically permute HMM states in ascending order of composite engagement.

    State 0 (State 1 in 1-based indexing) is assigned to the lowest engagement level.
    """
    # Composite engagement of each state is the average standardized emission mean
    composite_means = model.means_.mean(axis=1)
    order = np.argsort(composite_means)

    # Permute model parameters
    model.startprob_ = model.startprob_[order]
    model.transmat_ = model.transmat_[order, :][:, order]
    model.means_ = model.means_[order, :]
    model.covars_ = model._covars_[order]
    return model


def fit_gaussian_hmm_with_restarts(
    X_scaled: np.ndarray,
    lengths: List[int],
    k: int,
    n_restarts: int = 10,
    max_iter: int = 100,
    random_seed: int = 42,
) -> Tuple[Any, float, bool]:
    """Fit GaussianHMM with multiple random restarts, keeping the highest log-likelihood model."""
    from hmmlearn.hmm import GaussianHMM

    best_model = None
    best_ll = -np.inf
    converged_any = False

    for r in range(n_restarts):
        seed = random_seed + (r * 101)
        model = GaussianHMM(
            n_components=k,
            covariance_type="diag",
            n_iter=max_iter,
            random_state=seed,
            init_params="stmc",
        )
        try:
            model.fit(X_scaled, lengths)
            ll = model.score(X_scaled, lengths)
            is_converged = bool(model.monitor_.converged)

            if ll > best_ll:
                best_ll = ll
                best_model = model
                converged_any = is_converged
        except Exception as e:
            logger.debug("HMM restart %d for K=%d failed: %s", r, k, str(e))
            continue

    if best_model is None:
        raise RuntimeError(
            f"All {n_restarts} restarts failed for GaussianHMM with K={k}"
        )

    # Reorder states deterministically by composite engagement level
    best_model = _order_hmm_states(best_model)
    return best_model, float(best_ll), converged_any


def select_optimal_hmm(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    k_range: Optional[List[int]] = None,
    n_restarts: int = 10,
    max_iter: int = 100,
    bic_simplicity_threshold: float = 10.0,
    random_seed: int = 42,
) -> HMMModelSelectionResult:
    """Select optimal GaussianHMM model across K in [2, 4] using BIC and validation log-likelihood.

    Rule: Prefer the simpler K if BIC difference < bic_simplicity_threshold (default 10.0).
    """
    if k_range is None:
        k_range = [2, 3, 4]

    X_train_scaled, train_lengths, scaler, indicators = prepare_sequence_data(
        train_df, fit_scaler=True
    )
    n_train_students = len(train_lengths)
    n_features = len(indicators)

    val_lengths = None
    X_val_scaled = None
    if val_df is not None:
        X_val_scaled, val_lengths, _, _ = prepare_sequence_data(
            val_df, scaler=scaler, fit_scaler=False
        )

    records = []
    models_dict = {}

    for k in k_range:
        model, train_ll, converged = fit_gaussian_hmm_with_restarts(
            X_train_scaled,
            train_lengths,
            k=k,
            n_restarts=n_restarts,
            max_iter=max_iter,
            random_seed=random_seed,
        )
        bic = _compute_hmm_bic(train_ll, k, n_features, n_train_students)

        val_ll = np.nan
        if X_val_scaled is not None and val_lengths is not None:
            try:
                val_ll = float(model.score(X_val_scaled, val_lengths))
            except Exception as e:
                logger.warning(
                    "Failed to score validation set on HMM K=%d: %s", k, str(e)
                )

        models_dict[k] = model
        records.append(
            {
                "k": k,
                "train_log_likelihood": train_ll,
                "bic": bic,
                "val_log_likelihood": val_ll,
                "converged": converged,
            }
        )

    model_table = pd.DataFrame(records)

    # Model selection rule:
    # 1. Find candidate with minimum BIC
    min_bic_idx = model_table["bic"].idxmin()
    min_bic_k = int(model_table.loc[min_bic_idx, "k"])
    min_bic_val = float(model_table.loc[min_bic_idx, "bic"])

    # 2. Check simpler candidates: if any simpler K has BIC - min_bic < threshold, select simpler
    chosen_k = min_bic_k
    simpler_candidates = model_table[model_table["k"] < min_bic_k].sort_values("k")
    for _, row in simpler_candidates.iterrows():
        cand_k = int(row["k"])
        cand_bic = float(row["bic"])
        if (cand_bic - min_bic_val) < bic_simplicity_threshold:
            logger.info(
                "Preferring simpler K=%d (BIC=%.2f) over K=%d (BIC=%.2f) as diff < %.1f",
                cand_k,
                cand_bic,
                min_bic_k,
                min_bic_val,
                bic_simplicity_threshold,
            )
            chosen_k = cand_k
            break

    best_model = models_dict[chosen_k]
    best_bic = float(model_table.loc[model_table["k"] == chosen_k, "bic"].iloc[0])
    is_converged = bool(
        model_table.loc[model_table["k"] == chosen_k, "converged"].iloc[0]
    )

    # Assign state names based on chosen K
    if chosen_k == 2:
        state_names = {0: "Low", 1: "High"}
    elif chosen_k == 3:
        state_names = {0: "Low", 1: "Medium", 2: "High"}
    elif chosen_k == 4:
        state_names = {0: "Low", 1: "Medium-Low", 2: "Medium-High", 3: "High"}
    else:
        state_names = {i: f"State {i + 1}" for i in range(chosen_k)}

    # Expected dwell times: E[T_k] = 1 / (1 - A_kk)
    transmat = best_model.transmat_
    dwell_times = {}
    for i in range(chosen_k):
        a_ii = transmat[i, i]
        if a_ii >= 1.0:
            dwell_times[i] = float("inf")
        else:
            dwell_times[i] = float(1.0 / (1.0 - a_ii))

    return HMMModelSelectionResult(
        best_model=best_model,
        best_k=chosen_k,
        best_bic=best_bic,
        model_table=model_table,
        scaler=scaler,
        indicator_names=indicators,
        state_names=state_names,
        transition_matrix=best_model.transmat_,
        start_probabilities=best_model.startprob_,
        emission_means=best_model.means_,
        expected_dwell_times=dwell_times,
        converged=is_converged,
    )


def decode_hmm_sequences(
    model: Any,
    X_scaled: np.ndarray,
    lengths: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode most likely Viterbi state path, posterior probabilities, and sequence log-likelihoods.

    Returns:
    - viterbi_paths: (N, 4) array of 1-indexed state codes (1..K)
    - posteriors: (N, 4, K) array of forward-backward state probabilities
    - seq_log_likelihoods: (N,) array of log p(x_{1:4})
    """
    n_students = len(lengths)
    n_periods = 4
    k = model.n_components

    # 1. Viterbi decoding
    viterbi_flat = model.predict(X_scaled, lengths)
    # Reshape and convert to 1-indexed states (1..K)
    viterbi_paths = viterbi_flat.reshape(n_students, n_periods) + 1

    # 2. Forward-backward posterior probabilities
    posteriors_flat = model.predict_proba(X_scaled, lengths)
    posteriors = posteriors_flat.reshape(n_students, n_periods, k)

    # 3. Per-student sequence log-likelihood
    # Efficient evaluation of 4-step sequence scores
    seq_scores = np.zeros(n_students, dtype=float)
    for i in range(n_students):
        seq = X_scaled[i * n_periods : (i + 1) * n_periods]
        seq_scores[i] = model.score(seq)

    return viterbi_paths, posteriors, seq_scores
