# Real-Survey Handover Checklist

## On the day real survey results arrive

1. Confirm institutional approval, participant consent, retention terms, and who may access the data.
2. Keep names and contact details outside the project. Store any re-identification key separately.
3. Place the approved export in the ignored `data/real/` directory; never commit responses or screenshots containing them.
4. Validate the file against the current questionnaire schema and mapping; review rejected rows and missingness before training.
5. Confirm the outcome is still the self-reported intention measure. Do not call it confirmed withdrawal.
6. Create a new, preregistered train/validation/test protocol. Do not reuse the existing held-out test after its prior evaluation history.
7. Run the full pipeline on the approved cohort; review cleaning, split, feature, state, model, calibration, uncertainty, subgroup, and privacy evidence before activation.
8. Rebuild docs and presentation materials; have a second reviewer check every displayed number and approve the model card before any supportive pilot.

## Re-read after the real-data run

Read [19_RESULTS_SUMMARY.md](docs/19_RESULTS_SUMMARY.md), [22_VIVA_QA.md](docs/22_VIVA_QA.md), [23_PRESENTATION_GUIDE.md](docs/23_PRESENTATION_GUIDE.md), [24_LIMITATIONS_ETHICS.md](docs/24_LIMITATIONS_ETHICS.md), and [26_REPRODUCIBILITY_AND_RUN_GUIDE.md](docs/26_REPRODUCIBILITY_AND_RUN_GUIDE.md). Recheck [DECISIONS.md](docs/DECISIONS.md) and add a dated decision for the new split and evaluation protocol.

## Twelve slide values to refresh

Update these from the new `docs/_numbers.json` and generated reports; do not copy old sample values:

1. Data-source label and survey cohort size.
2. Training-set sample size.
3. Validation-set sample size.
4. Held-out test-set sample size.
5. At-risk class prevalence.
6. Classification champion and its cross-validation score.
7. Held-out test F1 and confidence interval.
8. Held-out test ROC AUC and confidence interval.
9. Regression champion's held-out R-squared and confidence interval.
10. Selected K-means cluster count and silhouette score.
11. Selected HMM state count and convergence status.
12. SHAP rank-consistency summary and statistical-significance verdict.

## Known limitations

- Current checked-in example outputs use generated data, not real survey responses.
- The target is self-reported intention, not confirmed dropout.
- The survey has four time points; longer-term trajectories are unknown.
- Existing artifacts conflict about held-out-test use. A new, preregistered cohort split is required for defensible test claims.
- State-feature improvements were not consistently statistically significant, regression evidence was weak, and SHAP feature ranks were unstable.
- Survey response bias and fairness cannot be ruled out simply because demographic fields are excluded.
- The interface is a research/support aid, not a basis for grades, discipline, or automated student decisions.

## Sharing the project folder

Include source, configuration, templates, documentation, tests, and the frontend lockfile. Exclude `.venv/`, `frontend/node_modules/`, `frontend/dist/`, `data/real/`, local raw/interim data, generated model artifacts, caches, secrets, and any file that could identify a participant. Share real-data results only through an institution-approved, access-controlled channel.
