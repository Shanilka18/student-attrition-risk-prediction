# Figure Index

| Figure | Description | Method / interpretation |
|---|---|---|
| [01_item_distributions.png](./01_item_distributions.png) | Response distribution for each questionnaire item | Shares of canonical text response levels from the generated full sample |
| [02_item_correlations.png](./02_item_correlations.png) | Item-by-item association heat map | Pearson correlations of item-specific pressure-oriented ranks; descriptive only |
| [03_missingness_map.png](./03_missingness_map.png) | Missingness across items and upload records | Shows raw-export blanks from configured random missingness |
| [04_at_risk_by_type.png](./04_at_risk_by_type.png) | At-risk share by synthetic student type | Uses hidden generator type labels for synthetic validation only |
| [05_at_risk_by_income_band.png](./05_at_risk_by_income_band.png) | At-risk share by household income band | Synthetic association; not a real population comparison |
| [06_configured_vs_achieved.png](./06_configured_vs_achieved.png) | Configured response-quality targets compared with achieved rates | Integer sample counts may differ slightly from target percentages |

All six figures are generated at 300 DPI with the configured brand palette by
`python -m mei.cli make-sample-data`. See
[`03_sample_data.md`](../03_sample_data.md) for generation settings,
calibration decisions, output definitions, and limitations.
