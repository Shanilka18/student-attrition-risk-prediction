> Summary: This glossary gives plain-language definitions for viva terms.
> Entries are ordered alphabetically by the displayed term.
> Definitions explain this project's usage without claiming external evidence.
> See [21_CONCEPTS_FOR_BEGINNERS.md](./21_CONCEPTS_FOR_BEGINNERS.md) for examples.
> See [19_RESULTS_SUMMARY.md](./19_RESULTS_SUMMARY.md) for active-run values.

# Glossary

| Term | Plain definition |
|---|---|
| A1–A3 | Questionnaire outcome items. They are excluded from the predictor features. |
| Ablation | A comparison that removes or adds one feature group to test its contribution. |
| Accuracy | The fraction of predictions that match their labels. It can hide poor performance on a less common class. |
| Attribution | A method-based summary of how input features contribute to a model output. |
| AUC | Area under a curve; ROC AUC summarizes how well scores rank positive cases above negative cases across thresholds. |
| Balanced accuracy | The average of recall for each class. |
| Brier score | The average squared difference between a predicted probability and the observed binary outcome. |
| Bootstrap | Repeated resampling of observed cases to estimate variation in a statistic. |
| Calibration | Agreement between predicted probabilities and observed event frequencies. |
| Champion | The selected model bundle for a task in a particular run. |
| Classification | Prediction of a category, such as a binary intention label. |
| Clustering | Unsupervised grouping of similar observations. |
| Confidence interval | An interval estimate that describes uncertainty under a stated method and assumptions. |
| Cross-validation | Repeated model fitting and evaluation across training folds. |
| Data leakage | Information from an outcome or held-out data improperly influencing model fitting or selection. |
| Deep neural network | A model built from multiple layers of learned transformations. |
| Early stopping | Ending iterative training when validation evidence stops improving. |
| Engagement indicator | A questionnaire item about one aspect of a student's reported study behavior. |
| Fairness | Consideration of whether errors, access, and impacts differ across people or groups. |
| Feature | A value supplied to a model as an input predictor. |
| F1 | A threshold-specific combination of precision and recall. |
| Gradient boosting | An ensemble method that adds learners sequentially to improve predictions. |
| Generalization gap | A difference between model performance on training and unseen validation data. |
| GRU | Gated recurrent unit, a neural network layer that processes sequential inputs. |
| Hidden Markov model | A probabilistic model with hidden states that generate observed sequences. |
| Hyperparameter | A model setting chosen before or during training rather than learned as an ordinary model coefficient. |
| Imputation | A declared method for handling a missing value. |
| Intention label | A self-reported statement about a possible future action; not proof that the action occurred. |
| K-means | A clustering method that assigns observations to a chosen number of groups using distances to group centers. |
| K-nearest neighbours | A method that predicts from the labels or values of nearby training observations. |
| Leaderboard | A table that ranks candidate models using a defined comparison metric. |
| Logistic regression | A linear model that estimates class probabilities. |
| LSTM | Long short-term memory, a gated recurrent neural network layer for sequence data. |
| MAE | Mean absolute error, the average absolute distance between a regression prediction and its target. |
| MCC | Matthews correlation coefficient, a balanced summary based on all confusion-matrix cells. |
| Missingness | The amount and pattern of absent values in collected data. |
| Model card | A record of a model's intended use, training data, performance, and limitations. |
| Outcome | The target value that a model is trained to predict. |
| Overfitting | Learning accidental details of training data that do not carry to new data. |
| Participant | A person who voluntarily provides research data under the approved study process. |
| PR AUC | Area under the precision-recall curve. |
| Questionnaire | A set of standardized questions used to collect survey responses. |
| Random forest | An ensemble that combines predictions from many decision trees. |
| Regression | Prediction of a numeric, continuous outcome. |
| Regularisation | A constraint or penalty that discourages overly complex model fits. |
| RMSE | Root mean squared error, an error measure that gives larger errors extra weight. |
| R-squared | A regression summary comparing model fit with a mean-only reference. |
| SHAP | A feature-attribution method that distributes a model output across input features. |
| Student-level split | A partition that keeps all records for one student in only one data partition. |
| Test set | Data held back for a final evaluation after model choices are frozen. |
| Threshold | A score cutoff that turns a probability or index into a categorical decision. |
| Train set | Data used to fit model parameters and fold-local transformations. |
| Underfitting | A model failing to capture patterns that are present and useful. |
| Validation set | Data used to compare choices without fitting the model on those cases. |
| XAI | Explainable artificial intelligence; methods that summarize model behavior for people. |

## Common mistakes

- Treating an intention label as a confirmed event.
- Treating an attribution as a causal effect.
- Confusing validation evidence with final held-out test evidence.

## If the examiner asks

Define the term in plain language, then explain exactly how it applies to the project's survey, model, or evaluation.
