> Summary: This guide explains the project's technical terms in plain language.
> Each concept has a simple analogy, a student-level definition, and a project example.
> The examples describe the survey design or generated active run.
> They are teaching examples, not additional evidence.
> For the live metrics, use [19_RESULTS_SUMMARY.md](./19_RESULTS_SUMMARY.md).

# Concepts for Beginners

## Data and validation

### Survey data
- **Like I am five:** Asking people questions and writing down their answers.
- **Student level:** Structured self-reports collected with the same questionnaire.
- **Our project:** Students report engagement indicators at several times.

### Cleaning
- **Like I am five:** Fixing messy cards before sorting them.
- **Student level:** Checking formats, allowed values, duplicates, and response quality.
- **Our project:** Survey responses are checked before feature construction.

### Missing values
- **Like I am five:** A blank answer where a sticker should be.
- **Student level:** An absent observation that needs an explicit policy.
- **Our project:** A missing period response is audited rather than guessed silently.

### Features
- **Like I am five:** Clues that help answer a question.
- **Student level:** Predictor variables supplied to a model.
- **Our project:** Engagement level and change summaries are candidate features; A1–A3 are outcomes and must not be predictors.

### Train, validation, and test
- **Like I am five:** Practice questions, a check-up, and a final quiz kept separate.
- **Student level:** Training fits a model, validation supports choices, and a held-out test estimates final performance.
- **Our project:** Students are split into these groups before model fitting.

### Cross-validation
- **Like I am five:** Practising with several different sets of homework.
- **Student level:** Repeating training and validation across folds to estimate variation.
- **Our project:** Leaderboard candidates are compared across stored folds without fitting transforms on held-out students.

### Overfitting
- **Like I am five:** Memorising yesterday's quiz instead of learning the lesson.
- **Student level:** Matching training noise so performance falls on unseen cases.
- **Our project:** A large training-to-validation gap would be a warning.

### Underfitting
- **Like I am five:** Using a rule too simple to notice the important clues.
- **Student level:** A model lacks enough capacity to represent useful structure.
- **Our project:** A very restricted classifier might miss changes across survey periods.

### Regularisation
- **Like I am five:** A rule that stops a drawing from getting too scribbly.
- **Student level:** A penalty that discourages overly complex model parameters.
- **Our project:** Regularised models can reduce overfitting when there are many candidate predictors.

### Early stopping
- **Like I am five:** Stop practising when more practice starts making you worse.
- **Student level:** Halt iterative training when validation performance stops improving.
- **Our project:** A temporal neural model can stop before it memorises the training cohort.

### Hyperparameter tuning
- **Like I am five:** Trying different recipe settings before choosing one.
- **Student level:** Selecting model settings using training and validation data.
- **Our project:** Candidate settings are compared without using the held-out test to choose them.

## Models and states

### Logistic regression
- **Like I am five:** A weighted checklist that adds up clues to estimate yes or no.
- **Student level:** A linear probabilistic classifier.
- **Our project:** It provides a readable baseline for intention classification.

### Random forest
- **Like I am five:** Asking many small decision trees and taking their vote.
- **Student level:** An ensemble of trees trained on varied samples and features.
- **Our project:** It tests whether nonlinear combinations of engagement features help.

### Gradient boosting
- **Like I am five:** Each new helper pays attention to earlier mistakes.
- **Student level:** Sequentially adding weak learners to reduce an objective.
- **Our project:** A boosted-tree candidate competes on the same split strategy.

### Deep nets
- **Like I am five:** A layered set of tiny pattern finders.
- **Student level:** Neural networks with multiple learned transformations.
- **Our project:** Temporal neural models are compared with simpler candidates; complexity does not guarantee better results.

### GRU and LSTM
- **Like I am five:** A notebook that decides what to remember from earlier pages.
- **Student level:** Gated recurrent neural units for sequential inputs.
- **Our project:** A recurrent candidate can process engagement observations in time order.

### Clustering
- **Like I am five:** Putting similar coloured blocks into groups without labels.
- **Student level:** Unsupervised grouping based on a similarity measure.
- **Our project:** Clustering proposes descriptive engagement profiles from features.

### Hidden Markov model
- **Like I am five:** Guessing an invisible mood from the visible clues on each day.
- **Student level:** A probabilistic model of hidden states and observed sequences.
- **Our project:** The HMM represents changing latent engagement states across the survey sequence.

### SHAP
- **Like I am five:** Sharing the credit for an answer among the clues.
- **Student level:** A method that attributes a model output to input features.
- **Our project:** SHAP ranks features for model predictions; low fold stability limits how confidently the ranking can be generalized.

### Leaderboard
- **Like I am five:** A score table for different players.
- **Student level:** A consistent ranking of candidate models by a selected validation score.
- **Our project:** The leaderboard compares classification and regression candidates; see [19_RESULTS_SUMMARY.md](./19_RESULTS_SUMMARY.md).

## Metrics and uncertainty

### Calibration
- **Like I am five:** If a jar says “often,” it should be right about that often.
- **Student level:** Predicted probabilities should match observed event frequencies.
- **Our project:** Calibration is assessed separately from ranking ability; unavailable test calibration metrics are not inferred.

### R-squared
- **Like I am five:** How much of the up-and-down pattern a guess can follow.
- **Student level:** A regression fit statistic relative to a mean-only reference.
- **Our project:** The regression test R-squared is checked alongside its uncertainty and errors.

### F1
- **Like I am five:** One score that rewards finding the right people and not naming too many.
- **Student level:** The harmonic mean of precision and recall for a chosen threshold.
- **Our project:** The classification F1 depends on the decision threshold.

### AUC
- **Like I am five:** How often a good answer gets placed ahead of a bad one.
- **Student level:** Area under a discrimination curve; ROC AUC summarizes ranking across thresholds.
- **Our project:** ROC AUC is reported beside threshold-specific metrics.

### MCC
- **Like I am five:** A score that checks whether the yes and no guesses agree with reality.
- **Student level:** A balanced correlation-like classification measure using all confusion-matrix cells.
- **Our project:** MCC complements F1 when the outcome groups are not equal in size.

### Confidence interval
- **Like I am five:** A range showing that the answer might wiggle if we repeated the test.
- **Student level:** An interval procedure quantifying sampling uncertainty under its assumptions.
- **Our project:** Bootstrap intervals accompany held-out metrics; they do not prove future performance.

## Common mistakes

- Treating an analogy as a full technical definition.
- Assuming a better AUC means probabilities are calibrated.
- Assuming a good SHAP ranking identifies a cause.
- Using a test result to tune the model.

## If the examiner asks

Give the simple intuition, then the precise definition, then the project example. If a method's evidence is weak, say so plainly and refer to [19_RESULTS_SUMMARY.md](./19_RESULTS_SUMMARY.md).
