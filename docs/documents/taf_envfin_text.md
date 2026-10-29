# TAF Section 10 Replacement Text

**Sub Objective:** Environmental & Financial Factors

**Tasks**

Design and administer a closed-ended questionnaire covering household income, financial aid, family financial duties, study-cost difficulty, commute time, cost and mode, living situation, study space, internet, device, mobile data, part-time work and hours, financial shocks, money worry, and classes missed because of these barriers. Preprocess responses using the approved item mappings and handle conditional and not-applicable values correctly. Derive Financial Stress, Commute Burden, Digital Access Gap, and Study Environment indices, then use clustering or latent-profile methods to describe environmental and financial risk profiles. Train a Gradient Boosting classifier (XGBoost or LightGBM) to predict the shared at-risk label and compare it with Logistic Regression and Random Forest under the same leakage-controlled validation. Use SHAP to explain feature contributions, report group-wise fairness, and produce the Environmental & Financial Index (EFI, 0–100).

**Novelty**

The component combines financial, travel, digital-access, and home-study conditions in one explainable index rather than relying on a single factor such as household income. It also discovers latent risk profiles, explains how measured conditions contribute to predictions, and reports model performance and error patterns across groups. Together, these outputs provide a broader, interpretable view of study-condition risk.
