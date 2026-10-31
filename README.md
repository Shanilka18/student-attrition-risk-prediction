
# Multi-Factor Student Attrition Risk Prediction in Higher Education

An Explainable AI-Based Framework for Early Identification of Student Attrition Risk in Sri Lankan Higher Education Institutions.

---

## 1. Project Overview

Student attrition is a significant challenge in higher education. Students may discontinue their studies due to a combination of academic difficulties, reduced engagement, psychosocial challenges, and environmental or financial constraints.

Traditional student monitoring approaches often focus primarily on academic performance. However, academic results alone may not provide a complete understanding of the circumstances influencing a student's decision or ability to continue higher education.

This research proposes a **Multi-Factor Student Attrition Risk Prediction Framework** that integrates four major dimensions:

1. Academic Intelligence
2. Student Engagement Intelligence
3. Psychosocial Intelligence
4. Environmental and Financial Intelligence

The proposed framework analyses these four dimensions to identify potential student attrition risks and generate an integrated risk prediction.

Machine learning techniques and Explainable Artificial Intelligence (XAI) are incorporated to improve prediction capabilities and provide understandable explanations of the factors influencing individual predictions.

The research focuses on the Sri Lankan higher education context and aims to support institutions in identifying students who may benefit from timely academic and other appropriate support.

---

## 2. Research Problem

Student attrition is influenced by multiple interconnected factors. A student experiencing academic difficulties may also face financial constraints, reduced engagement, stress, or insufficient social support.

Analysing these factors independently may overlook important relationships between them.

Therefore, there is a need for an integrated framework that considers multiple dimensions of student life rather than relying exclusively on academic performance.

This research addresses the following challenges:

- Identifying students who may be at risk of discontinuing higher education.
- Analysing multiple dimensions that contribute to student attrition.
- Integrating heterogeneous student information into a unified prediction framework.
- Improving the interpretability of machine learning predictions.
- Providing understandable risk information to support early intervention.

---

## 3. Research Objectives

The main objective is to develop an explainable multi-factor student attrition risk prediction framework for higher education institutions.

Specific objectives include:

- Identify relevant academic, engagement, psychosocial, and environmental/financial factors associated with student attrition.
- Develop four individual analytical components to evaluate different dimensions of student circumstances.
- Extract meaningful features and generate component-level risk representations.
- Integrate the four components through a common multi-factor fusion framework.
- Apply suitable machine learning algorithms to estimate student attrition risk.
- Evaluate and compare model performance using appropriate evaluation metrics.
- Apply Explainable AI techniques to understand the contribution of different factors to individual predictions.
- Present prediction results through an interactive dashboard.

---

# 4. The Four Research Components


## 4.1 Academic Intelligence Component

### Overview

The Academic Intelligence Component analyses students' academic performance and progression patterns to identify early warning signs associated with potential student attrition.

Rather than considering academic performance as a single measurement, this component focuses on changes in academic performance over multiple semesters. This allows the framework to examine students' academic trajectories and identify patterns that may indicate increasing academic difficulties.

The component aims to identify students who may be experiencing declining academic performance, repeated academic difficulties, or challenges in completing their degree requirements.

### Key Academic Indicators

The component considers academic information reported by students through the research questionnaire.

**1. Academic Performance**
- Self-reported GPA or CGPA ranges, where collected.
- Students' perceptions of their academic performance.
- Reported academic difficulties.
- Changes in perceived academic performance.

**2. Academic Progression**
- Self-reported study progress.
- Reported difficulties completing modules.
- Reported failed or repeated modules, where collected.
- Challenges in completing academic requirements.

**3. Academic Behaviour and Consistency**
- Self-reported consistency in completing academic activities.
- Reported assignment and assessment difficulties.
- Study-related challenges affecting academic progress.

### Data Sources

Academic Intelligence uses student responses collected through the structured research questionnaire.

The questionnaire gathers self-reported academic information, including GPA ranges, academic difficulties, study progress, and other relevant academic indicators.

No academic records are currently retrieved directly from the SLIIT student information system, university databases, or official institutional records.

### Methodology

1. Collect academic information through the structured research questionnaire.
2. Perform data cleaning and preprocessing.
3. Handle missing values and inconsistent responses.
4. Extract relevant academic indicators from questionnaire responses.
5. Analyse reported academic performance and progression patterns.
6. Investigate relationships between academic indicators and other student-related risk factors.
7. Explore suitable machine learning approaches based on the available data and prediction objectives.
8. Generate an Academic Risk Index (ARI) on a 0–100 scale.
### Academic Risk Index (ARI)

The Academic Risk Index represents the estimated academic-related attrition risk based on the student's academic performance and progression indicators.

The index considers multiple academic indicators rather than relying on a single GPA value.

A higher ARI indicates a higher estimated academic risk within the component's scoring framework.

### Expected Output

- Semester-wise academic performance analysis.
- Academic progression indicators.
- Academic performance trend visualizations.
- Academic Risk Index (ARI: 0–100).
- Academic risk information for integration into the Multi-Factor Fusion Framework.

## 4.2 Student Engagement Intelligence Component

### Overview

The Student Engagement Intelligence Component evaluates students' involvement and participation in academic activities, particularly their interaction with digital learning environments and university learning platforms.

Student engagement is an important dimension of student attrition because changes in learning participation may provide early indications of reduced academic involvement.

For example, students who frequently access learning materials, follow course announcements, and engage with recorded lectures may demonstrate different engagement patterns from students who rarely interact with these resources.

This component focuses on identifying student engagement patterns and evaluating how reduced or inconsistent engagement may be associated with potential student attrition.

### Key Engagement Dimensions

The component focuses on the following dimensions:

**1. Learning Management System (LMS) Engagement**
- Frequency of logging into the university CourseWeb platform.
- Regularity of accessing CourseWeb.
- Frequency of interacting with course-related resources.
- Consistency of digital learning platform usage.

**2. Learning Resource Engagement**
- Frequency of accessing recorded lectures.
- Regularity of reviewing lecture materials.
- Engagement with available online learning resources.
- Consistency in accessing academic content outside scheduled lectures.

**3. Academic Communication Engagement**
- Frequency of checking course announcements.
- Regularity of following academic notifications.
- Awareness of course updates and academic information.
- Consistency in monitoring university learning platforms.

### Data Collection

Student Engagement Intelligence uses self-reported information collected through the structured research questionnaire. Students provide information about their learning engagement, including:

- Frequency of CourseWeb usage.
- Access to recorded lectures and learning materials.
- Frequency of checking course announcements.
- Participation in academic activities.
- Engagement with online learning resources.

These indicators represent students' self-reported engagement behaviours. Actual CourseWeb activity logs or LMS usage records are not directly extracted from the university system.
### Methodology

1. Collect student engagement information through the structured questionnaire.
2. Perform data cleaning and preprocessing.
3. Extract relevant engagement indicators from student responses.
4. Analyse self-reported CourseWeb usage frequency and learning resource access patterns.
5. Examine students' reported interaction with recorded lectures and course announcements.
6. Identify patterns of consistent, reduced, or irregular engagement.
7. Investigate relationships between engagement indicators and other student-related risk factors.
8. Generate an Engagement Risk Index (ERI) on a 0–100 scale.
### Engagement Risk Index (ERI)

The Engagement Risk Index represents the estimated attrition risk associated with students' learning engagement patterns.

The index considers multiple engagement dimensions rather than relying exclusively on the number of LMS logins.

A higher ERI indicates a higher estimated engagement-related attrition risk.

The index can help identify students who may require additional academic engagement and learning support.

### Expected Output

- CourseWeb usage and engagement indicators.
- Recorded lecture engagement analysis.
- Academic announcement interaction indicators.
- Student engagement pattern analysis.
- Engagement Risk Index (ERI: 0–100).
- Engagement-related risk information for integration into the Multi-Factor Fusion Framework.
## 4.3 Psychosocial Intelligence Framework

### Overview

The Psychosocial Intelligence Framework focuses on identifying psychosocial factors that may influence students' academic experiences and their ability or willingness to continue higher education.

Academic performance and LMS activity alone may not fully represent the challenges students experience.

Psychosocial factors such as stress, motivation, belongingness, self-efficacy, and peer support can provide additional insights into students' circumstances.

This component uses a structured questionnaire to collect psychosocial information and applies statistical and machine learning techniques to analyse the collected responses.

### Key Psychosocial Dimensions

The framework considers five major dimensions.

| Dimension | Description |
|---|---|
| Stress | Perceived academic responsibilities, workload, and pressure associated with managing multiple tasks. |
| Motivation | Students' willingness, interest, and drive to continue academic activities. |
| Belongingness | Students' sense of acceptance, connection, and inclusion within the university community. |
| Self-Efficacy | Students' belief in their ability to successfully complete academic tasks and overcome challenges. |
| Peer Support | The perceived availability of encouragement, assistance, and support from fellow students. |

### Data Collection

A structured questionnaire is used to collect student responses.

The current questionnaire consists of 15 Likert-scale items, with three questions allocated to each psychosocial dimension.

The five-point Likert scale is used to measure responses.

| Dimension | Questionnaire Items | Number of Items |
|---|---|---:|
| Stress | S1, S2, S3 | 3 |
| Motivation | M1, M2, M3 | 3 |
| Belongingness | B1, B2, B3 | 3 |
| Self-Efficacy | SE1, SE2, SE3 | 3 |
| Peer Support | PS1, PS2, PS3 | 3 |
| **Total** | | **15** |

### Methodology

1. Collect psychosocial questionnaire responses with appropriate participant consent.
2. Perform data cleaning and preprocessing.
3. Apply appropriate scoring and reverse-scoring where required.
4. Calculate scores for the five psychosocial dimensions.
5. Evaluate questionnaire internal consistency using Cronbach's Alpha.
6. Prepare psychosocial features for machine learning.
7. Train and compare suitable machine learning models.
8. Estimate individual student attrition probabilities using the selected prediction approach.
9. Generate a Psychosocial Risk Index (PRI) on a 0–100 scale based on the model-estimated risk.
10. Apply SHAP to explain the contribution of individual psychosocial factors to predictions.

### Machine Learning Models

The component explores suitable algorithms, including:

- Random Forest
- XGBoost
- Ridge Regression as an appropriate regularized baseline where applicable

Model selection is based on evaluation results, generalization performance, and suitability for the prediction task.

### Psychosocial Risk Index (PRI)

The Psychosocial Risk Index represents the model-estimated attrition risk associated with the psychosocial information.

The index is expressed on a 0–100 scale.

A higher score represents a higher estimated risk within the model's prediction framework.

The index is intended as a research and early-warning indicator, not as a psychological diagnosis or a definitive prediction of a student's future.

### Explainable AI – SHAP

SHAP (SHapley Additive exPlanations) is used to explain how individual input features contribute to model predictions.

It supports:

- Identifying influential psychosocial factors.
- Understanding individual student predictions.
- Visualizing positive and negative feature contributions.
- Improving transparency and interpretability.

### Expected Output

- Five psychosocial dimension scores.
- Psychosocial Risk Index (PRI).
- Model-estimated risk information.
- Risk categorization where configured.
- Individual SHAP explanations.
- Visual representations of psychosocial risk factors.

---


## 4.4 Environmental and Financial Intelligence Component

### Overview

The Environmental and Financial Intelligence Component investigates external circumstances and socioeconomic challenges that may influence students' ability to continue higher education.

Unlike academic and engagement indicators, which primarily describe students' academic progress and learning participation, this component focuses on the external conditions surrounding students.

Financial difficulties, transportation challenges, employment responsibilities, and limited access to educational resources may create barriers that affect students' academic continuity.

The component aims to identify these challenges and investigate how their effects may differ across students with different academic and engagement profiles.

### Key Environmental and Financial Dimensions

**1. Financial Circumstances**

This dimension examines financial challenges that may affect students' ability to continue their studies.

Potential indicators include:

- Household financial circumstances.
- Personal financial difficulties.
- Education-related expenses.
- Availability of financial assistance.
- Access to scholarships and financial support.
- Financial difficulties affecting academic participation.

**2. Transportation and Accessibility**

This dimension evaluates challenges associated with travelling to university and accessing educational facilities.

Potential indicators include:

- Distance between home and university.
- Daily commuting time.
- Availability of transportation.
- Transportation costs.
- Difficulties travelling to university.
- Accessibility of educational facilities.

**3. Digital and Learning Resource Accessibility**

This dimension examines students' access to the resources required for higher education.

Potential indicators include:

- Availability of a suitable laptop or computer.
- Reliability of internet connectivity.
- Access to digital learning resources.
- Availability of a suitable study environment.
- Technology-related barriers affecting online learning.

**4. Employment and External Responsibilities**

This dimension considers external responsibilities that may compete with students' academic commitments.

Potential indicators include:

- Part-time employment.
- Number of working hours.
- Work-study balance.
- Family responsibilities.
- Time constraints affecting academic activities.

### Data Collection

Environmental and Financial Intelligence uses information collected through the structured student questionnaire. The questionnaire covers factors such as:

- Financial difficulties and educational expenses.
- Availability of financial support.
- Transportation and commuting challenges.
- Internet connectivity and access to digital devices.
- Employment and part-time work responsibilities.
- Family responsibilities.
- Availability of a suitable study environment.

All these indicators are obtained from student questionnaire responses. No additional institutional database records are currently integrated.

### Methodology

1. Collect environmental and financial information through the structured research questionnaire.
2. Perform data cleaning and preprocessing.
3. Encode categorical responses and prepare relevant numerical features.
4. Analyse financial difficulties, transportation challenges, digital accessibility, employment, and external responsibilities.
5. Identify environmental and financial circumstances that may be associated with student attrition risk.
6. Explore relationships between environmental and financial indicators and other component-level risk factors.
7. Generate an Environmental Risk Index (EnRI) on a 0–100 scale.

### Causal Forest Approach

Causal Forest is an extension of Random Forest methodology designed to estimate heterogeneous treatment effects.

Within this research, it can be explored to investigate whether particular environmental or financial circumstances have different estimated effects on student continuation outcomes across different groups of students.

For example:

- Do financial difficulties have different estimated effects on students with different academic performance levels?
- Do transportation challenges have different estimated effects depending on students' engagement levels?
- Are students experiencing employment-related time constraints more vulnerable under particular academic circumstances?

The analysis aims to provide insights beyond simple correlations by investigating differences in estimated effects while accounting for relevant confounding variables.

Causal Forest requires an appropriately defined exposure or treatment variable, an outcome, and suitable assumptions about confounding. Its estimates should not automatically be interpreted as proof of causation.

### Environmental Risk Index (EnRI)

The Environmental Risk Index represents the estimated environmental and financial risk associated with external circumstances affecting students.

It considers the relevant environmental and financial indicators and the results of the component's analytical process.

The index is expressed on a 0–100 scale, where a higher score represents a higher estimated environmental and financial risk.

### Expected Output

- Financial circumstance indicators.
- Transportation and accessibility analysis.
- Digital resource accessibility indicators.
- Employment and external responsibility indicators.
- Heterogeneous effect analysis using the proposed Causal Forest approach.
- Environmental Risk Index (EnRI: 0–100).
- Environmental and financial risk information for integration into the Multi-Factor Fusion Framework.

# 5. Multi-Factor Fusion Framework

### Overview

The Multi-Factor Fusion Framework is the central integration layer of the proposed research.

It combines the analytical outputs of the four independent research components to generate an overall student attrition risk prediction.

Each component focuses on a different dimension of student circumstances and produces a standardized risk index ranging from 0 to 100.

### Four Component Risk Indices

| Component | Risk Index | Analytical Focus |
|---|---|---|
| Academic Intelligence | Academic Risk Index (ARI) | Academic performance, progression, and performance trends |
| Student Engagement Intelligence | Engagement Risk Index (ERI) | CourseWeb usage, recorded lecture engagement, and academic communication |
| Psychosocial Intelligence | Psychosocial Risk Index (PRI) | Stress, motivation, belongingness, self-efficacy, and peer support |
| Environmental and Financial Intelligence | Environmental Risk Index (EnRI) | Financial circumstances, transportation, digital accessibility, and employment responsibilities |

### Fusion Methodology

1. Receive the four component-level risk indices and their relevant analytical features.
2. Validate and preprocess component outputs.
3. Standardize the component risk representations to ensure consistency.
4. Combine the four component outputs through the shared fusion framework.
5. Apply an appropriate analytical or machine learning approach based on the available data and prediction objectives.
6. Generate an integrated student risk representation.
7. Apply Explainable AI techniques, where supported by the selected model, to interpret the integrated results.
8. Present the overall risk representation and component-level contributions through the dashboard.

The ability to estimate and validate actual student attrition probabilities depends on the availability of reliable student continuation or discontinuation outcome labels.

### Overall Student Attrition Risk

The integrated framework aims to identify students who may be at risk of discontinuing their studies by considering all four dimensions together.

For example, a student may demonstrate:

- Acceptable academic performance.
- Reduced engagement with learning resources.
- Increased psychosocial challenges.
- Significant financial difficulties.

An academic-only prediction approach may not identify all these concerns.

The proposed fusion framework considers the combined information to provide a broader understanding of potential attrition risk.

### Explainability at the Fusion Level

SHAP can be applied to the integrated prediction model to identify the contributions of the four component risk dimensions and their underlying features.

This helps explain:

- Which component contributes most to an individual prediction.
- How different dimensions interact within the model.
- Which factors increase or decrease estimated attrition probability.
- Why two students with similar overall risk scores may have different underlying risk profiles.

### Expected Output

- Academic Risk Index (ARI).
- Engagement Risk Index (ERI).
- Psychosocial Risk Index (PRI).
- Environmental Risk Index (EnRI).
- Overall Student Attrition Risk (0–100).
- Overall risk category, where configured.
- Integrated SHAP explanations.
- Component-level risk visualizations.
# 6. Machine Learning Approach

Machine learning is used to identify patterns in student data and estimate the likelihood of student attrition.

The research explores multiple algorithms to determine suitable approaches for the individual components and the integrated framework.

### Algorithms Under Consideration

| Algorithm | Purpose |
|---|---|
| Random Forest | Identifies complex relationships and patterns using an ensemble of decision trees. |
| XGBoost | Uses gradient boosting to improve predictive performance on structured datasets. |
| Ridge Regression | Provides a regularized linear baseline for suitable numerical prediction tasks. |

### Model Evaluation

Models are evaluated using appropriate metrics depending on the prediction task.

**Classification Metrics**
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- Log Loss
- Brier Score

**Regression Metrics**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
- Spearman Rank Correlation

The project uses appropriate training, validation, and testing procedures to assess model performance and reduce overfitting.

Model selection is based on the overall evaluation results rather than accuracy alone.
### Evaluation Considerations

The evaluation metrics are selected according to the available target variable and prediction task.

Classification metrics require labelled outcome data, such as whether a student continued or discontinued their studies. Regression metrics require an appropriate continuous target variable.

If verified student continuation or discontinuation outcomes are unavailable, model development and evaluation are limited to the objectives supported by the available questionnaire data. Performance on synthetic datasets is treated as technical testing and does not establish real-world student attrition prediction accuracy.

---

# 7. Explainable Artificial Intelligence (XAI)

Model explainability is an important aspect of this research.

A prediction system should provide more than a risk score. It should also help users understand which factors contributed to a particular prediction.

The project explores SHAP to explain model predictions.

### SHAP Capabilities

- Global feature importance.
- Individual prediction explanations.
- Positive and negative feature contributions.
- Comparison of influential factors.
- Improved transparency of model behaviour.

SHAP explanations indicate how features influence a model's output. They do not establish causal relationships between those features and student attrition.

---

# 8. System Architecture

The proposed system follows a modular architecture that allows each research component to be developed, tested, and integrated independently.

### Main Layers

**1. Data Collection Layer**
The data collection layer gathers structured information from undergraduate students through a research questionnaire. The questionnaire covers four major dimensions:

1. **Academic Factors:** Self-reported academic performance, study progress, and academic difficulties.
2. **Engagement Factors:** Self-reported CourseWeb usage, access to learning materials, and academic engagement.
3. **Psychosocial Factors:** Stress, motivation, belongingness, self-efficacy, and peer support.
4. **Environmental and Financial Factors:** Financial difficulties, transportation, internet access, employment, family responsibilities, and study environment.

The collected questionnaire responses are processed and transformed into structured features for the respective intelligence components.

**Data Source Note:** The current prototype relies on questionnaire-based data collection. It does not directly retrieve student records from the SLIIT database, institutional academic information systems, or CourseWeb activity logs.

**2. Data Preprocessing Layer**

Performs data cleaning, missing-value handling, feature encoding, normalization, and feature engineering.

**3. Component Analysis Layer**

Contains four independent analytical components responsible for extracting relevant indicators and generating component-level outputs.

**4. Multi-Factor Fusion Layer**

Combines the outputs from all four components into a unified analytical framework.

**5. Machine Learning Layer**

Performs model training, validation, prediction, and evaluation.

**6. Explainability Layer**

Uses SHAP to interpret model predictions and identify influential factors.

**7. Presentation Layer**

Displays student-level predictions, component scores, risk information, and explanations through an interactive dashboard.

---

# 9. Technology Stack

The following technologies are used or considered in the current development approach.

| Category | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript |
| Backend | Python, FastAPI |
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Statistical Analysis | SciPy |
| Machine Learning | Scikit-learn, XGBoost |
| Explainable AI | SHAP |
| Data Visualization | Matplotlib, Seaborn |
| Development Environment | Visual Studio Code |
| Version Control | Git, GitHub |

Actual dependencies and versions should be maintained in the project's dependency files.

---

## 10. Dataset and Data Management

### 10.1 Data Collection Sources

The research dataset is collected through a structured questionnaire distributed to undergraduate students. The questionnaire gathers information related to the four major dimensions of student attrition risk.

| Intelligence Component | Data Collection Method | Main Indicators |
|---|---|---|
| Academic Intelligence | Student questionnaire | Self-reported GPA, academic performance, study progress, academic difficulties |
| Engagement Intelligence | Student questionnaire | CourseWeb usage frequency, recorded lecture access, announcement checking, learning engagement |
| Psychosocial Intelligence | Student questionnaire | Stress, motivation, belongingness, self-efficacy, peer support |
| Environmental and Financial Intelligence | Student questionnaire | Financial difficulties, transportation, internet access, employment, family responsibilities, study environment |

### 10.2 Data Collection Approach

All current research data are collected directly from students through questionnaire responses.

The research does not currently use:
- Direct student record extraction from the SLIIT database.
- Official academic records retrieved from university information systems.
- Automatically collected CourseWeb or LMS activity logs.
- Other institutional database records.

Engagement indicators, including CourseWeb usage and recorded lecture access, are based on students' self-reported questionnaire responses rather than actual system activity logs.

### 10.3 Data Preprocessing

The collected questionnaire responses undergo preprocessing before being used by the machine learning models. The preprocessing activities include:

- Handling missing or incomplete responses.
- Encoding categorical variables.
- Converting Likert-scale responses into numerical values.
- Normalizing or standardizing numerical features where appropriate.
- Checking data consistency and validity.
- Preparing component-specific datasets for model development.

### 10.4 Synthetic Dataset Usage

Synthetic datasets may be used during the initial development and testing stages when actual questionnaire responses are insufficient or unavailable.

Synthetic records are generated for testing data preprocessing, model training pipelines, API functionality, and dashboard visualizations.

Synthetic data are clearly distinguished from actual questionnaire responses and are not presented as genuine student responses or real institutional records.
# 11. Research Ethics and Privacy

Student attrition prediction involves personal information, including potentially sensitive psychosocial and financial information.

Therefore, responsible data management is essential.

The research considers the following principles:

- Obtain appropriate institutional approval and participant consent.
- Explain the purpose of data collection.
- Collect only information relevant to the research.
- Protect student identifiers and confidential information.
- Restrict access to collected information.
- Use student information only for approved research purposes.
- Avoid discriminatory or stigmatizing interpretations.
- Treat predictions as early-warning indicators rather than definitive outcomes.
- Encourage appropriate supportive interventions instead of punitive decisions.

The psychosocial framework is designed for research and student-support purposes and is not a clinical diagnostic system.

---

# 12. Expected Research Contributions

The proposed research aims to contribute the following:

1. A multi-dimensional student attrition risk prediction framework.
2. Four complementary analytical components covering different student-related dimensions.
3. An integrated fusion approach for combining heterogeneous student information.
4. Application of machine learning techniques to estimate student attrition risk.
5. Explainable predictions using SHAP.
6. A dashboard for communicating risk information and contributing factors.
7. A research approach relevant to student retention in Sri Lankan higher education.

---

# 13. Development Status

The project is being developed incrementally.

| Research Area | Main Development Activities |
|---|---|
| Academic Intelligence | Academic data preparation, feature extraction, and risk analysis |
| Engagement Intelligence | LMS activity processing and engagement feature extraction |
| Psychosocial Intelligence | Questionnaire design, scoring, reliability analysis, ML modelling, and SHAP |
| Environmental and Financial Intelligence | Environmental and financial feature preparation and risk analysis |
| Fusion Framework | Integration of the four component outputs |
| Machine Learning | Model training, comparison, and evaluation |
| Dashboard | Visualization of overall and component-level risk information |
| Validation | Testing, evaluation, and research validation |

Implementation and validation progress should be updated as development continues.

---

# 14. Future Enhancements

Potential future improvements include:

- Complete integration of all four research components.
- Improve the quality and coverage of available student data.
- Validate the framework using authorized real-world student records.
- Improve model calibration and predictive reliability.
- Expand individual and global SHAP visualizations.
- Improve dashboard usability and accessibility.
- Introduce secure persistent data storage where required.
- Evaluate the system with academic stakeholders.
- Explore additional machine learning approaches when justified by the research.

---

# 15. Sustainable Development Goals (SDGs)

This research is aligned with two United Nations Sustainable Development Goals.

### SDG 4 – Quality Education

The research aims to support educational continuity by identifying students who may require additional assistance, contributing to more inclusive and equitable higher education.

### SDG 10 – Reduced Inequalities

By considering financial, environmental, and psychosocial circumstances, the framework recognizes that students may experience different barriers to accessing and continuing higher education.

---

# 16. Project Disclaimer

This project is an academic research prototype intended to explore explainable, multi-factor student attrition risk prediction.

The generated risk scores and predictions are intended to support research and potential student-support processes. They should not be treated as certain outcomes, psychological diagnoses, or standalone decisions about individual students.

The effectiveness of the proposed framework must be established through appropriate validation using authorized and representative real-world data.

---

**Research Area:** Student Attrition Prediction in Higher Education  
**Research Context:** Sri Lanka  
**Approach:** Multi-Factor Machine Learning and Explainable AI  
**Main Components:** Academic, Engagement, Psychosocial, and Environmental/Financial Intelligence
