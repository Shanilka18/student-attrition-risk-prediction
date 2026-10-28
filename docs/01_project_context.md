# 01 — Project Context and Component Architecture

## What it is
This research project is **Component 2: Multimodal Student Engagement Intelligence (MEI)** within a broader four-component undergraduate research initiative titled *"Multi-Factor Student Attrition Risk Prediction Using Academic, Engagement, Psychosocial, Environmental Factors"* (Project ID: `KND_CDAP_IT_05`). 

The overall initiative fuses four distinct dimensions of risk into an early-warning student attrition prevention system:
1. **Academic Intelligence Framework** (IT23164680): Historical GPA trajectories, examination scores, and sequential performance decay.
2. **Multimodal Student Engagement Intelligence (MEI - THIS COMPONENT)** (IT23191242): Multidimensional engagement trajectories across 4 semester periods, latent engagement states (KMeans, GMM, HMM), and calibrated risk index (0–100).
3. **Psychosocial Intelligence Framework** (IT23279148): Validated psychological scales measuring stress (PSS-10), academic motivation (AMS), self-efficacy (GSE), and institutional belonging (PSSM).
4. **Environmental & Financial Intelligence Framework** (IT23445154): Socio-economic constraints, household income brackets, commute times, external employment, device access, and causal inference.

### One-Paragraph Plain Explanation
Student dropout in higher education rarely happens overnight; it begins with subtle changes in how students interact with their learning environment across the semester. The Multimodal Student Engagement Intelligence (MEI) component analyzes 4 distinct periods of the academic term using 7 key indicators—ranging from resource access and study time to quiz attempts and class participation—collected via a structured questionnaire. By learning latent behavioral states (such as consistent, sporadic, or declining engagement) and training a leaderboard of machine learning models, MEI produces a single calibrated risk score from 0 to 100 along with plain-language explanations of what is driving that risk, allowing university advisors to intervene well before academic failure or withdrawal occurs.

## Why we do it
Traditional early-warning systems suffer from two fatal weaknesses:
1. **Lagging Indicators**: They flag students only after significant academic decline (such as GPA dropping below 2.0 or failing midterms), when it is often too late for remediation.
2. **Single-Signal Oversimplification**: Systems that track engagement often rely solely on single trace variables (e.g., Moodle login count), ignoring multidimensional habits such as study consistency, review time, and interactive participation.

MEI provides early, trajectory-aware, and explainable assessment of engagement decline throughout the semester.

## The Critical Pivot: Proposal vs. Brief Truth
In the preliminary project proposal and earlier Topic Assessment Form (TAF), behavioral data were envisioned as automated trace exports from Learning Management Systems (Moodle logs, Zoom attendance, mouse-tracking metrics). 

**The Current Operational Truth (PROJECT_BRIEF.md):**
- **Overriding Correction**: Campus LMS server trace logs, Zoom logs, and biometric/mouse telemetry are **outdated and not used**.
- **Questionnaire-Driven Ingestion**: All empirical data are gathered via a single anonymous, structured student questionnaire (Google Forms style) split into 4 chronological periods.
- **Why this changed**: Direct server LMS log access across entire faculties raised severe privacy, administrative data governance, and ethical hurdle constraints. A standardized, closed-ended retrospective questionnaire captures multidimensional self-reported behavioral allocation consistently without ethical barriers or tracking overhead.
- **Agnostic Architecture**: The codebase is architected to remain data-source agnostic (`data_source: sample | survey`), currently built on synthetic survey generator data that perfectly mirrors the survey schema and can be swapped for real questionnaire responses via YAML mapping alone.

## Questionnaire Design (Engagement Component)
The engagement questionnaire partitions the 16-week semester into **4 periods** ($P_1, P_2, P_3, P_4$, approximately 4 weeks each):
- **$P_1$ (Early)**: Weeks 1–4 (Orientation, initial lectures)
- **$P_2$ (Early-Middle)**: Weeks 5–8 (Formative assessments, mid-semester ramp)
- **$P_3$ (Late-Middle)**: Weeks 9–12 (Major assignments, project deadlines)
- **$P_4$ (Recent)**: Weeks 13–16 (Revision period, final submissions)

Across each of the 4 periods, **7 engagement indicators** are measured on a 1–5 ordinal scale (yielding 28 temporal inputs):
1. `resource_access`: Days per week online learning materials were opened ($1 = 0$, $2 = 1\text{–}2$, $3 = 3\text{–}4$, $4 = 5\text{–}6$, $5 = \text{every day}$).
2. `study_time`: Weekly independent study hours ($1 = <2$, $2 = 2\text{–}5$, $3 = 6\text{–}10$, $4 = 11\text{–}15$, $5 = >15$).
3. `review_time`: Weekly hours reviewing past lecture material ($1 = \text{none}$, $2 = <1$, $3 = 1\text{–}2$, $4 = 3\text{–}4$, $5 = >4$).
4. `quiz_practice`: Attempt frequency on quizzes and practice questions ($1 = \text{never}$ to $5 = \text{always}$).
5. `assignment_engagement`: Timely start and submission of coursework ($1 = \text{never}$ to $5 = \text{always}$).
6. `study_consistency`: Regularity of study schedule agreement ($1 = \text{strongly disagree}$ to $5 = \text{strongly agree}$).
7. `participation`: Attendance and participation in discussions, lectures, and group work ($1 = \text{never}$ to $5 = \text{always}$).

### Outcome Items (Target Definition — NEVER Features)
Three 5-point agreement items measure attrition intent:
- $A_1$: *"I have seriously thought about leaving my degree"*
- $A_2$: *"I have considered deferring or taking a break"*
- $A_3$: *"I do not plan to continue next semester"*

## Formulas and Target Definitions
- **Continuous Target (Track B — Regression)**:
  $$\text{attrition\_intent\_score} = \left(\frac{\frac{1}{3}\sum_{i=1}^3 A_i - 1}{4}\right) \times 100$$
  This yields a continuous scale from $0$ to $100$.

- **Binary Target (Track A — Classification)**:
  $$\text{at\_risk} = \begin{cases} 1 & \text{if } \text{attrition\_intent\_score} \ge 60 \\ 0 & \text{otherwise} \end{cases}$$

- **Multimodal Engagement Index (MEI)**:
  $$\text{MEI} = \text{round}(100 \times \hat{P}(\text{at\_risk} = 1))$$
  calibrated output from the champion Track A classification model.
  - Low Risk: $0 \le \text{MEI} \le 34$
  - Moderate Risk: $35 \le \text{MEI} \le 64$
  - High Risk: $65 \le \text{MEI} \le 100$

## Decisions and Rejected Alternatives
- **Decision**: Overrode proposal text regarding LMS/Zoom server clickstreams in favor of structured retrospective questionnaires.
  - *Reason*: Aligns with `PROJECT_BRIEF.md`, eliminates unfeasible university database access dependencies, and addresses retrospective behavioral dimensions (such as independent study hours and consistency) that server logs cannot observe.
- **Decision**: Strict separation of survey outcome questions ($A_1\text{–}A_3$) from model inputs.
  - *Reason*: Rule G3 prohibits target leakage. Including intent questions as predictors would produce trivial, non-generalizable models.

## Common Mistakes
- Confusing our component with the overall 4-component project; MEI focuses specifically on the Engagement dimension.
- Assuming server telemetry is still expected; survey questionnaire responses represent the sole source of truth.
- Allowing outcome items $A_1\text{–}A_3$ to leak into the feature matrix.

## How to Explain This in a Viva
"Our work addresses the student engagement dimension of a four-part attrition early-warning initiative. Rather than treating engagement as a static tally of Moodle logins, we capture temporal trajectory patterns across four four-week periods using seven validated indicators. We discover latent engagement states using clustering and Hidden Markov Models, feed these into a competitive multi-model leaderboard, and calibrate the champion model's probability into the Multimodal Engagement Index (MEI, 0–100) with local SHAP attributions."

## Explain Like I Am 5
"Imagine four doctors checking on a patient: one checks blood tests (grades), one checks how healthy they feel inside (feelings), one checks if they have a warm coat and food (money), and we check how much energy they have to run around and play in class (engagement). We look at how active they were four different times during school to help them before they get sick."
