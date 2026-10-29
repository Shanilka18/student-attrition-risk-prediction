# Final Questionnaire Instrument: Environmental & Financial Factors (v2)

**Project:** Student Attrition Risk Prediction (`KND-CDAP-IT-05`)  
**Component:** Environmental & Financial Causal Risk Intelligence Framework (ERI / EFI)  
**Author:** Senior Survey-Methodology Expert & ML Lead  
**Document:** `docs/questionnaire_envfin_v2.md`  
**Target Completion Time:** 3 to 4 minutes (17 closed-ended single-choice items)  
**Deployment Target:** Google Forms / University Survey Portal  

---

## Google Forms Section Header & Instructions

### Section Title:
**Section 3: Environmental & Financial Circumstances (Study Conditions)**

### Section Description (shown to participating students):
> *Please answer the following questions based on your current personal and household circumstances during this semester. Your responses will be used strictly for academic research and statistical analysis to help the university design proactive student support services and emergency bursary interventions.*  
>  
> *All responses are voluntary, confidential, and held under cryptographic pseudonymity. Individual responses will never be used for administrative disciplinary actions or individual grading.*

---

## Questionnaire Items (Q23–Q39)

### 23. Which range best describes your household's total monthly income (combined income of all household members, not just your own)?
- [ ] Less than LKR 25,000
- [ ] LKR 25,000 – 50,000
- [ ] LKR 50,001 – 100,000
- [ ] LKR 100,001 – 200,000
- [ ] More than LKR 200,000
- **Type:** Single-choice (Radio buttons)
- **Conditional Logic:** None.
- **Methodological Rationale:** Baseline household socioeconomic tier. Captures household economic capacity to absorb macro-economic shocks (inflation, currency devaluation, semester fee increases).

---

### 24. Do you receive a scholarship, bursary, or other financial aid for your studies?
- [ ] Yes, full coverage
- [ ] Yes, partial coverage
- [ ] No
- **Type:** Single-choice (Radio buttons)
- **Conditional Logic:** None.
- **Methodological Rationale:** Measures institutional and philanthropic financial buffers that shield economically vulnerable students from direct tuition insolvency.

---

### 25. Are you currently financially responsible for supporting family members (e.g. siblings, parents)?
- [ ] Yes
- [ ] No
- **Type:** Single-choice (Radio buttons)
- **Conditional Logic:** None.
- **Methodological Rationale:** Captures breadwinner burdens. Students providing primary or supplementary household subsistence face acute time and cash trade-offs against degree continuation.

---

### 26. How often do you find it difficult to pay for your study-related costs (fees, travel, materials)?
- [ ] Never
- [ ] Sometimes
- [ ] Often
- [ ] Always
- [ ] Prefer not to say
- **Type:** Single-choice (Radio buttons)
- **Conditional Logic:** None.
- **Methodological Rationale:** Subjective metric of recurring liquidity friction. "Prefer not to say" is explicitly retained to respect privacy while flagging informative non-response (MNAR).

---

### 27. On a typical day, how long does it take you to travel to campus (one way)?
- [ ] Less than 30 minutes
- [ ] 30 – 59 minutes
- [ ] 60 – 120 minutes
- [ ] More than 120 minutes
- [ ] Not applicable (I study fully online)
- **Type:** Single-choice (Radio buttons)
- **Conditional Logic:** If "Not applicable (I study fully online)", travel time burden is structurally zero.
- **Methodological Rationale:** Quantifies physical transit friction and fatigue. Chronic long-distance commuting steals time from sleep, revision, and continuous assessment.

---

### 28. Approximately how much do you spend on travel to campus per month?
- [ ] Less than LKR 2,500
- [ ] LKR 2,500 – 5,000
- [ ] LKR 5,001 – 10,000
- [ ] More than LKR 10,000
- [ ] Not applicable
- **Type:** Single-choice (Radio buttons)
- **Conditional Logic:** Structural N/A for online learners.
- **Methodological Rationale:** Measures direct out-of-pocket transportation costs, enabling calculation of transport-to-income burden ratios.

---

### 29. What is your primary mode of transport to campus?
- [ ] Walking
- [ ] Public transportation
- [ ] Private vehicle (own/family)
- [ ] Hired vehicle (three-wheeler / taxi / PickMe / Uber)
- [ ] Other
- [ ] Not applicable (I study fully online)
- **Type:** Single-choice (Radio buttons)
- **Conditional Logic:** Structural N/A for online learners.
- **Methodological Rationale:** Differentiates transit reliability and daily stress levels (crowded bus/train delays vs. predictable personal transport).

---

### 30. What is your current living situation during the semester?
- [ ] Living with family
- [ ] Hostel / boarding house
- [ ] Rented accommodation
- [ ] Other
- **Type:** Single-choice (Radio buttons)
- **Conditional Logic:** None.
- **Methodological Rationale:** Captures domestic stability, distance from family support networks, and exposure to private rental cost pressures.

---

### 31. Do you have a quiet, suitable place to study where you live?
- [ ] Yes, always
- [ ] Sometimes
- [ ] No
- **Type:** Single-choice (Radio buttons)
- **Conditional Logic:** None.
- **Methodological Rationale:** Environmental study space suitability. Inadequate domestic study space impairs concentration, continuous project work, and exam preparation.

---

### 32. How would you describe your home internet access?
- [ ] Reliable, high-speed
- [ ] Reliable, but slow/limited
- [ ] Unstable / frequently unavailable
- [ ] No home internet access
- **Type:** Single-choice (Radio buttons)
- **Conditional Logic:** None.
- **Methodological Rationale:** Core digital access indicator. Modern higher education requires regular broadband access for LMS lecture slides, laboratory code repositories, and online tests.

---

### 33. What device do you primarily use for coursework?
- [ ] Personal laptop / computer
- [ ] Smartphone only
- [ ] Shared/family device
- [ ] University/library device only
- **Type:** Single-choice (Radio buttons)
- **Conditional Logic:** None.
- **Methodological Rationale:** Device adequacy. Attempting university-level coursework exclusively on a mobile smartphone creates severe computational bottlenecks and study fatigue.

---

### 34. Do you have a mobile data plan you can rely on if home internet is unavailable?
- [ ] Yes, sufficient for coursework
- [ ] Yes, but limited/insufficient
- [ ] No
- **Type:** Single-choice (Radio buttons)
- **Conditional Logic:** None.
- **Methodological Rationale:** Secondary connectivity backup. Protects against academic disconnection during frequent power outages or fiber broadband breakdowns.

---

### 35. Are you currently working a part-time or informal job alongside your studies?
- [ ] No
- [ ] Occasionally
- [ ] Yes
- **Type:** Single-choice (Radio buttons)
- **Conditional Logic:** If "Yes" or "Occasionally", route to Question 36. If "No", skip to Question 37 (job hours structurally default to 0).
- **Methodological Rationale:** Primary employment gate. Differentiates dedicated full-time students from students balancing wage labor alongside their degree.

---

### 36. About how many hours per week do you work in this job?
*(Shown only if answered "Yes" or "Occasionally" to Question 35)*
- [ ] None
- [ ] Less than 10
- [ ] 10–20
- [ ] 21–30
- [ ] More than 30
- **Type:** Single-choice (Dropdown or Radio buttons)
- **Conditional Logic:** Displayed conditionally based on Q35.
- **Methodological Rationale:** Quantifies external employment intensity. Working in excess of 15–20 hours/week is an established threshold for elevated academic disengagement and withdrawal risk.

---

### 37. During this semester, has your household faced a major unexpected expense or loss of income?
- [ ] No
- [ ] Yes, a minor one
- [ ] Yes, a major one
- **Type:** Single-choice (Radio buttons)
- **Conditional Logic:** None.
- **Methodological Rationale:** Measures acute economic shock (family illness, parent job loss, crop failure, inflation surge) that rapidly destabilizes previously manageable tuition payments.

---

### 38. I often worry about whether I can afford to continue my studies.
- [ ] Strongly disagree
- [ ] Disagree
- [ ] Neutral
- [ ] Agree
- [ ] Strongly agree
- **Type:** 5-point Likert Scale
- **Conditional Logic:** None.
- **Methodological Rationale:** Evaluates subjective, psychological financial anxiety. High cognitive worry drains mental bandwidth and precedes formal institutional withdrawal.

---

### 39. In the last month, how often have you missed a class or deadline because of cost, travel or internet problems?
- [ ] Never
- [ ] Once or twice
- [ ] Several times
- [ ] Very often
- **Type:** 4-point Frequency Scale
- **Conditional Logic:** None.
- **Methodological Rationale:** Direct behavioral manifestation of environmental/financial barriers before formal institutional deregistration occurs.

---

## 3. Summary of Canonical Questionnaire Inventory

| # | Canonical Variable Name | Item Type | Number of Levels | Pressure Direction | Structural N/A |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Q23** | `income_band` | Ordinal | 5 | Reversed | None |
| **Q24** | `aid_status` | Ordinal | 3 | Higher | None |
| **Q25** | `family_support_duty` | Binary | 2 | Higher | None |
| **Q26** | `cost_difficulty` | Ordinal | 4 (+ PNS) | Higher | None (PNS = -1) |
| **Q27** | `commute_time` | Ordinal | 4 (+ SNA) | Higher | Online = -2 |
| **Q28** | `commute_spend` | Ordinal | 4 (+ SNA) | Higher | Online = -2 |
| **Q29** | `transport_mode` | Nominal | 5 (+ SNA) | Mapped | Online = -2 |
| **Q30** | `living_situation` | Nominal | 4 | Mapped | None |
| **Q31** | `quiet_study_place` | Ordinal | 3 | Higher | None |
| **Q32** | `home_internet` | Ordinal | 4 | Higher | None |
| **Q33** | `main_device` | Nominal | 4 | Mapped | None |
| **Q34** | `mobile_data_plan` | Ordinal | 3 | Higher | None |
| **Q35** | `part_time_job` | Ordinal | 3 | Higher | Gate for Q36 |
| **Q36** | `job_hours` | Ordinal | 5 | Higher | Non-working = 0 |
| **Q37** | `financial_shock` | Ordinal | 3 | Higher | None |
| **Q38** | `money_worry` | Ordinal | 5 | Higher | None |
| **Q39** | `missed_due_to_barriers`| Ordinal | 4 | Higher | None |
