# Student Engagement Questionnaire (final wording, ready for Google Forms)

Component: Multimodal Student Engagement Intelligence · Project KND_CDAP_IT_05
Completion time: about 5–6 minutes · 12 questions (7 grids, 1 typed number, 3 outcome questions, plus consent and ID)

This is the same questionnaire that the generated sample data copy (same wording, same answer formats, same codes). When the real answers arrive, the system reads this exact format.

---

## Section 0 — Consent and ID

**Consent text (paste at the top of the form)**
This short survey is part of a university research project on how students study during a semester. It is voluntary and anonymous. You can stop at any time. We ask only closed questions (you choose an answer). Your answers are used for research only and are never used to judge you or affect your marks.

**C1. Do you agree to take part?** (Multiple choice, required) — Yes / No (end the form if No)

**C2. Your anonymous study code** (Short answer, required) — Use the code you were given or make one from the last 4 digits of your phone number plus your birth month (e.g. 4821-07). Do not write your name.

---

## Section 1 — How you studied during this semester

**Instructions text**
Think about this semester. We split it into four parts of about 4 weeks each:

- **Period 1** – the first weeks (Weeks 1–4)
- **Period 2** – the next weeks (Weeks 5–8)
- **Period 3** – the weeks after that (Weeks 9–12)
- **Period 4** – the most recent weeks (Weeks 13 to now)

For each question, choose **one answer for each period**. There are no right or wrong answers. Your best guess is fine.

For every grid below: type = **Multiple-choice grid**, rows = Period 1 to Period 4, "Require a response in each row" = ON. The columns (answer choices) differ from question to question, exactly as written.

| # | Question | Answer choices (columns, in this order) | Format |
|---|---|---|---|
| Q1 | On how many days per week did you open online learning materials (slides, notes, recordings, the learning site)? | 0 days · 1–2 days · 3–4 days · 5–6 days · Every day | Range |
| Q2 | About how many hours per week did you study on your own, outside lectures? | Less than 2 hours · 2–5 hours · 6–10 hours · 11–15 hours · More than 15 hours | Time range |
| Q3 | Did you usually review lecture material within 2 days after the lecture? | No · Yes | Yes/No |
| Q4 | How many quizzes or practice question sets did you attempt? | None · 1–2 · 3–5 · 6 or more | Count range |
| Q5 | How many of your assignments did you submit on time? | None · Less than half · Half or more, but not all · All of them | Share |
| Q6 | I studied on a regular schedule. | Strongly disagree · Disagree · Neutral · Agree · Strongly agree | Agreement scale |
| Q7 | How active were you in lectures, discussions and group work? | Low · Medium · High | Low/Medium/High |

**Attention-check row:** inside Q6, add a fifth row called "For this row, please select Neutral" (students who choose anything else are removed in cleaning).

---

## Section 2 — One typed answer

**Q8. In a typical week this semester, about how many hours did you study outside lectures?** (Short answer, number only, 0 to 60)
Used to cross-check Q2 and as an extra numeric feature. Students will still type things like "10 hours" or "8-10"; the cleaning code handles that.

---

## Section 3 — Thinking about your studies (outcome questions)

These three answers define the target the model learns to predict. They are never used as model inputs.

| # | Question | Answer choices (in this order) | Format |
|---|---|---|---|
| Q9 | I have seriously thought about leaving my degree. | Strongly disagree · Disagree · Neutral · Agree · Strongly agree | Agreement scale |
| Q10 | Have you considered deferring or taking a break from your studies? | No · Maybe · Yes | 3 choices |
| Q11 | How likely are you to continue your degree next semester? | Very unlikely · Unlikely · Not sure · Likely · Very likely | Likelihood scale |

---

## How the answers become numbers (the system does this automatically)

Codes start at 1 and are ordered so that a higher code always means more engagement (Q1–Q7).

| Item | Column name pattern | Levels | Codes |
|---|---|---|---|
| Q1 resource access | `resource_access_p1…p4` | 5 | 0 days=1, 1–2=2, 3–4=3, 5–6=4, every day=5 |
| Q2 study time | `study_time_p1…p4` | 5 | <2h=1, 2–5=2, 6–10=3, 11–15=4, >15=5 |
| Q3 review | `review_time_p1…p4` | 2 | No=1, Yes=2 |
| Q4 quizzes/practice | `quiz_practice_p1…p4` | 4 | None=1, 1–2=2, 3–5=3, 6+=4 |
| Q5 assignments | `assignment_engagement_p1…p4` | 4 | None=1, <half=2, half or more=3, all=4 |
| Q6 consistency | `study_consistency_p1…p4` | 5 | Strongly disagree=1 … Strongly agree=5 |
| Q7 participation | `participation_p1…p4` | 3 | Low=1, Medium=2, High=3 |
| Q8 typed hours | `study_hours_exact` | number | 0–60 |
| Q9 | `intent_leave` | 5 | 1–5 (more agreement = more risk) |
| Q10 | `intent_defer` | 3 | No=1, Maybe=2, Yes=3 |
| Q11 | `intent_continue` | 5 | Very unlikely=1 … Very likely=5 (reversed when scoring) |

**Rescaling.** Because items have different numbers of levels, every answer is converted to a 0–1 value with (code − 1) / (levels − 1) before engagement scores are built. Example: "6–10 hours" = code 3 of 5 → 0.50; "Yes" on a yes/no item → 1.0; "Medium" on a 3-level item → 0.50.

**Risk label.** Each outcome item is rescaled to 0–1 (Q11 reversed), then attrition-intent score = 100 × the average of the three. A student is "at risk" when the score is 60 or more. Example: Q9 = Agree (0.75), Q10 = Maybe (0.5), Q11 = Unlikely, reversed (0.75) → 100 × 0.667 = 66.7 → at risk.

---

## Notes for you

- **GPA and money questions** belong to your teammates' components. Their questions follow the same idea (ranges such as below 2.0 / 2.0–2.99 / 3.0–3.49 / 3.5–4.0 instead of an exact GPA; income bands; yes/no access questions). Keep one shared form so all four components use the same students.
- Ask your supervisors to approve the outcome wording (Q9–Q11) and the 60-point rule before you collect data. The label is self-reported intention, not actual dropout, and your documents say so.
- Aim for at least 300 complete responses if you can. Fewer than 150 cannot be used for training; between 150 and 500 works but deep models will probably lose to simpler ones.
