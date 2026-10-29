# Four-Component Survey Agreement

All four components use one approved master form and data contract. No component changes shared wording, coding, section order, or export fields independently.

## Shared consent text

Use this exact text in all four components after supervisor and ethics approval:

> You are invited to take part in research about conditions related to continuing higher education. Participation is voluntary; you may skip a question or stop before submitting. Questions include household finances and study conditions. We do not request names or contact details. The same pseudonymous ID links components, so responses are not fully anonymous. Data are kept in approved restricted storage and reported in aggregate, with small groups suppressed. Responses do not affect grades, discipline, eligibility, or individual support decisions. Select “I agree” to consent; otherwise exit.

## Common form and data contract

| Item | Shared rule |
|---|---|
| `student_id` | Reuse one random pseudonym across components: `P-` plus five digits. Check uniqueness; never use institutional IDs or as a feature. |
| `timestamp` | ISO 8601 UTC; metadata only, not a feature or merge key. |
| Attention check | Same approved wording, options, key, and failure code; freeze before launch. Exact item/key are not in the reviewed materials. |
| Shared outcome | Same approved intention items and scoring: `attrition_intent_score` 0–100; `at_risk = 1` at 60 or above. This is intention, not withdrawal. |
| Section order | Consent first; one master fixes shared-item and component order/routing. Environmental & Financial is Section 3 in questionnaire v2. |
| Form export | One group-designated owner exports the approved form and response file; no separate component forms. Name the owner before launch. |
| Merge key | `student_id` only; validate uniqueness and overlap. Never merge on timestamp or direct identifiers. |
| Minimum sample size | Proposal target: at least 300 eligible records, subject to approval; the same rule applies to all components. |
| Data sharing | Store pseudonymous data in approved restricted storage. No real responses or linkage files in Git, public cloud, or website. Share aggregates only, with one suppression rule. |
| Ethics approval status | Unverified; treat as pending. No real-data collection or sharing before documented approval of the common form and data plan. |

**Release gate:** Group sign-off on shared wording, coding, order, export, sample rule, storage, and approval before opening the form.
