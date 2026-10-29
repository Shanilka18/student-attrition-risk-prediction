> Summary: This index orders the viva and presentation pack.
> The quick path is designed for a 30-minute review.
> The full path is designed for a 2-hour study session.
> Start with generated results, then read methods and limits.
> All project claims should be checked against the active run artifacts.

# Viva and Presentation Documentation

## 30-minute quick read

1. Read [18_END_TO_END_WALKTHROUGH.md](./18_END_TO_END_WALKTHROUGH.md) for the full project story.
2. Read [19_RESULTS_SUMMARY.md](./19_RESULTS_SUMMARY.md) for the active-run evidence.
3. Study the generated 12-number cheat sheet in [23_PRESENTATION_GUIDE.md](./23_PRESENTATION_GUIDE.md).
4. Read the 15 marked questions in [22_VIVA_QA.md](./22_VIVA_QA.md).
5. Read [24_LIMITATIONS_ETHICS.md](./24_LIMITATIONS_ETHICS.md) before making any claim about student use.

## 2-hour full read

| Time | Read | Purpose |
|---|---|---|
| 0–10 min | This index and [27_GLOSSARY.md](./27_GLOSSARY.md) | Learn the vocabulary. |
| 10–25 min | [18_END_TO_END_WALKTHROUGH.md](./18_END_TO_END_WALKTHROUGH.md) | Follow the pipeline in order. |
| 25–40 min | [19_RESULTS_SUMMARY.md](./19_RESULTS_SUMMARY.md) and [20_FIGURE_CATALOG.md](./20_FIGURE_CATALOG.md) | Check what the active evidence says. |
| 40–60 min | [21_CONCEPTS_FOR_BEGINNERS.md](./21_CONCEPTS_FOR_BEGINNERS.md) | Refresh technical concepts simply. |
| 60–85 min | [22_VIVA_QA.md](./22_VIVA_QA.md) | Practise concise and honest answers. |
| 85–105 min | [23_PRESENTATION_GUIDE.md](./23_PRESENTATION_GUIDE.md) | Rehearse the talk and demo. |
| 105–120 min | [24_LIMITATIONS_ETHICS.md](./24_LIMITATIONS_ETHICS.md), [25_PROPOSAL_UPDATE_NOTES.md](./25_PROPOSAL_UPDATE_NOTES.md), [26_REPRODUCIBILITY_AND_RUN_GUIDE.md](./26_REPRODUCIBILITY_AND_RUN_GUIDE.md) | Review ethics, proposal wording, and reproducibility. |

## Regeneration

Run `python -m mei.cli docs` after activating the project environment. It rebuilds `docs/_numbers.json`, both results-summary files, the figure catalog, the final figure exports, and the number cheat sheet from the active run. The narrative files remain reviewable Markdown. Update any narrative that depends on a changed method or real-data result.

The results script does not use a website build or add viva material to the website. Do not copy these files into the frontend.

## Common mistakes

- Quoting a value from an old phase report instead of the active [19_RESULTS_SUMMARY.md](./19_RESULTS_SUMMARY.md).
- Treating generated-data results as evidence about real students.
- Reading an association or a SHAP contribution as a cause.

## If the examiner asks

Start with the exact active-run summary. State whether the evidence is generated or survey-based, then give the relevant limitation before interpreting a metric.
