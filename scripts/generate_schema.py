"""Script to generate config/schema.yaml deterministically."""

from pathlib import Path
import yaml

indicators = {
    "resource_access": {
        "label": "Resource Access",
        "description": "Days per week accessing online learning materials",
        "options": {
            1: "0 days (never)",
            2: "1–2 days",
            3: "3–4 days",
            4: "5–6 days",
            5: "Every day",
        },
    },
    "study_time": {
        "label": "Study Time",
        "description": "Hours per week spent on independent study",
        "options": {
            1: "<2 hours",
            2: "2–5 hours",
            3: "6–10 hours",
            4: "11–15 hours",
            5: ">15 hours",
        },
    },
    "review_time": {
        "label": "Review Time",
        "description": "Hours per week reviewing past lecture materials",
        "options": {
            1: "None",
            2: "<1 hour",
            3: "1–2 hours",
            4: "3–4 hours",
            5: ">4 hours",
        },
    },
    "quiz_practice": {
        "label": "Quiz Practice",
        "description": "Attempted quizzes and practice problem sets",
        "options": {
            1: "Never",
            2: "Rarely",
            3: "Sometimes",
            4: "Often",
            5: "Always",
        },
    },
    "assignment_engagement": {
        "label": "Assignment Engagement",
        "description": "Started and submitted coursework assignments on time",
        "options": {
            1: "Never",
            2: "Rarely",
            3: "Sometimes",
            4: "Often",
            5: "Always",
        },
    },
    "study_consistency": {
        "label": "Study Consistency",
        "description": "Studied according to a regular, structured schedule",
        "options": {
            1: "Strongly disagree",
            2: "Disagree",
            3: "Neutral",
            4: "Agree",
            5: "Strongly agree",
        },
    },
    "participation": {
        "label": "Participation",
        "description": "Took active part in lectures, forum discussions, and group sessions",
        "options": {
            1: "Never",
            2: "Rarely",
            3: "Sometimes",
            4: "Often",
            5: "Always",
        },
    },
}

likert_5 = {
    1: "Strongly disagree",
    2: "Disagree",
    3: "Neutral",
    4: "Agree",
    5: "Strongly agree",
}

cols = []

# 1. Meta / ID / Consent
cols.append(
    {
        "name": "student_id",
        "type": "string",
        "role": "id",
        "period": None,
        "description": "Anonymous unique student identifier token",
        "allowed_codes": None,
        "text_labels": None,
    }
)

cols.append(
    {
        "name": "timestamp",
        "type": "string",
        "role": "meta",
        "period": None,
        "description": "Submission timestamp recorded by survey platform",
        "allowed_codes": None,
        "text_labels": None,
    }
)

cols.append(
    {
        "name": "consent",
        "type": "integer",
        "role": "consent",
        "period": None,
        "description": "Informed consent to participate in research study",
        "allowed_codes": [1, 0],
        "text_labels": {
            1: "I consent to participate",
            0: "I do not consent",
        },
    }
)

# 2. 28 Features (7 indicators x 4 periods)
for ind_name, ind_data in indicators.items():
    for p in range(1, 5):
        w_start = (p - 1) * 4 + 1
        w_end = p * 4
        cols.append(
            {
                "name": f"{ind_name}_p{p}",
                "type": "integer",
                "role": "feature",
                "period": p,
                "indicator": ind_name,
                "description": f"{ind_data['label']} in Period {p} (Weeks {w_start}–{w_end})",
                "allowed_codes": [1, 2, 3, 4, 5],
                "text_labels": ind_data["options"],
            }
        )

# 3. Attention Check
cols.append(
    {
        "name": "attention_check",
        "type": "integer",
        "role": "attention_check",
        "period": None,
        "description": "Attention screener item: Select 'Sometimes' to confirm reading",
        "allowed_codes": [1, 2, 3, 4, 5],
        "expected_code": 3,
        "text_labels": {
            1: "Never",
            2: "Rarely",
            3: "Sometimes",
            4: "Often",
            5: "Always",
        },
    }
)

# 4. 3 Outcomes
cols.append(
    {
        "name": "intent_leave",
        "type": "integer",
        "role": "outcome",
        "period": None,
        "item_id": "A1",
        "statement": "I have seriously thought about leaving my degree",
        "description": "Outcome item A1: Thought about leaving degree",
        "allowed_codes": [1, 2, 3, 4, 5],
        "text_labels": likert_5,
    }
)

cols.append(
    {
        "name": "intent_defer",
        "type": "integer",
        "role": "outcome",
        "period": None,
        "item_id": "A2",
        "statement": "I have considered deferring or taking a break",
        "description": "Outcome item A2: Considered deferring or taking a break",
        "allowed_codes": [1, 2, 3, 4, 5],
        "text_labels": likert_5,
    }
)

cols.append(
    {
        "name": "intent_not_continue",
        "type": "integer",
        "role": "outcome",
        "period": None,
        "item_id": "A3",
        "statement": "I do not plan to continue next semester",
        "description": "Outcome item A3: Do not plan to continue next semester",
        "allowed_codes": [1, 2, 3, 4, 5],
        "text_labels": likert_5,
    }
)

schema_data = {
    "version": "1.0.0",
    "num_columns": len(cols),
    "num_features": 28,
    "num_outcomes": 3,
    "columns": cols,
}

out_path = Path("config/schema.yaml")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    yaml.dump(schema_data, f, sort_keys=False, allow_unicode=True)

print(f"Generated {out_path} with {len(cols)} columns.")
