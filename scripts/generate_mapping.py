"""Script to generate config/mapping.yaml deterministically."""

from pathlib import Path
import yaml

period_labels = {
    1: ["Period 1: Weeks 1–4", "Period 1 (Weeks 1-4)", "Period 1", "P1"],
    2: ["Period 2: Weeks 5–8", "Period 2 (Weeks 5-8)", "Period 2", "P2"],
    3: ["Period 3: Weeks 9–12", "Period 3 (Weeks 9-12)", "Period 3", "P3"],
    4: ["Period 4: Weeks 13–16", "Period 4 (Weeks 13-16)", "Period 4", "P4"],
}

indicator_questions = {
    "resource_access": [
        "How many days per week did you open online learning materials?",
        "Days per week accessing online learning materials",
        "Resource Access",
        "resource_access",
    ],
    "study_time": [
        "How many hours per week did you spend studying independently?",
        "Hours per week spent studying",
        "Study Time",
        "study_time",
    ],
    "review_time": [
        "How many hours per week did you review past lecture materials?",
        "Hours per week reviewing past material",
        "Review Time",
        "review_time",
    ],
    "quiz_practice": [
        "How often did you attempt quizzes or practice questions?",
        "Attempted quizzes or practice questions",
        "Quiz & Practice Questions",
        "quiz_practice",
    ],
    "assignment_engagement": [
        "How consistently did you start and submit assignments on time?",
        "Started and submitted assignments on time",
        "Assignment Engagement",
        "assignment_engagement",
    ],
    "study_consistency": [
        "I studied on a regular, structured schedule",
        "Studied on a regular schedule",
        "Study Consistency",
        "study_consistency",
    ],
    "participation": [
        "How often did you participate in lectures, discussions, and group work?",
        "Took part in lectures, discussions, group work",
        "Class Participation",
        "participation",
    ],
}

columns_mapping = {}

# Meta / ID / Consent
columns_mapping["student_id"] = [
    "student_id",
    "Student ID",
    "Student Identifier",
    "ID",
    "Anonymous ID",
    "Student Number",
    "Registration Number",
    "Student_ID",
    "StudentId",
]

columns_mapping["timestamp"] = [
    "timestamp",
    "Timestamp",
    "Submission Time",
    "Date submitted",
    "Time",
    "Timestamp (UTC)",
    "Survey Timestamp",
]

columns_mapping["consent"] = [
    "consent",
    "Consent",
    "Informed Consent",
    "Do you consent to participate in this study?",
    "Participation Consent",
    "I agree to take part in this research study",
]

# 28 Features
for ind_key, questions in indicator_questions.items():
    for p in range(1, 5):
        col_name = f"{ind_key}_p{p}"
        variants = [
            col_name,
            f"{ind_key}_P{p}",
            f"{ind_key} P{p}",
            f"{ind_key} period {p}",
            f"{ind_key}_{p}",
        ]
        for q in questions:
            for pl in period_labels[p]:
                variants.append(f"{q} [{pl}]")
                variants.append(f"{q} - {pl}")
        # Remove duplicates while preserving order
        unique_variants = list(dict.fromkeys(variants))
        columns_mapping[col_name] = unique_variants

# Attention check
columns_mapping["attention_check"] = [
    "attention_check",
    "Attention Check",
    "Quality Control Check",
    "Please select 'Sometimes' to confirm you are reading each question carefully.",
    "Attention check: Select 'Sometimes' [Period 1]",
    "Please select 'Sometimes' to confirm you are paying attention",
    "Attention Check [Period 1: Weeks 1–4]",
]

# Outcomes
columns_mapping["intent_leave"] = [
    "intent_leave",
    "I have seriously thought about leaving my degree",
    "A1: I have seriously thought about leaving my degree",
    "A1 [I have seriously thought about leaving my degree]",
    "A1_intent_leave",
    "Outcome A1: Leave degree",
]

columns_mapping["intent_defer"] = [
    "intent_defer",
    "I have considered deferring or taking a break",
    "A2: I have considered deferring or taking a break",
    "A2 [I have considered deferring or taking a break]",
    "A2_intent_defer",
    "Outcome A2: Defer studies",
]

columns_mapping["intent_not_continue"] = [
    "intent_not_continue",
    "I do not plan to continue next semester",
    "A3: I do not plan to continue next semester",
    "A3 [I do not plan to continue next semester]",
    "A3_intent_not_continue",
    "Outcome A3: Not continue next semester",
]

mapping_data = {
    "version": "1.0.0",
    "fuzzy_matching": {
        "enabled": True,
        "threshold": 88.0,
        "scorer": "token_sort_ratio",
        "auto_accept": False,  # Strict Rule: Suggest only, never auto-accept
        "description": "Fuzzy fallback used only to suggest mappings for manual user confirmation.",
    },
    "columns": columns_mapping,
}

out_path = Path("config/mapping.yaml")
with open(out_path, "w", encoding="utf-8") as f:
    yaml.dump(mapping_data, f, sort_keys=False, allow_unicode=True)

print(f"Generated {out_path} covering {len(columns_mapping)} columns.")
