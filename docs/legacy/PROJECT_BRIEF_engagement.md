# PROJECT BRIEF

## Project Details
- **Project ID:** KND-CDAP-IT-05
- **Topic:** Multi-Factor Student Attrition Risk Prediction Using Academic, Engagement, Psychosocial, Environmental Factors
- **Component:** Environmental & Financial Causal Risk Intelligence Framework (ERI / EFI)

## Engineering Guidelines & Rules

- **G14 Small commits.** Work on a branch feature/<topic>, never directly on main. After EVERY numbered STEP of a prompt: run ruff, black --check and pytest for the touched code. If all pass, commit with a conventional message (feat:, fix:, docs:, test:, refactor:, chore:) describing only that step, then `git push -u origin <branch>`. If anything fails, fix it first; never commit failing code; never use --no-verify or --force. One commit = one logical change (code + its tests + its doc). At the end of a phase merge into main with --no-ff, push main and tag ef-NN. Never commit data/real/, .env, .venv, node_modules, model binaries or any file over 5 MB (check `git status` and file sizes before each commit). Show `git log --oneline -10` in every final reply.
