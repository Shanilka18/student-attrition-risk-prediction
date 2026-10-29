# EF-05b Phase Report: EFI Package Rename and Repository Cleanup

## Outcome

The active command-line package is `efi`, and the repository's tracked files
outside `docs/phase_reports/` are guarded against the removed project's
terminology. The active design, configuration, CLI, website copy, and project
documentation now use the Environmental & Financial component's vocabulary.
Work remains on `environmental-chamodya`; no merge to `main` was performed.

## Work completed

- Recorded the initial term inventory in [ef_05b_inventory.md](ef_05b_inventory.md).
- Renamed the CLI package and updated command references. The CLI exposes
  `profiles` and runs as `python -m efi.cli`.
- Removed obsolete component features and model paths, updated active labels,
  and rejected incompatible saved model preprocessors rather than silently
  scoring with a stale feature schema.
- Removed tracked legacy documents and datasets, including the stale generated
  dataset that contained fields from the former design. Removed the local
  `artifacts/legacy` directory. Historical EF phase reports were retained
  because they document this component's work.
- Added `tests/test_repository_terms.py` and an empty-by-default
  `tests/allowlist_terms.txt`. The guard scans tracked files outside this
  phase-report directory and passed with no allow-list entries.
- Updated the README, project context, and decisions log to document the
  current package name and component scope.
- Added ignore rules for local-only reference prompt material. The named
  prompt-pack file was untracked; its local copy was preserved and is ignored.

## Validation

- `python -m efi.cli --help`: passed; lists `profiles`.
- Repository term guard: **2 passed**.
- Cleaner and generator tests: **14 passed**.
- API and leaderboard tests: **30 passed**.
- Inference tests: **6 passed, 3 skipped**.
- Schema/codec mapping tests: **10 passed**.
- Ruff and Black checks passed on changed Python files.
- The affected Python test groups that do not load PyTorch passed. Broader
  collection remains blocked on this Windows environment by PyTorch's
  `c10.dll` initialization failure (`WinError 1114`) in tests that import
  PyTorch; this is not reported as a green full-suite run.
- A separate Ruff/Black probe of the untouched
  `tests/test_schema_codec_mapping.py` reports existing import/style findings;
  that unrelated file was not changed.

## Commits

- `77b614f` — inventory
- `ca6e300` — rename CLI package
- `ab3449b` — remove obsolete active paths
- `5b01f0e` — remove archived assets
- `77b84a3` — add repository guard
- `3aacfd9` — update project documentation

## Governance

The repository instructions define G14 but no active G13 procedure was found
in `AGENTS.md` or `PROJECT_BRIEF.md`. The user explicitly directed that work
remain on `environmental-chamodya` and not merge to `main`; that direction was
followed. The `ef-05b` tag marks this completed phase on the existing branch.
