# Phase 12 Report: Frontend Project, Design System, Application Shell, and Shared Components

**Phase**: 12 — Frontend Project, Design System, Application Shell, and Shared Components  
**Component ID**: `KND_CDAP_IT_05`  
**Date**: 2026-10-27  
**Role**: Senior Product Designer and Front-End Engineer  
**Status**: COMPLETED

---

## 1. What Was Done

1. **Front-End Project Architecture & Tooling**:
   - Initialized a production-ready React 18, Vite, and TypeScript environment under `frontend/`.
   - Configured path aliases (`@/*`), strict TypeScript options (`strict: true`, `noUnusedLocals: true`), and testing integration with Vitest and React Testing Library.
   - Configured Tailwind CSS with custom theme tokens mapped exclusively to `PROJECT_BRIEF.md §10` (`paper #F5F2EC`, `surface #FFFFFF`, `line #DDD8CE`, `ink #1A1D21`, `muted #5B6168`, `forest #1F4E46`). Default Tailwind purples, blues, indigos, and pinks were completely stripped out.
   - Bundled typography locally using Fontsource (`@fontsource/source-serif-4` for display headings and statistics; `@fontsource/public-sans` for interface copy and data tables), ensuring complete offline and air-gapped capability without external CDN dependencies.
   - Enforced tabular numerals globally via CSS font feature settings (`tnum 1`, `cv05 1`).

2. **Automated Design Token & Forbidden Pattern Linter**:
   - Created `frontend/scripts/check-forbidden.cjs`, an automated scanner that parses all 42 frontend source files with mathematical HSL color analysis and regex pattern matching.
   - Prohibits forbidden hues between $245^\circ$ and $335^\circ$ with saturation $> 20\%$ (eliminating purple, indigo, violet, and magenta).
   - Prohibits forbidden marketing buzzwords (`prototype`, `demo`, `beta`, `mock`, `placeholder`, `smart`, `magic`, `powered by ai`) and bans all emojis, sparkles, and robot symbols.

3. **Complete Shared UI Primitive Library (18 Components)**:
   - Built 18 reusable, accessible UI primitives under `frontend/src/components/ui/`: `Button` (Primary forest, Secondary outline, Tertiary ghost, Destructive clay, Loading state), `IconButton`, `Card` (1px line border, 6px radius, zero drop shadows), `Stat` (editorial metric with sample size and direction tags), `Tag` / `StatusTag` (Good fit, Slight overfit, Overfit, Risk tiers with non-color indicators), `Tabs` (Radix UI accessible tab strip), `Tooltip`, `Term` (contextual glossary term with hover/focus definition popovers), `DataTable` (TanStack Table with multi-column sorting and pagination), `ChartCard` (Apache ECharts container), `EmptyState`, `ErrorState`, `FileDropzone` (drag-and-drop CSV/XLSX uploader), `Stepper` (workflow stage indicator), `Dialog`, `Drawer`, `Toast`, and `Skeleton`.

4. **Canonical Institutional Glossary**:
   - Authored `frontend/src/content/glossary.ts` providing plain-English institutional definitions and formal mathematical terms for 16 core concepts: MEI, Risk Band, Latent Pattern, Engagement Level, F1 Score, AUC-ROC, MCC, Balanced Accuracy, Calibration, Generalisation Gap, $R^2$, RMSE, MAE, SHAP Attribution, Hidden Markov Model (HMM) State, and Optimal Decision Threshold.

5. **Application Shell & Layout**:
   - Implemented `AppShell` with a fixed 232px left navigation rail (`Sidebar`) featuring serif institutional branding, active navigation indicators, and offline platform version metadata.
   - Built `TopBar` displaying dynamic real backend provenance ("Training data: Sample survey responses (generated)" or "Training data: Your survey data") and active champion model badge (`Model: k-Nearest Neighbours`) read live from `/api/status`.
   - Created research-appropriate empty-state view placeholders for all primary routes: Overview (`/`), Model Leaderboard (`/leaderboard`), Engagement Patterns (`/patterns`), What Drives Risk (`/drivers`), Score Students (`/score`), Train on New Data (`/train`), 404 (`/not-found`), and an interactive component showcase gallery at `/_components`.

6. **FastAPI Production Integration**:
   - Updated `src/mei/api/app.py` to automatically mount built production assets from `frontend/dist/assets` and serve the client-side SPA fallback for root and deep links via buffered `HTMLResponse` (preventing Starlette `BaseHTTPMiddleware` deadlocks with streaming `FileResponse`).
   - Enabled single-command execution for the entire platform via `python -m mei.cli serve`.

7. **Responsive Verification & Headless Screenshots**:
   - Implemented `scripts/capture_screenshots.py` using headless Chrome to capture responsive viewports:
     - `docs/figures/ui_shell_1440.png` (Desktop 1440px): 76.2 KB
     - `docs/figures/ui_shell_1024.png` (Tablet 1024px): 74.7 KB
     - `docs/figures/ui_shell_390.png` (Mobile 390px): 36.4 KB
     - `docs/figures/ui_shell_components_1440.png` (Showcase 1440px): 148.6 KB

---

## 2. Files Created or Modified

### Created Files
- `frontend/package.json`: Frontend package manifest and npm script runner.
- `frontend/package-lock.json`: Dependency lockfile (308 packages).
- `frontend/tsconfig.json` & `frontend/tsconfig.node.json`: TypeScript compiler configuration.
- `frontend/vite.config.ts`: Vite bundling and development server proxy configuration.
- `frontend/tailwind.config.js`: Custom project design tokens and typography configuration.
- `frontend/postcss.config.js`: PostCSS Tailwind processor.
- `frontend/index.html`: Web application HTML entry point with local typography metadata.
- `frontend/scripts/check-forbidden.cjs`: Automated forbidden token, color, and copy linter.
- `frontend/src/index.css`: Tailwind layer directives, local Fontsource font imports, and tabular numeral utilities.
- `frontend/src/main.tsx`: Application DOM mount and React Query context provider.
- `frontend/src/App.tsx`: React Router layout definitions and route mappings.
- `frontend/src/types/api.ts`: TypeScript contracts mirroring backend API schemas.
- `frontend/src/api/client.ts`: Typed fetch client with structured error mapping.
- `frontend/src/api/queries.ts`: TanStack Query hooks for backend data fetching.
- `frontend/src/content/glossary.ts`: Canonical plain-English definitions and technical descriptions.
- `frontend/src/components/layout/AppShell.tsx`: Root layout shell.
- `frontend/src/components/layout/Sidebar.tsx`: Fixed 232px navigation rail.
- `frontend/src/components/layout/TopBar.tsx`: Contextual header with real active model status.
- `frontend/src/components/ui/Button.tsx`: Accessible button with 4 variants, sizes, and spinner.
- `frontend/src/components/ui/IconButton.tsx`: Square icon button with mandatory aria-label.
- `frontend/src/components/ui/Card.tsx`: 1px line border, 6px radius, zero drop shadow container.
- `frontend/src/components/ui/Stat.tsx`: Research metric card with tabular numerals and sample counts.
- `frontend/src/components/ui/Tag.tsx`: High-contrast status badges and risk tiers with non-color indicators.
- `frontend/src/components/ui/Tabs.tsx`: Radix UI headless accessible tab bar.
- `frontend/src/components/ui/Tooltip.tsx`: Radix UI hover/focus tooltip.
- `frontend/src/components/ui/Term.tsx`: Interactive glossary term popover.
- `frontend/src/components/ui/DataTable.tsx`: TanStack Table with keyboard sorting and pagination.
- `frontend/src/components/ui/ChartCard.tsx`: Apache ECharts visualization container.
- `frontend/src/components/ui/EmptyState.tsx`: Calm, actionable empty container.
- `frontend/src/components/ui/ErrorState.tsx`: Error container with retry callback.
- `frontend/src/components/ui/FileDropzone.tsx`: Drag-and-drop CSV/XLSX file upload zone.
- `frontend/src/components/ui/Stepper.tsx`: Workflow wizard step indicator.
- `frontend/src/components/ui/Dialog.tsx`: Radix UI modal dialog.
- `frontend/src/components/ui/Drawer.tsx`: Off-canvas responsive slide-out drawer.
- `frontend/src/components/ui/Toast.tsx`: Floating notification message.
- `frontend/src/components/ui/Skeleton.tsx`: Content loading placeholder.
- `frontend/src/pages/OverviewPage.tsx`: Executive overview route.
- `frontend/src/pages/LeaderboardPage.tsx`: Model benchmark route.
- `frontend/src/pages/PatternsPage.tsx`: Longitudinal trajectories route.
- `frontend/src/pages/DriversPage.tsx`: SHAP explainability insights route.
- `frontend/src/pages/ScorePage.tsx`: Student cohort scoring route.
- `frontend/src/pages/TrainPage.tsx`: Pipeline retraining orchestrator route.
- `frontend/src/pages/ComponentGalleryPage.tsx`: Internal design system component showcase (`/_components`).
- `frontend/src/pages/NotFoundPage.tsx`: 404 handler with return-to-overview action.
- `frontend/src/__tests__/Button.test.tsx`: Unit tests for Button variants, loading, and click events.
- `frontend/src/__tests__/DataTable.test.tsx`: Unit tests for DataTable rendering, sorting, and pagination.
- `frontend/src/__tests__/Term.test.tsx`: Unit tests for Term glossary popover and aria attributes.
- `scripts/capture_screenshots.py`: Headless browser script capturing responsive viewports.
- `docs/figures/ui_shell_1440.png`: Desktop responsive screenshot (1440x900).
- `docs/figures/ui_shell_1024.png`: Tablet responsive screenshot (1024x768).
- `docs/figures/ui_shell_390.png`: Mobile responsive screenshot (390x844).
- `docs/figures/ui_shell_components_1440.png`: Component showcase screenshot (1440x1100).
- `docs/15_frontend_architecture.md`: Comprehensive frontend architectural documentation.
- `docs/design_tokens.md`: Formal design tokens and accessibility reference table.

### Modified Files
- `src/mei/api/app.py`: Integrated SPA static mount (`/assets`) and buffered `HTMLResponse` fallback for production serving via `mei.cli serve`.
- `src/mei/api/routes/scoring.py`: Made `explain_dataframe` and `score_dataframe` imports lazy inside request handlers, reducing API cold-boot import latency by 58 seconds.
- `docs/DECISIONS.md`: Logged decisions D-053, D-054, and D-055.

---

## 3. Key Numbers Read Directly from Output Files

| Metric / Dimension | Value Read From Artifact | Provenance File |
| :--- | :--- | :--- |
| **Frontend Source Files Scanned** | `42 files` | `frontend/scripts/check-forbidden.cjs` |
| **Forbidden Words Violations** | `0 violations` | `frontend/scripts/check-forbidden.cjs` |
| **Forbidden Colors (Hue $245^\circ\text{–}335^\circ$)** | `0 violations` | `frontend/scripts/check-forbidden.cjs` |
| **Forbidden Glyphs / Emojis** | `0 violations` | `frontend/scripts/check-forbidden.cjs` |
| **Shared UI Components Built** | `18 components` | `frontend/src/components/ui/` |
| **Vitest Unit Tests** | `11 passed / 11 total (100%)` | `frontend/src/__tests__/` |
| **Vite Production Bundling Duration** | `63.0 seconds` | `frontend/dist/` build log |
| **Total Production JS Bundle Size** | `1,584.31 kB (514.78 kB gzipped)` | `frontend/dist/assets/index-*.js` |
| **Total Production CSS Bundle Size** | `38.30 kB (9.45 kB gzipped)` | `frontend/dist/assets/index-*.css` |
| **Backend Integration Pytest Suite** | `18 passed / 18 total (100%)` | `tests/test_api.py` |
| **Backend Smoke Test Suite** | `10 passed / 10 total (100%)` | `tests/test_smoke.py` |
| **API Cold Boot Import Latency** | `1.983 seconds` | `scratch/test_api_boot.py` |
| **Desktop Screenshot Size (1440x900)** | `76.2 KB` | `docs/figures/ui_shell_1440.png` |
| **Tablet Screenshot Size (1024x768)** | `74.7 KB` | `docs/figures/ui_shell_1024.png` |
| **Mobile Screenshot Size (390x844)** | `36.4 KB` | `docs/figures/ui_shell_390.png` |
| **Component Showcase Screenshot Size** | `148.6 KB` | `docs/figures/ui_shell_components_1440.png` |

---

## 4. Checks Executed with Explicit PASS/FAIL Status

1. **Forbidden Copy and Color CI Linter**:
   - `node frontend/scripts/check-forbidden.cjs`
   - **PASS**: 42 files scanned; zero violations for forbidden copy, colors, or emojis.

2. **Frontend Component Unit Tests (Vitest & RTL)**:
   - `node npm-cli.js test`
   - **PASS**: 3 test files (`Button.test.tsx`, `DataTable.test.tsx`, `Term.test.tsx`), 11 tests passed in 17.56s.

3. **TypeScript Compilation & Production Bundle**:
   - `tsc && vite build`
   - **PASS**: Compiled with zero TypeScript errors; bundled assets into `frontend/dist`.

4. **FastAPI Backend Integration Test Suite**:
   - `python -m pytest tests/test_api.py`
   - **PASS**: 18 passed in 46.90s.

5. **System Smoke Test Suite**:
   - `python -m pytest tests/test_smoke.py`
   - **PASS**: 10 passed in 131.36s.

6. **Python Code Formatting & Linting**:
   - `ruff check src/mei/api` and `ruff format --check src/mei/api`
   - **PASS**: Zero lint errors; imports properly ordered.

7. **Visual Design System Compliance & Screenshot Verification**:
   - `python scripts/capture_screenshots.py`
   - **PASS**: All 4 responsive viewports captured cleanly and visually verified:
     - 1440px desktop: 232px navigation rail, TopBar model/data badges, empty state card with zero shadows.
     - 1024px tablet: Proportionally responsive layout.
     - 390px mobile: Collapsible mobile menu button and responsive layout.
     - 1440px components showcase: Verified all 18 shared UI primitives at `/_components`.

---

## 5. Risks or Decisions Logged

- **D-053**: Custom palette isolation and forbidden token CI enforcement (`check-forbidden.cjs`). Eliminates startup SaaS tropes and preserves academic authority.
- **D-054**: Radix UI headless accessible primitives and local Fontsource typography bundling. Ensures WCAG 2.1 AA accessibility and full air-gapped intranet operation.
- **D-055**: Single-command production deployment via FastAPI static mount with buffered `HTMLResponse` SPA fallback.
- **Risk Logged**: In Windows PowerShell environments without network bypass permissions, internal IPC loopback sockets used by `anyio` in `TestClient` can block; resolved by documenting the `BypassSandbox: true` requirement for network-bound test runners and running local static files through buffered responses.

---

## 6. STOP Rule Enforcement

Phase 12 is fully complete. Per Global Rules G1–G13, work terminates immediately here. Phase 13 (Overview & Leaderboard Pages) will not be started early.
