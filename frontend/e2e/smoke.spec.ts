import { test, expect } from '@playwright/test';

test.describe('Counselor Dashboard E2E Smoke Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Intercept API calls to provide robust, hermetic end-to-end testing
    await page.route('**/api/meta', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          model_version: '1.0.0',
          data_type: 'SYNTHETIC',
          banner_text: 'SYNTHETIC DATA - pipeline validation only',
          training_date: '2026-10-24',
          total_students_evaluated: 2000,
          primary_target: 'Institutional Attrition',
        }),
      });
    });

    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'healthy', service: 'ERI Research API' }),
      });
    });

    await page.route('**/api/leaderboard**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          track: 'C',
          total_models: 1,
          selected_model: 'logistic_l2',
          selection_decision: { reason: 'Parsimony rule applied: simpler linear model preferred.' },
          category_winners: {
            'Linear/Shallow': { model: 'logistic_l2', composite_score: 0.521 },
          },
          leaderboard: [
            {
              model_name: 'logistic_l2',
              category: 'Linear/Shallow',
              rank: 1,
              composite_score: 0.521,
              pr_auc: 0.339,
              roc_auc: 0.635,
              brier_score: 0.224,
              overfit_gap: 0.103,
              is_selected: true,
            },
          ],
        }),
      });
    });

    await page.route('**/api/causal/summary', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          metadata: {
            methodology: 'Semi-parametric AIPW and sIPW with Rosenbaum E-values',
            sample_size: 2000,
            gate_status: 'PASSED',
          },
          exposures: {
            financial_hardship: {
              exposure: 'financial_hardship',
              label: 'High Financial Distress',
              sipw_rd: 0.145,
              aipw_rd: 0.145,
              ci_lower: 0.088,
              ci_upper: 0.201,
              e_value: 3.055,
              refutations_passed: true,
            },
          },
        }),
      });
    });

    await page.route('**/api/causal/diagnostics', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          balance_smd: [],
          exposure_diagnostics: {},
        }),
      });
    });

    await page.route('**/api/causal/heterogeneity', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          gate_status: 'PASSED',
          sample_size: 2000,
          events_count: 351,
          conditional_ate: {},
        }),
      });
    });

    await page.route('**/api/dag', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          nodes: [],
          edges: [],
          exposures: ['financial_hardship'],
          mediators: ['later_gpa_decline'],
          confounders: ['household_income_bracket'],
          primary_outcome: 'attrition',
          image_url: '/dag/image',
        }),
      });
    });
    await page.route('**/dag/image', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'image/png',
        body: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', 'base64'),
      });
    });
  });

  test('renders synthetic warning banner prominently on all views', async ({ page }) => {
    await page.goto('/');
    const banner = page.locator('text=SYNTHETIC DATA - pipeline validation only');
    await expect(banner).toBeVisible({ timeout: 10000 });
  });

  test('navigates to Leaderboard and verifies model selection podium', async ({ page }) => {
    await page.goto('/');
    const leaderboardBtn = page.getByRole('button', { name: 'Leaderboard', exact: true });
    await expect(leaderboardBtn).toBeVisible({ timeout: 10000 });
    await leaderboardBtn.click();
    await expect(page.locator('text=Model Zoo Benchmark Leaderboard')).toBeVisible({ timeout: 10000 });
  });

  test('navigates to Causal Analysis and verifies non-predictive causal isolation', async ({ page }) => {
    await page.goto('/');
    const causalBtn = page.getByRole('button', { name: 'Causal Inference', exact: true });
    await expect(causalBtn).toBeVisible({ timeout: 10000 });
    await causalBtn.click();
    await expect(page.locator('text=Applied Causal Inference')).toBeVisible({ timeout: 10000 });
  });

  test('navigates to Batch Inference and verifies survey upload zone', async ({ page }) => {
    await page.goto('/');
    const predictBtn = page.getByRole('button', { name: 'Upload & Score', exact: true });
    await expect(predictBtn).toBeVisible({ timeout: 10000 });
    await predictBtn.click();
    await expect(page.locator('text=Upload Survey Cohort & Automated Risk Scoring')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=Zero PII Guarantee')).toBeVisible({ timeout: 10000 });
  });
});

