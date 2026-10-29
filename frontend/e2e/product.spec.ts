import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';

const pages = [
  { path: '/', heading: 'Engagement Index', unavailableHeading: 'Project results are unavailable' },
  { path: '/leaderboard', heading: 'Model leaderboard' },
  { path: '/patterns', heading: 'Study Patterns' },
  { path: '/drivers', heading: 'Risk Drivers' },
  { path: '/score', heading: 'Score students' },
  { path: '/train', heading: 'Train on New Data' },
];

async function forceUnavailableApi(page: Page) {
  await page.route('**/api/**', (route) =>
    route.fulfill({
      status: 503,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'The service is temporarily unavailable.' }),
    }),
  );
}

for (const pageInfo of pages) {
  test(`${pageInfo.path} stays within desktop, tablet, and mobile widths`, async ({ page }) => {
    await forceUnavailableApi(page);
    const heading = pageInfo.unavailableHeading ?? pageInfo.heading;

    for (const viewport of [
      { width: 1440, height: 900 },
      { width: 1024, height: 768 },
      { width: 390, height: 844 },
    ]) {
      await page.setViewportSize(viewport);
      await page.goto(pageInfo.path);
      await expect(
        page.getByRole('heading', { name: heading, exact: false }).first(),
      ).toBeVisible();

      const dimensions = await page.evaluate(() => ({
        viewport: document.documentElement.clientWidth,
        scroll: document.documentElement.scrollWidth,
      }));
      expect(dimensions.scroll).toBeLessThanOrEqual(dimensions.viewport + 1);
    }
  });
}

test('core pages have no serious or critical axe violations', async ({ page }) => {
  await forceUnavailableApi(page);
  test.setTimeout(90_000);

  for (const pageInfo of pages) {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto(pageInfo.path);
    const heading = pageInfo.unavailableHeading ?? pageInfo.heading;
    await expect(
      page.getByRole('heading', { name: heading, exact: false }).first(),
    ).toBeVisible();

    const results = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .analyze();
    const seriousViolations = results.violations.filter(
      (violation) => violation.impact === 'serious' || violation.impact === 'critical',
    );
    expect(seriousViolations, `${pageInfo.path} axe violations`).toEqual([]);
  }
});
