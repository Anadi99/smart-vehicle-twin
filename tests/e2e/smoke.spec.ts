import { expect, test } from '@playwright/test'

test('dashboard loads and shows Overview nav', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('Overview').first()).toBeVisible()
})

test('sidebar navigation links are all present', async ({ page }) => {
  await page.goto('/')
  for (const label of ['Overview', 'Telemetry', 'Predictions', '3D Twin', 'Diagnostics']) {
    await expect(page.getByRole('link', { name: label })).toBeVisible()
  }
})

test('Overview page shows KPI cards', async ({ page }) => {
  await page.goto('/overview')
  await expect(page.getByText(/speed/i).first()).toBeVisible()
  await expect(page.getByText(/battery soc/i).first()).toBeVisible()
})

test('Telemetry page renders charts section', async ({ page }) => {
  await page.goto('/telemetry')
  await expect(page.getByText(/telemetry/i).first()).toBeVisible()
})

test('Predictions page has Run inference button', async ({ page }) => {
  await page.goto('/predictions')
  await expect(page.getByRole('button', { name: /run inference/i })).toBeVisible()
})

test('3D Twin page renders', async ({ page }) => {
  await page.goto('/twin')
  await expect(page.getByText(/3D Twin/i).first()).toBeVisible()
})

test('Diagnostics page shows health section', async ({ page }) => {
  await page.goto('/diagnostics')
  await expect(page.getByText(/diagnostics/i).first()).toBeVisible()
})
