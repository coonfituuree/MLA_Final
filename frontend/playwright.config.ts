import { defineConfig } from '@playwright/test'
export default defineConfig({
  testDir: './e2e', workers: 1, reporter: [['list'], ['json', { outputFile: '../reports/browser-results.json' }]],
  use: { baseURL: process.env.PLAYWRIGHT_BASE_URL ?? 'http://localhost:5173', launchOptions: { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE, args: ['--no-sandbox'] } },
})
