import AxeBuilder from "@axe-core/playwright";
import { expect, type Page, test } from "@playwright/test";

async function executePaletteCommand(page: Page, query: string, name: RegExp) {
  await page.keyboard.press("Control+k");
  await page.getByRole("searchbox", { name: "Search commands" }).fill(query);
  await page.getByRole("dialog", { name: "Commands" }).getByRole("button", { name }).click();
}
test("projects and inspects the compiler-owned model", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("status")).toContainText(
    "browser Studio demonstrates interaction and layout",
  );
  await expect(page.getByRole("heading", { name: "Relation view" })).toBeVisible();
  await page.getByRole("button", { name: /decay Relation/ }).click();
  await expect(page.getByRole("heading", { name: "decay", exact: true })).toBeVisible();
});
test("does not offer retired native workflows", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("button", { name: /Sampled DC drive|CAD authoring/ })).toHaveCount(0);
});
test("selects exact CAD Domains", async ({ page }) => {
  await page.goto("/");
  await executePaletteCommand(page, "open CAD example", /Open CAD example/);
  await expect(page.getByRole("heading", { name: "Semantic geometry" })).toBeVisible();
});
test("has no serious or critical WCAG violations", async ({ page }) => {
  await page.goto("/");
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"])
    .analyze();
  expect(
    results.violations.filter((item) => item.impact === "serious" || item.impact === "critical"),
  ).toEqual([]);
});

test("edits Parameters without offering state initial values as revision values", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: /state Field/ }).click();
  await expect(page.locator("#inspector-value-input")).toHaveCount(0);
  await page.getByRole("button", { name: /rate Parameter/ }).click();
  await expect(page.locator("#inspector-value-input")).toBeVisible();
});
