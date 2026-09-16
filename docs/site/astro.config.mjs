import { existsSync } from 'node:fs';
import { stat, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { satteri } from '@astrojs/markdown-satteri';
import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';

import { katexMathPlugin } from './src/plugins/katex.ts';
import { canonicalGuideLinks } from './src/plugins/canonical-guide-links.ts';

const SOURCE_SHA = /^[0-9a-f]{40}$/;

function outputDirectory() {
  const configured = process.env.EQIORA_SITE_ASTRO_OUT_DIR;
  if (configured) {
    return resolve(configured);
  }
  return fileURLToPath(new URL('../../build/site-astro/', import.meta.url));
}

const robots = {
  name: 'eqiora-static-robots',
  hooks: {
    'astro:build:done': async ({ dir }) => {
      const output = fileURLToPath(dir);
      const sitemap = resolve(output, 'sitemap-index.xml');
      if (!(await stat(sitemap)).isFile()) {
        throw new Error('Starlight sitemap-index.xml is absent from the static output');
      }
      await writeFile(
        resolve(output, 'robots.txt'),
        'User-agent: *\nAllow: /\nSitemap: https://eqiora.org/sitemap-index.xml\n',
        { encoding: 'utf8', flag: 'wx', mode: 0o644 },
      );
    },
  },
};

if (process.env.EQIORA_SITE_BUILD_PROFILE === 'complete') {
  if (!SOURCE_SHA.test(process.env.EQIORA_SITE_SOURCE_SHA ?? '')) {
    throw new Error('EQIORA_SITE_SOURCE_SHA must be the exact 40-character lowercase source commit');
  }

  const requiredSuccessorInputs = [
    'src/assets/brand/eqiora-mark.svg',
    'src/assets/gallery/exact-cylinder-pressure-presentation.png',
    'src/assets/gallery/exact-cylinder-pressure-thumbnail.png',
    'src/assets/gallery/mixed-boundary-elasticity-displacement.png',
    'src/assets/gallery/karman-vortex-street-poster.png',
    'src/assets/gallery/karman-vortex-street-reduced-motion.png',
    'src/assets/gallery/karman-vortex-street.webm',
    'src/assets/gallery/karman-vortex-street.mp4',
    'src/components/site/ExactSourceLink.astro',
    'src/components/site/Header.astro',
    'src/components/site/Footer.astro',
    'src/components/site/ReleaseIdentity.astro',
    'src/content/docs/index.mdx',
    'src/content/docs/evidence/index.mdx',
    'src/content/docs/gallery/index.mdx',
    'src/content/docs/gallery/exact-cylinder-steady-stokes.mdx',
    'src/content/docs/gallery/mixed-boundary-elasticity.mdx',
    'src/content/docs/gallery/karman-vortex-street.mdx',
    'src/content/docs/reference/index.mdx',
    'src/content/docs/reference/python/index.mdx',
    'src/content/docs/reference/rust/index.mdx',
    'src/content/docs/reference/cli/index.mdx',
    'src/content/docs/learn/index.mdx',
    'src/content/docs/learn/mathematical-modeling/index.mdx',
    'src/content/docs/learn/mathematical-modeling/algebraic-relations-networks.mdx',
    'src/content/docs/learn/mathematical-modeling/boundary-interface-conditions.mdx',
    'src/content/docs/learn/mathematical-modeling/conservation-laws.mdx',
    'src/content/docs/learn/mathematical-modeling/constitutive-laws.mdx',
    'src/content/docs/learn/mathematical-modeling/fields-spatial-domains.mdx',
    'src/content/docs/learn/mathematical-modeling/models-not-simulations.mdx',
    'src/content/docs/learn/mathematical-modeling/ordinary-differential-equations.mdx',
    'src/content/docs/learn/mathematical-modeling/quantities-dimensions-units.mdx',
    'src/content/docs/guides/index.mdx',
    'src/content/docs/guides/run-and-inspect.mdx',
    'src/content/docs/guides/how-eqiora-fits-together.mdx',
    'src/content/docs/guides/modeling.mdx',
    'src/content/docs/guides/execution-and-arrays.mdx',
    'src/content/docs/guides/differentiation.mdx',
    'src/styles/site/tokens.css',
    'src/styles/site/layout.css',
    'src/styles/site/components.css',
    'public/favicon.svg',
    'public/apple-touch-icon.png',
    'public/social-card.svg',
  ];
  const missing = requiredSuccessorInputs.filter(
    (relative) => !existsSync(fileURLToPath(new URL(relative, import.meta.url))),
  );
  if (missing.length > 0) {
    throw new Error(`successor-site dependencies are absent: ${missing.join(', ')}`);
  }
}

export default defineConfig({
  site: 'https://eqiora.org/',
  base: '/',
  output: 'static',
  outDir: outputDirectory(),
  trailingSlash: 'always',
  markdown: {
    processor: satteri({
      features: { math: true },
      mdastPlugins: [katexMathPlugin, canonicalGuideLinks],
    }),
  },
  integrations: [
    starlight({
      title: 'Eqiora',
      routeMiddleware: './src/route-data.ts',
      editLink: { baseUrl: 'https://github.com/nkiyohara/eqiora/edit/main/' },
      description: 'Meaning-first scientific modeling and execution.',
      logo: {
        src: './src/assets/brand/eqiora-mark.svg',
        alt: '',
        replacesTitle: false,
      },
      favicon: '/favicon.svg',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/nkiyohara/eqiora',
        },
      ],
      sidebar: [
        { label: 'Learn', items: [
          { label: 'Browse topics', link: '/learn/' },
          { label: 'Mathematical modeling', collapsed: true, items: [{ autogenerate: { directory: 'learn/mathematical-modeling' } }] },
          { label: 'Heat transfer', collapsed: true, items: [{ autogenerate: { directory: 'learn/heat-transfer' } }] },
          { label: 'Numerical simulation', collapsed: true, items: [{ autogenerate: { directory: 'learn/numerical-simulation' } }] },
          { label: 'Circuits and dynamics', collapsed: true, items: [{ autogenerate: { directory: 'learn/circuits-dynamics' } }] },
          { label: 'Structural mechanics', collapsed: true, items: [{ autogenerate: { directory: 'learn/structural-mechanics' } }] },
          { label: 'Fluid mechanics', collapsed: true, items: [{ autogenerate: { directory: 'learn/fluid-mechanics' } }] },
          { label: 'Inverse problems', collapsed: true, items: [{ autogenerate: { directory: 'learn/inverse-problems' } }] },
        ] },
        { label: 'Guides', items: [
          { label: 'Choose a task', link: '/guides/' },
          { label: 'Run and inspect', link: '/guides/run-and-inspect/' },
          { label: 'How Eqiora fits together', link: '/guides/how-eqiora-fits-together/' },
          { label: 'Create and share a library', link: '/guides/creating-libraries/' },
          { label: 'Modeling and realization', link: '/guides/modeling/' },
          { label: 'Execution, diagnostics, arrays', link: '/guides/execution-and-arrays/' },
          { label: 'Differentiation', link: '/guides/differentiation/' },
        ] },
        { label: 'Gallery', items: [
          { label: 'Investigations', link: '/gallery/' },
          { label: 'Steady cylinder flow', link: '/gallery/exact-cylinder-steady-stokes/' },
          { label: 'Mixed-boundary elasticity', link: '/gallery/mixed-boundary-elasticity/' },
          { label: 'Kármán vortex street', link: '/gallery/karman-vortex-street/' },
        ] },
        { label: 'Reference', items: [
        { label: 'Find a definition', link: '/reference/' },
        {
          label: 'Language',
          items: [
            { label: 'Overview', link: '/reference/language/' },
            { label: 'Declarations', link: '/reference/language/declarations/' },
            { label: 'Quantities and units', link: '/reference/language/units/' },
            { label: 'Equations and state', link: '/reference/language/equations/' },
            { label: 'Component composition', link: '/reference/language/composition/' },
          ],
        },
        {
          label: 'Standard library',
          items: [
            { label: 'Overview', link: '/reference/standard-packages/' },
            { label: 'Electrical components', link: '/reference/standard-packages/electrical/' },
            { label: 'Sampled controls', link: '/reference/standard-packages/controls/' },
            { label: 'Continuum laws and boundaries', link: '/reference/standard-packages/continuum/' },
          ],
        },
        { label: 'Python API', link: '/reference/python/' },
        { label: 'Rust API', link: '/reference/rust/' },
        { label: 'CLI', link: '/reference/cli/' },
        ] },
        { label: 'Contributing', items: [
          { label: 'Contribute a change', link: '/contributing/' },
          { label: 'Architecture', link: '/contributing/architecture/' },
        ] },
      ],
      head: [
        { tag: 'meta', attrs: { property: 'og:type', content: 'website' } },
        {
          tag: 'meta',
          attrs: { property: 'og:image', content: 'https://eqiora.org/social-card.svg' },
        },
        {
          tag: 'meta',
          attrs: { name: 'twitter:card', content: 'summary_large_image' },
        },
        {
          tag: 'meta',
          attrs: { name: 'twitter:image', content: 'https://eqiora.org/social-card.svg' },
        },
        { tag: 'link', attrs: { rel: 'apple-touch-icon', href: '/apple-touch-icon.png' } },
      ],
      customCss: [
        'katex/dist/katex.min.css',
        '/src/styles/site/tokens.css',
        '/src/styles/site/layout.css',
        '/src/styles/site/components.css',
      ],
      components: {
        PageTitle: './src/components/site/PageTitle.astro',
        Header: './src/components/site/Header.astro',
        Footer: './src/components/site/Footer.astro',
        Search: './src/components/site/Search.astro',
        ThemeSelect: './src/components/site/ThemeSelect.astro',
      },
      pagefind: true,
      lastUpdated: false,
    }),
    robots,
  ],
  vite: {
    build: {
      sourcemap: false,
    },
  },
});
