import { copyFile } from 'node:fs/promises';
await copyFile(new URL('../eqiora/syntaxes/eqiora.tmLanguage.json', import.meta.url), new URL('src/grammar.json', import.meta.url));
