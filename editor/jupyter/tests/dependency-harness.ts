// Import the actual locked consumers through Vite, not replacement renderers.
export { Sanitizer } from '@jupyterlab/apputils/lib/sanitizer.js';
export { MermaidManager } from '@jupyterlab/mermaid/lib/manager.js';
export { default as katex } from 'katex';
import 'katex/dist/katex.min.css';
