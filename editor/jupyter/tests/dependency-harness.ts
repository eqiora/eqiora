// Load the actual locked consumers in a separate browser fixture.
import { Sanitizer } from '@jupyterlab/apputils/lib/sanitizer.js';
import { MermaidManager } from '@jupyterlab/mermaid/lib/manager.js';
import katex from 'katex';
import 'katex/dist/katex.min.css';
Object.assign(window, { dependencyProbe: { Sanitizer, MermaidManager, katex } });
