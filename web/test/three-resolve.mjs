/**
 * The browser resolves bare "three" through the import map in index.html,
 * which points at a CDN. Node has no import map, and the copy of three
 * installed for the tests lives in test/node_modules, which is not on the
 * resolution path from web/public/js. Rather than add a package.json above
 * the app source (Vercel would read it) or vendor a second copy, this hook
 * maps the bare specifier to the test's own three.
 *
 * Registered by test/mesh.mjs via module.register before it imports any
 * app code.
 */
import { createRequire } from "module";
import { pathToFileURL } from "url";
import { dirname } from "path";

const require = createRequire(import.meta.url);
// the ES build, not the CommonJS one node's resolver reaches first: the app
// does `import * as THREE`, and the CJS namespace hides the classes behind
// a default export
const root = dirname(dirname(require.resolve("three")));
const esm = pathToFileURL(`${root}/build/three.module.js`).href;

export function resolve(specifier, context, next) {
  if (specifier === "three") return { url: esm, shortCircuit: true };
  return next(specifier, context);
}
