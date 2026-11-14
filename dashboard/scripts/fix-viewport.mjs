// Runs after `next build`. The site always shows the desktop layout, so phones should fit the 1280 px page to
// their screen width. Next.js always adds initial-scale=1 to the viewport tag, which would open the page zoomed in
// on a phone, so the built HTML files are patched to a plain `width=1280` (browsers then pick the fitting zoom).
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const root = new URL("../dist", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
let patched = 0;

function walk(dir) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p);
    else if (name.endsWith(".html")) {
      const html = readFileSync(p, "utf8");
      const out = html.replace(/<meta name="viewport" content="[^"]*"\s*\/?>/, '<meta name="viewport" content="width=1280"/>');
      if (out !== html) { writeFileSync(p, out); patched++; }
    }
  }
}

walk(root);
console.log(`viewport set to width=1280 in ${patched} page(s)`);
