import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const consumerRequire = createRequire(require.resolve("micromatch"));
const braces = consumerRequire("braces");
const depthError = { name: "SyntaxError", message: /maximum depth of 100/ };

test("micromatch resolves the reviewed local braces fork", () => {
  assert.equal(consumerRequire("braces/package.json").name, "@pixelproof/braces");
});

test("ordinary braces, ranges and glob consumers retain their behavior", () => {
  assert.deepEqual(braces.expand("src/{app,lib}/file{1..3}.ts"), [
    "src/app/file1.ts", "src/app/file2.ts", "src/app/file3.ts",
    "src/lib/file1.ts", "src/lib/file2.ts", "src/lib/file3.ts",
  ]);
  assert.equal(braces.compile("{a,{b,c}}"), "(a|(b|c))");
  assert.equal(braces.stringify(braces.parse("{a,{b,c}}")), "{a,{b,c}}");
  assert.deepEqual(require("micromatch")(["a.ts", "b.js", "c.css"], "*.{ts,js}"), ["a.ts", "b.js"]);
  assert.deepEqual(braces.expand("\\{a,b\\}"), ["{a,b}"]);
  assert.doesNotThrow(() => braces.parse('"' + "{".repeat(300) + '"'));
  assert.doesNotThrow(() => braces.parse("\\{".repeat(300)));
});

for (const method of ["parse", "compile", "expand", "stringify", "create"]) {
  test(`${method} rejects deep braces, parentheses and unclosed patterns`, () => {
    for (const pattern of [
      "{".repeat(4000) + "a,b" + "}".repeat(4000),
      "(".repeat(4000) + "a" + ")".repeat(4000),
      "{".repeat(4000) + "a,b",
    ]) {
      assert.throws(() => braces[method](pattern), depthError);
    }
    assert.throws(() => braces[method]("{".repeat(200) + "a,b" + "}".repeat(200), { maxDepth: Infinity }), depthError);
  });
}

for (const method of ["compile", "expand", "stringify"]) {
  test(`${method} bounds recursion for caller-provided ASTs too`, () => {
    const ast = { type: "root", nodes: [] };
    let parent = ast;
    for (let i = 0; i < 200; i++) {
      const child = { type: "brace", nodes: [], commas: 1, open: true, close: true, parent };
      parent.nodes.push(child);
      parent = child;
    }
    assert.throws(() => braces[method](ast), depthError);
  });
}
