# Local braces depth guard

This directory vendors `braces` 3.0.3 (MIT, original LICENSE and README retained)
Registry provenance: tarball SHA-1 `490332f40919452272d55a8480adc0c441358789`.
The unchanged entry point, helpers, README and LICENSE were compared with that tarball.

It is installed as the private package `@pixelproof/braces` 3.0.3-pixelproof.1. The root npm override
routes every transitive `braces` consumer to this directory.

On 4 October 2026, GHSA-vfj7-8cjw-p6xm had no patched upstream release:
https://github.com/advisories/GHSA-vfj7-8cjw-p6xm
The upstream report recommends bounding parser nesting:
https://github.com/micromatch/braces/issues/70

Changes from upstream are limited to package metadata, `lib/depth.js`, parser
stack checks before both brace and parenthesis pushes, and depth counters in
compile, expand and stringify. The latter also protect callers that provide an
AST directly. Nesting greater than 100 is rejected with a SyntaxError, consistent
with the existing input-length validation. Options cannot disable the bound.
Literal escaped/quoted braces do not increase parser nesting. Applications must
still handle invalid-pattern errors; this is not a sandbox for arbitrary ASTs or
a new bound on combinatorial expansion output.

The recursive stringify call retains upstream's empty parent argument. No other
matching or expansion semantics are intentionally changed. `tests/braces-depth.test.mjs`
covers ordinary patterns, escaped input, ranges, malformed/deep patterns, direct
ASTs and the actual micromatch dependency resolution. The unpatched parser accepts
the deep fixtures, so the rejection tests distinguish the mitigation from upstream.

The npm audit service does not certify local fork code. A zero-advisory result
must therefore be read together with review of this patch and its regression
tests. No advisory allowlist or CI audit bypass is used. Replace this fork and
remove the override after a reviewed upstream fix passes the regression suite.
