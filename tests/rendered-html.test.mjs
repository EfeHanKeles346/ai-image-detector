import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render(pathname = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${pathname}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`http://localhost${pathname}`, {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the PixelProof product shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<html lang="tr">/i);
  assert.match(html, /<title>PixelProof — AI Görsel Kanıt Laboratuvarı<\/title>/i);
  assert.match(html, /Bir görsel yükle, modellerin ne gördüğünü karşılaştır/);
  assert.match(html, /Fotoğrafı buraya bırakın/);
  assert.match(html, /E20 ResNet-18/);
  assert.match(html, /Görseli analiz et/);
  assert.match(html, /aria-label="Analiz edilecek görseli seç"/);
  assert.match(html, /aria-label="Çalışacak model zinciri"/);
  assert.match(html, /gerçeklik sertifikası değildir/i);
  assert.match(html, /Sonucu nasıl okuyacaksın/);
  assert.match(html, /R1b en yeni fakat/);

  assert.doesNotMatch(html, /Your site is taking shape|Building your site/);
  assert.doesNotMatch(html, /name=["']codex-preview["']/i);
  assert.doesNotMatch(html, /\/Users\/|file:\/\//i);
  assert.doesNotMatch(html, /p\(AI\)|AI olasılığı/i);
});

test("packages the Sites hosting contract without stale starter assets", async () => {
  const [sourceConfig, packagedConfig, headers, packageJson] = await Promise.all([
    readFile(new URL("../.openai/hosting.json", import.meta.url), "utf8"),
    readFile(new URL("../dist/.openai/hosting.json", import.meta.url), "utf8"),
    readFile(new URL("../dist/client/_headers", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.deepEqual(JSON.parse(packagedConfig), JSON.parse(sourceConfig));
  assert.match(headers, /\/assets\/\*/);
  assert.match(headers, /immutable/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});
