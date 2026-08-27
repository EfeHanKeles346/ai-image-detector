import assert from "node:assert/strict";
import test from "node:test";
import {
  AnalysisHttpError,
  AnalysisResponseError,
  LatestRequestGate,
  analysisEndpoint,
  analysisErrorMessage,
  parseAnalysis,
  resolveApiOrigin,
} from "../app/analysis-contract.ts";

const validAnalysis = {
  p_ai: 0.72,
  verdict: "ai",
  method: "auto",
  method_label: "Otomatik",
  auto_selected: true,
  engine: "test",
  resolution: "1024×768",
  enough_evidence: true,
  tile_map: null,
  project_model: {
    score: 0.72,
    threshold: 0.9895,
    triggered: false,
    research_only: true,
    limitation: "E20 worst-source limitation",
    artifact_id: "e20-tile-resnet18-seed2024",
    artifact_sha256: "a".repeat(64),
    revision: "E20-v2 seed 2024",
    seed: 2024,
    aggregation: "top3",
    tile_px: 128,
    tile_count: 51,
  },
  decision: {
    label: "ai",
    triggered_by: ["cf_vit"],
    arms: {
      cf_vit: { label: "CF ViT-S", score: 2.1, threshold: 1.5, band: "ai" },
    },
    caveats: [],
    bytes_per_pixel: 0.8,
    provenance: "test fixture",
  },
  r1b_research: {
    id: "e32_r1b_cfvit_iphone_correction",
    label: "E32 R1b · CF-ViT",
    score: 0.82,
    threshold: 0.1259,
    triggered: true,
    band: "ai_signal",
    research_only: true,
    affects_decision: false,
    artifact_sha256: "b".repeat(64),
    limitation: "Measured external real false positives",
    evaluation: { ipn_worst_device_fp: 0.4, owner_gallery_fp: 144 / 210 },
  },
};

test("API origin is explicit in deployments and deliberately local in development", () => {
  assert.equal(resolveApiOrigin(undefined, true), "http://127.0.0.1:8799");
  assert.equal(resolveApiOrigin(undefined, false), "");
  assert.equal(resolveApiOrigin(" https://api.example.test/ ", false), "https://api.example.test");
  assert.equal(analysisEndpoint(""), "/predict");
  assert.equal(analysisEndpoint("https://api.example.test"), "https://api.example.test/predict");
  assert.throws(() => resolveApiOrigin("file:///tmp/socket", false), /HTTP\(S\)/);
  assert.throws(() => resolveApiOrigin("api.example.test", false), /mutlak/);
});

test("analysis responses are validated before UI state receives them", () => {
  assert.deepEqual(parseAnalysis(validAnalysis), validAnalysis);
  assert.throws(() => parseAnalysis({ ...validAnalysis, p_ai: 4 }), AnalysisResponseError);
  assert.throws(
    () => parseAnalysis({ ...validAnalysis, decision: { ...validAnalysis.decision, arms: [] } }),
    AnalysisResponseError,
  );
  assert.throws(
    () => parseAnalysis({ ...validAnalysis, tile_map: { image_w: 0, tiles: [] } }),
    AnalysisResponseError,
  );
  assert.throws(
    () => parseAnalysis({
      ...validAnalysis,
      project_model: { ...validAnalysis.project_model, artifact_sha256: "not-a-hash" },
    }),
    AnalysisResponseError,
  );
  assert.throws(
    () => parseAnalysis({
      ...validAnalysis,
      r1b_research: { ...validAnalysis.r1b_research, affects_decision: true },
    }),
    AnalysisResponseError,
  );
});

test("only the newest request may publish and clearing invalidates it", () => {
  const gate = new LatestRequestGate();
  const first = gate.begin();
  assert.equal(gate.isCurrent(first.id), true);

  const second = gate.begin();
  assert.equal(first.signal.aborted, true);
  assert.equal(gate.isCurrent(first.id), false);
  assert.equal(gate.isCurrent(second.id), true);

  gate.cancel();
  assert.equal(second.signal.aborted, true);
  assert.equal(gate.isCurrent(second.id), false);
});

test("HTTP, malformed-response and network failures have distinct messages", () => {
  assert.match(analysisErrorMessage(new AnalysisHttpError(413)), /büyük/);
  assert.match(analysisErrorMessage(new AnalysisHttpError(415)), /desteklenmiyor/);
  assert.match(analysisErrorMessage(new AnalysisHttpError(422, "Piksel sınırı aşıldı")), /Piksel/);
  assert.match(analysisErrorMessage(new AnalysisHttpError(503)), /hazır değil/);
  assert.match(analysisErrorMessage(new AnalysisResponseError()), /beklenmeyen/);
  assert.match(analysisErrorMessage(new TypeError("fetch failed")), /ulaşılamadı/);
});
