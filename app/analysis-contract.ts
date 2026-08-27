export type Verdict = "ai" | "real" | "uncertain";
export type Tile = { x: number; y: number; p_ai: number; texture: number };
export type TileMap = {
  p_ai: number;
  tiles: Tile[];
  tile_px: number;
  image_w: number;
  image_h: number;
};
export type DecisionArm = {
  label: string;
  score: number;
  threshold: number;
  band: "ai" | "insufficient";
};
export type Decision = {
  label: "ai" | "insufficient";
  triggered_by: string[];
  arms: Record<string, DecisionArm>;
  caveats: string[];
  bytes_per_pixel: number;
  provenance: string;
};
export type ProjectModelResult = {
  score: number;
  threshold: number;
  triggered: boolean;
  research_only: true;
  limitation: string;
  artifact_id: string;
  artifact_sha256: string;
  revision: string;
  seed: number;
  aggregation: string;
  tile_px: number;
  tile_count: number;
};
export type R1bResearchResult = {
  id: string;
  label: string;
  score: number;
  threshold: number;
  triggered: boolean;
  band: "ai_signal" | "insufficient_evidence";
  research_only: true;
  affects_decision: false;
  artifact_sha256: string;
  limitation: string;
  evaluation: {
    ipn_worst_device_fp: number;
    owner_gallery_fp: number;
  };
};
export type Analysis = {
  p_ai: number;
  verdict: Verdict;
  method: string;
  method_label: string;
  auto_selected: boolean;
  engine: string;
  resolution: string;
  enough_evidence: boolean;
  tile_map: TileMap | null;
  project_model: ProjectModelResult | null;
  decision: Decision | null;
  r1b_research: R1bResearchResult | null;
};

export class AnalysisResponseError extends Error {
  constructor(message = "Analiz servisi beklenmeyen bir yanıt döndürdü.") {
    super(message);
    this.name = "AnalysisResponseError";
  }
}

export class AnalysisHttpError extends Error {
  readonly status: number;
  readonly detail?: string;

  constructor(status: number, detail?: string) {
    super(detail || `HTTP ${status}`);
    this.name = "AnalysisHttpError";
    this.status = status;
    this.detail = detail;
  }
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function stringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function parseDecisionArm(value: unknown): DecisionArm {
  if (
    !record(value) ||
    typeof value.label !== "string" ||
    !finite(value.score) ||
    !finite(value.threshold) ||
    (value.band !== "ai" && value.band !== "insufficient")
  ) {
    throw new AnalysisResponseError();
  }
  return value as DecisionArm;
}

function parseDecision(value: unknown): Decision | null {
  if (value === null) return null;
  if (
    !record(value) ||
    (value.label !== "ai" && value.label !== "insufficient") ||
    !stringArray(value.triggered_by) ||
    !record(value.arms) ||
    !stringArray(value.caveats) ||
    !finite(value.bytes_per_pixel) ||
    typeof value.provenance !== "string"
  ) {
    throw new AnalysisResponseError();
  }
  for (const arm of Object.values(value.arms)) parseDecisionArm(arm);
  return value as Decision;
}

function parseTileMap(value: unknown): TileMap | null {
  if (value === null) return null;
  if (
    !record(value) ||
    !finite(value.p_ai) ||
    !finite(value.tile_px) ||
    !finite(value.image_w) ||
    !finite(value.image_h) ||
    value.tile_px <= 0 ||
    value.image_w <= 0 ||
    value.image_h <= 0 ||
    !Array.isArray(value.tiles)
  ) {
    throw new AnalysisResponseError();
  }
  for (const tile of value.tiles) {
    if (
      !record(tile) ||
      !finite(tile.x) ||
      !finite(tile.y) ||
      !finite(tile.p_ai) ||
      !finite(tile.texture)
    ) {
      throw new AnalysisResponseError();
    }
  }
  return value as TileMap;
}

function parseProjectModel(value: unknown): ProjectModelResult | null {
  if (value === null) return null;
  if (
    !record(value) ||
    !finite(value.score) ||
    value.score < 0 ||
    value.score > 1 ||
    !finite(value.threshold) ||
    value.threshold < 0 ||
    value.threshold > 1 ||
    typeof value.triggered !== "boolean" ||
    value.research_only !== true ||
    typeof value.limitation !== "string" ||
    typeof value.artifact_id !== "string" ||
    typeof value.artifact_sha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(value.artifact_sha256) ||
    typeof value.revision !== "string" ||
    typeof value.seed !== "number" ||
    !Number.isInteger(value.seed) ||
    typeof value.aggregation !== "string" ||
    typeof value.tile_px !== "number" ||
    !Number.isInteger(value.tile_px) ||
    value.tile_px <= 0 ||
    typeof value.tile_count !== "number" ||
    !Number.isInteger(value.tile_count) ||
    value.tile_count <= 0
  ) {
    throw new AnalysisResponseError();
  }
  return value as ProjectModelResult;
}

function parseR1bResearch(value: unknown): R1bResearchResult | null {
  if (value === null) return null;
  if (
    !record(value) ||
    typeof value.id !== "string" ||
    typeof value.label !== "string" ||
    !finite(value.score) || value.score < 0 || value.score > 1 ||
    !finite(value.threshold) || value.threshold < 0 || value.threshold > 1 ||
    typeof value.triggered !== "boolean" ||
    (value.band !== "ai_signal" && value.band !== "insufficient_evidence") ||
    value.research_only !== true ||
    value.affects_decision !== false ||
    typeof value.artifact_sha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(value.artifact_sha256) ||
    typeof value.limitation !== "string" ||
    !record(value.evaluation) ||
    !finite(value.evaluation.ipn_worst_device_fp) ||
    !finite(value.evaluation.owner_gallery_fp) ||
    value.evaluation.ipn_worst_device_fp < 0 || value.evaluation.ipn_worst_device_fp > 1 ||
    value.evaluation.owner_gallery_fp < 0 || value.evaluation.owner_gallery_fp > 1
  ) {
    throw new AnalysisResponseError();
  }
  return value as R1bResearchResult;
}

export function parseAnalysis(value: unknown): Analysis {
  if (
    !record(value) ||
    !finite(value.p_ai) ||
    value.p_ai < 0 ||
    value.p_ai > 1 ||
    (value.verdict !== "ai" && value.verdict !== "real" && value.verdict !== "uncertain") ||
    typeof value.method !== "string" ||
    typeof value.method_label !== "string" ||
    typeof value.auto_selected !== "boolean" ||
    typeof value.engine !== "string" ||
    typeof value.resolution !== "string" ||
    typeof value.enough_evidence !== "boolean"
  ) {
    throw new AnalysisResponseError();
  }

  return {
    ...(value as Omit<Analysis, "tile_map" | "project_model" | "decision">),
    tile_map: parseTileMap(value.tile_map),
    project_model: parseProjectModel(value.project_model ?? null),
    decision: parseDecision(value.decision),
    r1b_research: parseR1bResearch(value.r1b_research ?? null),
  };
}

export function resolveApiOrigin(configured: string | undefined, development: boolean): string {
  const candidate = configured?.trim();
  if (!candidate) return development ? "http://127.0.0.1:8799" : "";

  let parsed: URL;
  try {
    parsed = new URL(candidate);
  } catch {
    throw new Error("NEXT_PUBLIC_PIXELPROOF_API_URL mutlak bir HTTP(S) adresi olmalıdır.");
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("NEXT_PUBLIC_PIXELPROOF_API_URL yalnız HTTP(S) kullanabilir.");
  }
  return parsed.href.replace(/\/$/, "");
}

export function analysisEndpoint(origin: string): string {
  return `${origin}/predict`;
}

export function analysisErrorMessage(error: unknown): string {
  if (error instanceof AnalysisResponseError) return error.message;
  if (error instanceof AnalysisHttpError) {
    if (error.status === 413) return "Görsel servis sınırından büyük. Daha küçük bir dosya seçin.";
    if (error.status === 415) return "Bu görsel biçimi desteklenmiyor. JPG, PNG veya WEBP kullanın.";
    if (error.status === 422 || error.status === 400) {
      return error.detail || "Görsel doğrulanamadı. Başka bir dosya deneyin.";
    }
    if (error.status === 429) return "Çok fazla istek gönderildi. Biraz bekleyip yeniden deneyin.";
    if (error.status === 503) return "Model servisi henüz hazır değil. Biraz sonra yeniden deneyin.";
    if (error.status === 404) {
      return "Analiz uç noktası bulunamadı. Dağıtımda API adresi yapılandırılmamış olabilir.";
    }
    if (error.status >= 500) return "Model çıkarımı başarısız oldu. Servis kayıtlarını kontrol edin.";
    return error.detail || `Analiz isteği reddedildi (HTTP ${error.status}).`;
  }
  return "Analiz servisine ulaşılamadı. Ağ bağlantısını ve servis adresini kontrol edin.";
}

export type RequestTicket = { id: number; signal: AbortSignal };

export class LatestRequestGate {
  private id = 0;
  private controller: AbortController | null = null;

  begin(): RequestTicket {
    this.cancel();
    this.controller = new AbortController();
    return { id: this.id, signal: this.controller.signal };
  }

  isCurrent(id: number): boolean {
    return id === this.id;
  }

  cancel(): void {
    this.id += 1;
    this.controller?.abort();
    this.controller = null;
  }
}
