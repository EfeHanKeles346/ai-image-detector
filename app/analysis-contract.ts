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
  decision: Decision | null;
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
    ...(value as Omit<Analysis, "tile_map" | "decision">),
    tile_map: parseTileMap(value.tile_map),
    decision: parseDecision(value.decision),
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
