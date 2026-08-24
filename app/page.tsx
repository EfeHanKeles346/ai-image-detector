"use client";

import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import {
  AnalysisHttpError,
  AnalysisResponseError,
  LatestRequestGate,
  analysisEndpoint,
  analysisErrorMessage,
  parseAnalysis,
  resolveApiOrigin,
  type Analysis,
  type Decision,
  type Verdict,
} from "./analysis-contract";

type Preview = { name: string; url: string; size: string; file: File };

const CAVEAT_TEXT: Record<string, string> = {
  "megapiksel-siniri": "Etkin B-Free koluna 2048px sınırı uygulandı (E23b)",
  "sikistirilmis-girdi": "Düşük byte/piksel — yalnız kaba sıkıştırma uyarısı (E23c)",
};

const API_ORIGIN = resolveApiOrigin(
  process.env.NEXT_PUBLIC_PIXELPROOF_API_URL,
  process.env.NODE_ENV === "development",
);

const PROJECT_METHOD = {
  id: "project_model",
  label: "E20 ResNet-18",
  hint: "Projede eğitildi · 128 px yerel kesitler · top-3",
};

const RESEARCH_METHODS = [
  { id: "auto", label: "Eski otomatik", hint: "Boyuta göre eski yöntem seçimi" },
  { id: "cnn", label: "CNN", hint: "Küçültülmüş görsel CNN'i" },
  { id: "stats", label: "İstatistik", hint: "68 elle tasarlanmış özellik" },
  { id: "tiles", label: "Özellik tile'ları", hint: "Eski istatistiksel kesit haritası" },
];

const SIGNAL_TEXT: Record<Verdict, string> = {
  ai: "sinyal AI yönünde",
  real: "sinyal gerçek yönünde",
  uncertain: "sinyal kararsız bantta",
};

function ExternalComparison({ decision, enoughEvidence }: {
  decision: Decision | null;
  enoughEvidence: boolean;
}) {
  return (
    <section className="comparison-card" aria-label="Haricî dedektör karşılaştırması">
      <div className="section-kicker">Haricî karşılaştırma · E26</div>
      {!enoughEvidence ? (
        <p>Karşılaştırma için görselin her iki boyutu da en az 48 piksel olmalıdır.</p>
      ) : !decision ? (
        <p>Haricî karşılaştırma modeli bu çalıştırmada kullanılamıyor.</p>
      ) : (
        <>
          <h4 className={decision.label === "ai" ? "comparison-ai" : "comparison-quiet"}>
            {decision.label === "ai"
              ? "Haricî katman AI kanıtı buldu"
              : "Haricî katmanda yeterli kanıt yok"}
          </h4>
          <p>
            {decision.label === "ai"
              ? `Eşiği aşan: ${decision.triggered_by
                  .map((id) => decision.arms[id]?.label ?? id)
                  .join(" + ")}. Bu sonuç da yeni kaynaklar için garanti değildir.`
              : "Bu sonuç ‘gerçektir’ anlamına gelmez; haricî katman da yalnız AI yönlü kanıt arar."}
          </p>
          <div className="arm-list">
            {Object.entries(decision.arms).map(([id, arm]) => (
              <span key={id} className={arm.band === "ai" ? "arm-hit" : ""}>
                {arm.label}: {arm.score.toFixed(2)} / {arm.threshold.toFixed(2)}
              </span>
            ))}
            {decision.caveats.map((caveat) => (
              <span key={caveat} className="arm-caveat">
                {CAVEAT_TEXT[caveat] ?? caveat}
              </span>
            ))}
          </div>
          <small>{decision.provenance}</small>
        </>
      )}
    </section>
  );
}

function ProjectResult({ analysis }: { analysis: Analysis }) {
  const project = analysis.project_model;
  if (!project) return null;
  const scorePercent = Math.min(100, Math.max(0, project.score * 100));
  const thresholdPercent = Math.min(100, Math.max(0, project.threshold * 100));

  return (
    <section
      className={`project-result ${project.triggered ? "project-triggered" : "project-below"}`}
      aria-label="Proje modeli sonucu"
    >
      <div className="section-kicker">Proje modeli · {project.revision}</div>
      <div className="project-result-heading">
        <span aria-hidden="true">{project.triggered ? "◆" : "◇"}</span>
        <div>
          <h3>{project.triggered ? "AI yönünde deneysel sinyal" : "Deneysel eşik aşılmadı"}</h3>
          <p>
            {project.triggered
              ? "Kendi modelimizin ham skoru, E20 kalibrasyonunda seçilen eşiğin üzerinde."
              : "Kendi modelimizin ham skoru deneysel eşiğin altında; bu, görselin gerçek olduğunu kanıtlamaz."}
          </p>
        </div>
      </div>

      <div className="project-meter" aria-label={`Skor ${project.score.toFixed(3)}, eşik ${project.threshold.toFixed(3)}`}>
        <div className="project-meter-track">
          <span className="project-meter-fill" style={{ width: `${scorePercent}%` }} />
          <span className="project-meter-threshold" style={{ left: `${thresholdPercent}%` }} />
        </div>
        <div className="project-meter-labels">
          <strong>Ham skor {project.score.toFixed(3)}</strong>
          <span>Deneysel eşik {project.threshold.toFixed(3)}</span>
        </div>
      </div>

      <dl className="model-facts">
        <div><dt>Model</dt><dd>{project.artifact_id}</dd></div>
        <div><dt>Kesit</dt><dd>{project.tile_count} × {project.tile_px}px</dd></div>
        <div><dt>Birleştirme</dt><dd>{project.aggregation}</dd></div>
        <div><dt>Artifact</dt><dd title={project.artifact_sha256}>{project.artifact_sha256.slice(0, 12)}…</dd></div>
      </dl>

      <p className="limitation-note"><strong>Açık sınır:</strong> {project.limitation}</p>
    </section>
  );
}

function LegacyResult({ analysis }: { analysis: Analysis }) {
  const scorePosition = Math.min(100, Math.max(0, analysis.p_ai * 100));
  return (
    <section className="legacy-result" aria-label="Eski araştırma yöntemi sonucu">
      <div className="section-kicker">Eski araştırma yöntemi</div>
      <p>
        Bu skor kanonik proje modeli değildir ve karara dahil edilmez. Yalnız geçmiş deneyleri
        karşılaştırmak için gösterilir.
      </p>
      <div className="probability">
        <div className="probability-labels">
          <span>0</span>
          <strong>{analysis.p_ai.toFixed(3)} · {SIGNAL_TEXT[analysis.verdict]}</strong>
          <span>1</span>
        </div>
        <div className="probability-track">
          <div className={`probability-fill verdict-${analysis.verdict}`} style={{ width: `${scorePosition}%` }} />
        </div>
      </div>
      <small>{analysis.method_label} · {analysis.resolution}</small>
    </section>
  );
}

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef<string | null>(null);
  const requestGateRef = useRef(new LatestRequestGate());
  const [preview, setPreview] = useState<Preview | null>(null);
  const [method, setMethod] = useState(PROJECT_METHOD.id);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => () => {
    requestGateRef.current.cancel();
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
  }, []);

  function cancelAnalysis() {
    requestGateRef.current.cancel();
    setLoading(false);
  }

  function releasePreview() {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
  }

  function choose(file?: File) {
    if (!file) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      setError("Bu dosya desteklenmiyor. JPG, PNG veya WEBP seçin.");
      return;
    }
    cancelAnalysis();
    releasePreview();
    setAnalysis(null);
    setError(null);
    const url = URL.createObjectURL(file);
    previewUrlRef.current = url;
    setPreview({ name: file.name, url, size: `${(file.size / 1024 / 1024).toFixed(2)} MB`, file });
  }

  function clear() {
    cancelAnalysis();
    releasePreview();
    setPreview(null);
    setAnalysis(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    choose(event.dataTransfer.files?.[0]);
  }

  async function analyze(useMethod = method) {
    if (!preview) return;
    const selectedPreview = preview;
    const ticket = requestGateRef.current.begin();
    setLoading(true);
    setError(null);
    try {
      const body = new FormData();
      body.append("image", selectedPreview.file);
      body.append("method", useMethod);
      const response = await fetch(analysisEndpoint(API_ORIGIN), {
        method: "POST",
        body,
        signal: ticket.signal,
      });
      let payload: unknown;
      try {
        payload = await response.json();
      } catch {
        throw new AnalysisResponseError();
      }
      if (!response.ok) {
        const detail =
          typeof payload === "object" && payload !== null &&
          "detail" in payload && typeof payload.detail === "string"
            ? payload.detail
            : undefined;
        throw new AnalysisHttpError(response.status, detail);
      }
      const parsed = parseAnalysis(payload);
      if (requestGateRef.current.isCurrent(ticket.id)) setAnalysis(parsed);
    } catch (caught) {
      if (ticket.signal.aborted || !requestGateRef.current.isCurrent(ticket.id)) return;
      setError(analysisErrorMessage(caught));
    } finally {
      if (requestGateRef.current.isCurrent(ticket.id)) setLoading(false);
    }
  }

  function pick(id: string) {
    setMethod(id);
    if (analysis) analyze(id);
  }

  const isProjectMethod = method === PROJECT_METHOD.id;

  return (
    <main>
      <header>
        <div className="header-content">
          <strong>PixelProof</strong>
          <span>Proje modeli · E20</span>
        </div>
      </header>

      <div className="container">
        <section className="intro">
          <span className="intro-kicker">Çalıştırılabilir model demosu</span>
          <h1>Kendi AI görsel modelimizi deneyin</h1>
          <p>
            Bir görsel yükleyin. Önce projede eğitilen ResNet-18 çalışır; varsa haricî karar
            katmanı sonucu ayrı bir karşılaştırma olarak gösterilir.
          </p>
        </section>

        <section className="grid">
          <div className="panel">
            <div className="panel-title">
              <h2>1. Görsel yükle</h2>
              <p>JPG, PNG veya WEBP · en fazla 12 MB</p>
            </div>

            {!preview ? (
              <div
                className={`dropzone ${dragging ? "dragging" : ""}`}
                onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
              >
                <div className="upload-symbol">↑</div>
                <strong>Fotoğrafı buraya bırakın</strong>
                <span>veya bilgisayarınızdan seçin</span>
                <button type="button" onClick={() => inputRef.current?.click()}>Dosya seç</button>
              </div>
            ) : (
              <div className="preview">
                <div className="preview-stage">
                  {/* Blob previews are local-only and cannot use the hosted image optimizer. */}
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={preview.url} alt="Seçilen görsel" />
                  {analysis?.tile_map && (
                    <div className="tile-overlay">
                      {analysis.tile_map.tiles.map((tile, index) => (
                        <span
                          key={index}
                          title={`Ham kesit skoru ${tile.p_ai.toFixed(3)}`}
                          style={{
                            left: `${(tile.x / analysis.tile_map!.image_w) * 100}%`,
                            top: `${(tile.y / analysis.tile_map!.image_h) * 100}%`,
                            width: `${(analysis.tile_map!.tile_px / analysis.tile_map!.image_w) * 100}%`,
                            height: `${(analysis.tile_map!.tile_px / analysis.tile_map!.image_h) * 100}%`,
                            background: `rgba(217,45,32,${Math.max(0, tile.p_ai - 0.5) * 1.4})`,
                            borderColor: tile.p_ai >= 0.7 ? "rgba(217,45,32,.85)" : "transparent",
                          }}
                        />
                      ))}
                    </div>
                  )}
                </div>
                <div className="file-info">
                  <div><strong>{preview.name}</strong><span>{preview.size}</span></div>
                  <button type="button" onClick={clear}>Kaldır</button>
                </div>
                {analysis?.tile_map && (
                  <p className="tile-note">
                    Harita her kesitin ham model skorunu gösterir; düzenleme konumunun kanıtı değildir.
                  </p>
                )}
              </div>
            )}
            <input
              ref={inputRef}
              hidden
              type="file"
              accept="image/jpeg,image/png,image/webp"
              aria-label="Analiz edilecek görseli seç"
              onChange={(event: ChangeEvent<HTMLInputElement>) => choose(event.target.files?.[0])}
            />
          </div>

          <div className="panel result-panel">
            <div className="panel-title">
              <h2>2. Proje modelini çalıştır</h2>
              <p>Birincil sonuç her zaman kendi E20 modelimize aittir</p>
            </div>

            <button
              type="button"
              className={`primary-method ${isProjectMethod ? "active" : ""}`}
              onClick={() => pick(PROJECT_METHOD.id)}
              disabled={loading}
              aria-pressed={isProjectMethod}
            >
              <span><strong>{PROJECT_METHOD.label}</strong><small>{PROJECT_METHOD.hint}</small></span>
              <b>{isProjectMethod ? "Seçili" : "Seç"}</b>
            </button>

            <details className="research-methods">
              <summary>Eski araştırma yöntemlerini aç</summary>
              <p>Geçmiş deneyleri karşılaştırmak içindir; proje modelinin yerine geçmez.</p>
              <div className="methods">
                {RESEARCH_METHODS.map((candidate) => (
                  <button
                    key={candidate.id}
                    type="button"
                    className={`method ${method === candidate.id ? "active" : ""}`}
                    onClick={() => pick(candidate.id)}
                    disabled={loading}
                    aria-pressed={method === candidate.id}
                    title={candidate.hint}
                  >
                    {candidate.label}
                  </button>
                ))}
              </div>
            </details>

            {!analysis ? (
              <div className="result empty-result" aria-live="polite" aria-busy={loading}>
                <div className="result-icon">P</div>
                <h3>{preview ? "Proje modeli hazır" : "Önce bir görsel yükleyin"}</h3>
                <p>
                  {preview
                    ? "Tek düğmeyle doğrulanmış E20 checkpoint’ini çalıştırabilirsiniz."
                    : "Model, görseli doğal çözünürlükte 128 px kesitler üzerinden inceler."}
                </p>
                {error && <p className="error-text" role="alert">{error}</p>}
                <button type="button" disabled={!preview || loading} onClick={() => analyze()}>
                  {loading ? "Model çalışıyor…" : isProjectMethod ? "Proje modelini çalıştır" : "Seçili yöntemi çalıştır"}
                </button>
              </div>
            ) : (
              <div className="result result-stack" aria-live="polite" aria-busy={loading}>
                {analysis.project_model
                  ? <ProjectResult analysis={analysis} />
                  : <LegacyResult analysis={analysis} />}
                <ExternalComparison
                  decision={analysis.decision}
                  enoughEvidence={analysis.enough_evidence}
                />
                {error && <p className="error-text" role="alert">{error}</p>}
                <button type="button" onClick={() => analyze()} disabled={loading}>
                  {loading ? "Model çalışıyor…" : "Tekrar çalıştır"}
                </button>
              </div>
            )}
          </div>
        </section>

        <aside>
          <strong>Bu ekrandaki ayrım:</strong> Birinci kart projede eğittiğimiz E20 ResNet-18’in
          deneysel sonucudur. Üç seed değerlendirmesinde worst-source yanlış alarmı
          %86,2 ± %3,1 olduğu için gerçeklik sertifikası değildir. Haricî E26 karşılaştırması
          ayrı tutulur; onun ölçülen iki-kollu worst-source sonucu 11/103, yani %10,7 idi
          (Wilson %95: %6,1–%18,1). İki sistem de negatif sonuçta “gerçektir” demez.
        </aside>
      </div>
    </main>
  );
}
