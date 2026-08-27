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
  type R1bResearchResult,
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

function ExternalComparison({ decision, enoughEvidence }: {
  decision: Decision | null;
  enoughEvidence: boolean;
}) {
  return (
    <section className="comparison-card" aria-label="Ana karar katmanı">
      <div className="card-topline">
        <div className="section-kicker">Ana karar katmanı · E26</div>
        <span className="status-chip status-measured">Ölçülmüş</span>
      </div>
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

function R1bResearch({ candidate }: { candidate: R1bResearchResult | null }) {
  if (!candidate) return null;
  const scorePercent = candidate.score * 100;
  const thresholdPercent = candidate.threshold * 100;

  return (
    <section
      className={`r1b-card ${candidate.triggered ? "r1b-triggered" : "r1b-quiet"}`}
      aria-label="E32 R1b deneysel ikinci görüş"
    >
      <div className="card-topline">
        <div className="section-kicker">Yeni proje adayı · R1b</div>
        <span className="status-chip status-research">Kararı etkilemez</span>
      </div>
      <h3>{candidate.triggered ? "R1b, AI yönünde sinyal verdi" : "R1b eşiği aşılmadı"}</h3>
      <p>
        {candidate.triggered
          ? "Bu yalnız deneysel bir tetiklenmedir; gerçek fotoğraflardaki yüksek yanlış alarm nedeniyle ana hüküm değildir."
          : "Bu yalnız yetersiz AI kanıtı demektir; görselin gerçek olduğunu doğrulamaz."}
      </p>
      <div className="research-meter" aria-label={`R1b skoru ${candidate.score.toFixed(3)}, eşik ${candidate.threshold.toFixed(3)}`}>
        <div className="research-meter-track">
          <span className="research-meter-fill" style={{ width: `${scorePercent}%` }} />
          <span className="research-meter-threshold" style={{ left: `${thresholdPercent}%` }} />
        </div>
        <div className="project-meter-labels">
          <strong>Model skoru {candidate.score.toFixed(3)}</strong>
          <span>Eşik {candidate.threshold.toFixed(3)} · olasılık değil</span>
        </div>
      </div>
      <div className="risk-grid">
        <div><strong>%{(candidate.evaluation.ipn_worst_device_fp * 100).toFixed(0)}</strong><span>IPN en kötü cihaz FP</span></div>
        <div><strong>%{(candidate.evaluation.owner_gallery_fp * 100).toFixed(1)}</strong><span>Galeri FP</span></div>
      </div>
      <p className="limitation-note"><strong>Ölçülen sınır:</strong> {candidate.limitation}</p>
      <small title={candidate.artifact_sha256}>Artifact {candidate.artifact_sha256.slice(0, 12)}…</small>
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

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef<string | null>(null);
  const requestGateRef = useRef(new LatestRequestGate());
  const [preview, setPreview] = useState<Preview | null>(null);
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

  async function analyze() {
    if (!preview) return;
    const selectedPreview = preview;
    const ticket = requestGateRef.current.begin();
    setLoading(true);
    setError(null);
    try {
      const body = new FormData();
      body.append("image", selectedPreview.file);
      body.append("method", PROJECT_METHOD.id);
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

  return (
    <main>
      <header>
        <div className="header-content">
          <div className="brand"><i aria-hidden="true">P</i><strong>PixelProof</strong></div>
          <span>Yerel araştırma demosu</span>
        </div>
      </header>

      <div className="container">
        <section className="intro">
          <span className="intro-kicker">Tek görsel · ayrıştırılmış kanıt katmanları</span>
          <h1>Bir görsel yükle, modellerin ne gördüğünü karşılaştır.</h1>
          <p>
            Ana E26 yorumu, yeni R1b araştırma adayı ve E20 taban modeli birbirine karıştırılmadan
            gösterilir. Hiçbir negatif sonuç gerçeklik sertifikası değildir.
          </p>
          <div className="intro-pills" aria-label="Demo özellikleri">
            <span>Dosya cihazında kalır</span><span>Tek seferde analiz</span><span>Sınırlar açıkça görünür</span>
          </div>
        </section>

        <section className="grid">
          <div className="panel">
            <div className="panel-title">
              <h2>Görsel</h2>
              <p>JPG, PNG veya WEBP · en fazla 12 MB</p>
            </div>

            {!preview ? (
              <div
                className={`dropzone ${dragging ? "dragging" : ""}`}
                onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
              >
                <div className="upload-symbol" aria-hidden="true">＋</div>
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
              <h2>Analiz</h2>
              <p>Ölçülmüş karar ile deneysel sinyaller ayrı kalır</p>
            </div>

            <div className="primary-method active" aria-label="Çalışacak model zinciri">
              <span><strong>{PROJECT_METHOD.label}</strong><small>{PROJECT_METHOD.hint}</small></span>
              <b>Sabit taban</b>
            </div>

            {!analysis ? (
              <div className="result empty-result" aria-live="polite" aria-busy={loading}>
                <div className="result-icon" aria-hidden="true">◎</div>
                <h3>{preview ? "Analize hazır" : "Önce bir görsel yükleyin"}</h3>
                <p>
                  {preview
                    ? "Tek çalıştırmada karar katmanı, R1b adayı ve E20 tabanı ayrı ayrı görünecek."
                    : "JPG, PNG veya WEBP seçin; sonuçlar aynı görsel üzerinde karşılaştırılsın."}
                </p>
                {error && <p className="error-text" role="alert">{error}</p>}
                <button type="button" disabled={!preview || loading} onClick={() => analyze()}>
                  {loading ? "Modeller inceliyor…" : "Görseli analiz et"}
                </button>
              </div>
            ) : (
              <div className="result result-stack" aria-live="polite" aria-busy={loading}>
                <ExternalComparison
                  decision={analysis.decision}
                  enoughEvidence={analysis.enough_evidence}
                />
                <R1bResearch candidate={analysis.r1b_research} />
                {analysis.project_model && (
                  <details className="baseline-details">
                    <summary>E20 taban modelinin teknik sonucunu göster</summary>
                    <ProjectResult analysis={analysis} />
                  </details>
                )}
                {error && <p className="error-text" role="alert">{error}</p>}
                <button type="button" onClick={() => analyze()} disabled={loading}>
                  {loading ? "Modeller inceliyor…" : "Yeniden analiz et"}
                </button>
              </div>
            )}
          </div>
        </section>

        <aside>
          <strong>Sonucu nasıl okuyacaksın?</strong> E26 ana karar katmanıdır; R1b en yeni fakat
          başarısız dış teste sahip proje adayıdır; E20 ise teknik tabandır. Modeller uyuşmazsa bu
          hata değil, veri kaymasının görünür kanıtıdır. Hiçbiri tek başına gerçeklik sertifikası
          vermez.
        </aside>
      </div>
    </main>
  );
}
