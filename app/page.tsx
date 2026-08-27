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

const PROJECT_METHOD = "project_model";

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
  if (!candidate) {
    return (
      <section className="main-model-card model-unavailable" aria-label="Yeni model kullanılamıyor">
        <div className="model-answer-icon" aria-hidden="true">—</div>
        <span className="main-model-kicker">Yeni modelimiz · E32 R1b</span>
        <h2>Yeni model bu oturumda etkin değil</h2>
        <p>R1b veri köküyle yerel demoyu başlattığınızda sonucu burada doğrudan göreceksiniz.</p>
      </section>
    );
  }
  const scorePercent = candidate.score * 100;
  const thresholdPercent = candidate.threshold * 100;
  const thresholdDistance = Math.abs(scorePercent - thresholdPercent);

  return (
    <section
      className={`main-model-card ${candidate.triggered ? "model-ai" : "model-insufficient"}`}
      aria-label="Yeni E32 R1b modelinin sonucu"
    >
      <div className="model-answer-heading">
        <div className="model-answer-icon" aria-hidden="true">{candidate.triggered ? "AI" : "?"}</div>
        <div>
          <span className="main-model-kicker">Yeni modelimizin cevabı · E32 R1b</span>
          <h2>{candidate.triggered ? "AI yönünde sinyal buldu" : "Yeterli AI sinyali bulamadı"}</h2>
        </div>
      </div>
      <p className="model-answer-copy">
        {candidate.triggered
          ? `Skor, karar eşiğinin ${thresholdDistance.toFixed(1)} puan üzerinde olduğu için model bu görseli AI yönünde işaretledi.`
          : `Skor, karar eşiğinin ${thresholdDistance.toFixed(1)} puan altında olduğu için model bu görseli AI olarak işaretlemedi; bu, görselin kesinlikle gerçek olduğu anlamına gelmez.`}
      </p>
      <div className="signal-score-row">
        <span>AI sinyal skoru</span>
        <strong>%{scorePercent.toFixed(1)}</strong>
      </div>
      <div
        className="signal-meter"
        role="progressbar"
        aria-label={`AI sinyal skoru yüzde ${scorePercent.toFixed(1)}`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Number(scorePercent.toFixed(1))}
      >
        <div className="signal-meter-track">
          <span className="signal-meter-fill" style={{ width: `${scorePercent}%` }} />
          <span className="signal-meter-threshold" style={{ left: `${thresholdPercent}%` }}>
            <i>Eşik %{thresholdPercent.toFixed(1)}</i>
          </span>
        </div>
        <div className="signal-meter-labels">
          <span>Düşük sinyal</span>
          <span>Yüksek sinyal</span>
        </div>
      </div>
      <p className="score-disclaimer">Bu yüzde kalibre edilmiş olasılık değil, modelin 0–100 ölçeğine çevrilmiş ham AI sinyal skorudur.</p>
    </section>
  );
}

function TechnicalDetails({ analysis }: { analysis: Analysis }) {
  const candidate = analysis.r1b_research;
  return (
    <details className="technical-details">
      <summary>
        <span><strong>Teknik detaylar</strong><small>Diğer modeller, eşikler ve test sonuçları</small></span>
        <b aria-hidden="true">＋</b>
      </summary>
      <div className="technical-stack">
        {candidate && (
          <section className="r1b-diagnostics" aria-label="R1b teknik sınırları">
            <div className="section-kicker">R1b test sınırları</div>
            <div className="risk-grid">
              <div><strong>%{(candidate.evaluation.ipn_worst_device_fp * 100).toFixed(0)}</strong><span>IPN en kötü cihaz FP</span></div>
              <div><strong>%{(candidate.evaluation.owner_gallery_fp * 100).toFixed(1)}</strong><span>Galeri FP</span></div>
            </div>
            <p className="limitation-note"><strong>Ölçülen sınır:</strong> {candidate.limitation}</p>
            <small title={candidate.artifact_sha256}>Artifact {candidate.artifact_sha256.slice(0, 12)}…</small>
          </section>
        )}
        <ExternalComparison decision={analysis.decision} enoughEvidence={analysis.enough_evidence} />
        {analysis.project_model && <ProjectResult analysis={analysis} />}
      </div>
    </details>
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
      body.append("method", PROJECT_METHOD);
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
          <span className="intro-kicker">Yeni modelimiz · E32 R1b</span>
          <h1>Yeni modelimiz bu görsel için ne diyor?</h1>
          <p>
            Fotoğrafını yükle; modelin doğrudan cevabını ve AI sinyal barını gör. Ayrıntılı teknik
            ölçümler yalnız istersen açılır.
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
                </div>
                <div className="file-info">
                  <div><strong>{preview.name}</strong><span>{preview.size}</span></div>
                  <button type="button" onClick={clear}>Kaldır</button>
                </div>
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
              <h2>Yeni modelin sonucu</h2>
              <p>E32 R1b · doğrudan model cevabı</p>
            </div>

            {!analysis ? (
              <div className="result empty-result" aria-live="polite" aria-busy={loading}>
                <div className="result-icon" aria-hidden="true">◎</div>
                <h3>{preview ? "Analize hazır" : "Önce bir görsel yükleyin"}</h3>
                <p>
                  {preview
                    ? "Tek düğmeyle E32 R1b modelinin cevabını görebilirsin."
                    : "JPG, PNG veya WEBP seç; yeni modelimizin cevabı burada görünsün."}
                </p>
                {error && <p className="error-text" role="alert">{error}</p>}
                <button type="button" disabled={!preview || loading} onClick={() => analyze()}>
                  {loading ? "Modeller inceliyor…" : "Görseli analiz et"}
                </button>
              </div>
            ) : (
              <div className="result result-stack" aria-live="polite" aria-busy={loading}>
                <R1bResearch candidate={analysis.r1b_research} />
                <TechnicalDetails analysis={analysis} />
                {error && <p className="error-text" role="alert">{error}</p>}
                <button type="button" onClick={() => analyze()} disabled={loading}>
                  {loading ? "Modeller inceliyor…" : "Yeniden analiz et"}
                </button>
              </div>
            )}
          </div>
        </section>

        <aside>
          <strong>Kısa not:</strong> Ana kart yalnız yeni E32 R1b modelimizin cevabıdır. Skor bir
          olasılık veya gerçeklik sertifikası değildir; model gerçek fotoğraflarda yanlış alarm
          verebildiği için sonucu deneysel kanıt olarak yorumla.
        </aside>
      </div>
    </main>
  );
}
