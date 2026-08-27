import type { Analysis, Decision, R1bResearchResult } from "./analysis-contract";

const CAVEAT_TEXT: Record<string, string> = {
  "megapiksel-siniri": "Etkin B-Free koluna 2048px sınırı uygulandı (E23b)",
  "sikistirilmis-girdi": "Düşük byte/piksel — yalnız kaba sıkıştırma uyarısı (E23c)",
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

export function R1bResearch({ candidate }: { candidate: R1bResearchResult | null }) {
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

export function TechnicalDetails({ analysis }: { analysis: Analysis }) {
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
