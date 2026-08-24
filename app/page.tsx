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

// Four methods, never blended — E9 showed a fixed blend adds +0.002 (noise).
// "auto" applies the measured 700px crossover from E11.
const METHODS = [
  { id: "auto", label: "Otomatik", hint: "Boyuta göre en güçlü yöntem" },
  { id: "cnn", label: "CNN", hint: "Küçültülmüş görsel · küçük girdilerde güçlü" },
  { id: "stats", label: "İstatistik", hint: "68 istatistik · fizik izleri, küçültme yok" },
  { id: "tiles", label: "Kare kare", hint: "En fazla 256 yerel kesit · dedektör-skor haritası" },
];

const SIGNAL_TEXT: Record<Verdict, string> = {
  ai: "sinyal: AI yönünde",
  real: "sinyal: gerçek yönünde",
  uncertain: "sinyal: kararsız bantta",
};

const VERDICT_COLOR: Record<Verdict, string> = { ai: "#d92d20", real: "#12b76a", uncertain: "#f79009" };

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef<string | null>(null);
  const requestGateRef = useRef(new LatestRequestGate());
  const [preview, setPreview] = useState<Preview | null>(null);
  const [method, setMethod] = useState("auto");
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
          typeof payload === "object" && payload !== null && "detail" in payload && typeof payload.detail === "string"
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

  // This is a display position for a raw research score, not a calibrated probability.
  const scorePosition = analysis ? Math.min(100, Math.max(0, analysis.p_ai * 100)) : 0;

  return (
    <main>
      <header>
        <div className="header-content">
          <strong>AI Image Detector</strong>
          <span>Module 1 · 4 yöntem</span>
        </div>
      </header>

      <div className="container">
        <section className="intro">
          <h1>Görsel Gerçeklik Analizi</h1>
          <p>Bir fotoğraf yükleyin, hangi yöntemle inceleneceğini seçin.</p>
        </section>

        <section className="grid">
          <div className="panel">
            <div className="panel-title">
              <h2>1. Görsel yükle</h2>
              <p>JPG, PNG veya WEBP</p>
            </div>

            {!preview ? (
              <div
                className={`dropzone ${dragging ? "dragging" : ""}`}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
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
                      {analysis.tile_map.tiles.map((t, i) => (
                        <span
                          key={i}
                          title={`%${(t.p_ai * 100).toFixed(0)}`}
                          style={{
                            left: `${(t.x / analysis.tile_map!.image_w) * 100}%`,
                            top: `${(t.y / analysis.tile_map!.image_h) * 100}%`,
                            width: `${(analysis.tile_map!.tile_px / analysis.tile_map!.image_w) * 100}%`,
                            height: `${(analysis.tile_map!.tile_px / analysis.tile_map!.image_h) * 100}%`,
                            background: `rgba(217,45,32,${Math.max(0, t.p_ai - 0.5) * 1.4})`,
                            borderColor: t.p_ai >= 0.7 ? "rgba(217,45,32,.85)" : "transparent",
                          }}
                        />
                      ))}
                    </div>
                  )}
                </div>
                <div className="file-info" style={{ textAlign: "left" }}>
                  <div><strong>{preview.name}</strong><span>{preview.size}</span></div>
                  <button type="button" onClick={clear}>Kaldır</button>
                </div>
                {analysis?.tile_map && (
                  <p className="tile-note">
                    Kırmızı yoğunluk o kesitin ham AI-yönlü dedektör skorudur. Konum doğruluğu
                    piksel maskeleriyle doğrulanmadı; bir düzenleme yerinin kanıtı değildir.
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
              onChange={(e: ChangeEvent<HTMLInputElement>) => choose(e.target.files?.[0])}
            />
          </div>

          <div className="panel">
            <div className="panel-title">
              <h2>2. Yöntem ve sonuç</h2>
              <p>Yöntemi değiştirince analiz yenilenir</p>
            </div>

            <div className="methods">
              {METHODS.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  className={`method ${method === m.id ? "active" : ""}`}
                  onClick={() => pick(m.id)}
                  disabled={loading}
                  aria-pressed={method === m.id}
                  title={m.hint}
                >
                  {m.label}
                </button>
              ))}
            </div>
            <p className="method-hint">{METHODS.find((m) => m.id === method)?.hint}</p>

            {!analysis ? (
              <div className="result" aria-live="polite" aria-busy={loading}>
                <div className="result-icon">?</div>
                <h3>{preview ? "Görsel analize hazır" : "Henüz sonuç yok"}</h3>
                <p>{preview ? "Analiz butonuna basarak modeli çalıştırın." : "Analiz için önce bir görsel yükleyin."}</p>
                {error && <p className="error-text" role="alert">{error}</p>}
                <button type="button" disabled={!preview || loading} onClick={() => analyze()}>
                  {loading ? "Analiz ediliyor…" : "Analiz et"}
                </button>
              </div>
            ) : (
              <div className="result" aria-live="polite" aria-busy={loading}>
                {/* ─── HÜKÜM — tek karar burada verilir (E22–E26) ─── */}
                {analysis.decision ? (
                  <div
                    style={{
                      borderRadius: 14,
                      padding: "20px 18px",
                      marginBottom: 18,
                      textAlign: "center",
                      background: analysis.decision.label === "ai" ? "rgba(217,45,32,.09)" : "rgba(90,98,112,.07)",
                      border: `2px solid ${analysis.decision.label === "ai" ? "#d92d20" : "rgba(90,98,112,.35)"}`,
                    }}
                  >
                    <div style={{ fontSize: 34, lineHeight: 1 }}>
                      {analysis.decision.label === "ai" ? "⚠️" : "◌"}
                    </div>
                    <strong
                      style={{
                        display: "block",
                        margin: "8px 0 4px",
                        fontSize: 20,
                        letterSpacing: 0.5,
                        color: analysis.decision.label === "ai" ? "#d92d20" : "#475467",
                      }}
                    >
                      {analysis.decision.label === "ai" ? "YAPAY ZEKÂ TESPİT EDİLDİ" : "YETERLİ KANIT YOK"}
                    </strong>
                    <p style={{ margin: "4px 0 12px", fontSize: 13.5 }}>
                      {analysis.decision.label === "ai"
                        ? `Yakalayan: ${analysis.decision.triggered_by
                            .map((id) => analysis.decision!.arms[id]?.label ?? id)
                            .join(" + ")} — skor dondurulmuş eşiğin üstünde. Ayrı değerlendirmedeki en yüksek yanlış alarm %10,7 idi; bu sonuç yeni kamera veya platformlar için garanti değildir.`
                        : "Hiçbir dedektör eşiğini aşmadı. Bu bir 'gerçektir' garantisi değil — sistem bilerek 'gerçek' kararı vermez (E23a)."}
                    </p>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 8, fontSize: 12, justifyContent: "center" }}>
                      {Object.entries(analysis.decision.arms).map(([id, arm]) => (
                        <span
                          key={id}
                          style={{
                            padding: "3px 9px",
                            borderRadius: 6,
                            background: arm.band === "ai" ? "rgba(217,45,32,.12)" : "rgba(0,0,0,.05)",
                            fontWeight: arm.band === "ai" ? 600 : 400,
                          }}
                        >
                          {arm.label}: {arm.score.toFixed(2)} / eşik {arm.threshold.toFixed(2)}
                          {arm.band === "ai" ? " ✓" : ""}
                        </span>
                      ))}
                      {analysis.decision.caveats.map((c) => (
                        <span key={c} style={{ padding: "3px 9px", borderRadius: 6, background: "rgba(247,144,9,.15)", color: "#b54708" }}>
                          {CAVEAT_TEXT[c] ?? c}
                        </span>
                      ))}
                    </div>
                    <p style={{ margin: "10px 0 0", fontSize: 11, opacity: 0.6 }}>{analysis.decision.provenance}</p>
                  </div>
                ) : (
                  <p className="error-text" style={{ marginBottom: 14 }}>
                    {!analysis.enough_evidence
                      ? "Resmî karar için görselin her iki boyutu da en az 48 piksel olmalıdır."
                      : "Karar katmanı kullanılamıyor — aşağıda yalnız araştırma skoru gösteriliyor."}
                  </p>
                )}

                {/* ─── ARAŞTIRMA SİNYALİ — karara dahil değildir ─── */}
                <div
                  style={{
                    borderRadius: 10,
                    border: "1px dashed rgba(90,98,112,.4)",
                    padding: "12px 14px",
                    textAlign: "left",
                    opacity: 0.85,
                  }}
                >
                  <p style={{ margin: 0, fontSize: 11, textTransform: "uppercase", letterSpacing: 0.8, color: "#5a6270" }}>
                    Araştırma skoru — karara dahil değil
                  </p>
                  <p style={{ margin: "4px 0 10px", fontSize: 12, color: "#5a6270" }}>
                    Projede eğittiğimiz modelin kalibre edilmemiş ham skoru. Görmediği kameraların gerçek fotoğraflarında
                    %79–100 yanlış alarm verdiğini ölçtük (E13/E24) — burada dürüstlükle, eğitim amaçlı duruyor.
                  </p>
                  <div className="probability">
                    <div className="probability-labels">
                      <span>0</span>
                      <strong>
                        ham skor = {analysis.p_ai.toFixed(3)} · {SIGNAL_TEXT[analysis.verdict]}
                      </strong>
                      <span>1</span>
                    </div>
                    <div className="probability-track">
                      <div className="probability-fill" style={{ width: `${scorePosition}%`, background: VERDICT_COLOR[analysis.verdict] }} />
                    </div>
                  </div>
                  <p className="model-info" style={{ marginTop: 8 }}>
                    {analysis.method_label}
                    {analysis.auto_selected && " · otomatik seçildi"}
                    {" · "}{analysis.resolution}
                    {!analysis.enough_evidence && " · ⚠️ ölçüm için çok küçük"}
                  </p>
                </div>

                {error && <p className="error-text" role="alert">{error}</p>}
                <button type="button" onClick={() => analyze()} disabled={loading} style={{ marginTop: 14 }}>
                  {loading ? "Analiz ediliyor…" : "Tekrar analiz et"}
                </button>
              </div>
            )}
          </div>
        </section>

        <aside>
          <strong>Karar nasıl veriliyor:</strong> Üstteki karar, dondurulmuş bir dedektörün
          skorunu 12 gerçek kamera kaynağıyla kalibre edilmiş eşikten geçirir (E22–E24) ve
          yalnız iki cevap verir: &quot;AI tespit edildi&quot; ya da &quot;yeterli kanıt yok&quot; —
          &quot;gerçektir&quot; demez, çünkü o kararın hiçbir kaynakta tutarlı verilemediğini ölçtük.
          Alttaki yöntemler projenin kendi araştırma sinyalleridir; karıştırılmazlar (E9: en iyi
          karışım +0.002). Bilinen sınırlar: GPT Image ailesinde ayrım gücü şans seviyesinde,
          ağır sıkıştırmada yakalama düşer. Bu bir araştırma demosudur.
          E26 iki-kollu OR sisteminin ayrı değerlendirmedeki worst-source yanlış alarmı
          iPhone kaynağında 11/103, yani %10,7 idi (Wilson %95 güven aralığı %6,1–%18,1).
          Bu sayı yalnız ölçülen 12 kaynak ve mevcut örneklem için geçerlidir.
        </aside>
      </div>
    </main>
  );
}
