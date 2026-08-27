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
} from "./analysis-contract";
import { R1bResearch, TechnicalDetails } from "./result-panels";

type Preview = { name: string; url: string; size: string; file: File };

const API_ORIGIN = resolveApiOrigin(
  process.env.NEXT_PUBLIC_PIXELPROOF_API_URL,
  process.env.NODE_ENV === "development",
);

const PROJECT_METHOD = "project_model";

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
