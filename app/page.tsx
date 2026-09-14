"use client";

import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import {
  AnalysisHttpError,
  AnalysisResponseError,
  LatestRequestGate,
  analysisErrorMessage,
  resolveApiOrigin,
} from "./analysis-contract";
import { demoFileError, OUTCOME_COPY, parseDemoAnalysis, type DemoAnalysis } from "./demo-contract";

type Preview = { name: string; url: string; size: string; file: File };

const API_ORIGIN = resolveApiOrigin(
  process.env.NEXT_PUBLIC_PIXELPROOF_API_URL ?? (process.env.NODE_ENV === "development" ? "http://127.0.0.1:8800" : undefined),
  process.env.NODE_ENV === "development",
);

const REQUEST_TIMEOUT_MS = 100_000;

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef<string | null>(null);
  const requestGateRef = useRef(new LatestRequestGate());
  const [preview, setPreview] = useState<Preview | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<DemoAnalysis | null>(null);
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
    cancelAnalysis();
    setAnalysis(null);
    const fileError = demoFileError(file);
    if (fileError) {
      releasePreview();
      setPreview(null);
      setError(fileError);
      return;
    }
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
    setAnalysis(null);
    const timer = window.setTimeout(() => {
      if (requestGateRef.current.isCurrent(ticket.id)) {
        requestGateRef.current.cancel();
        setLoading(false);
        setError("İnceleme çok uzun sürdü. Biraz bekleyip yeniden deneyin.");
      }
    }, REQUEST_TIMEOUT_MS);
    try {
      const response = await fetch(`${API_ORIGIN}/analyze`, {
        method: "POST",
        headers: { "Content-Type": selectedPreview.file.type },
        body: selectedPreview.file,
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
      const parsed = parseDemoAnalysis(payload);
      if (requestGateRef.current.isCurrent(ticket.id)) setAnalysis(parsed);
    } catch (caught) {
      if (ticket.signal.aborted || !requestGateRef.current.isCurrent(ticket.id)) return;
      setError(analysisErrorMessage(caught));
    } finally {
      window.clearTimeout(timer);
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
          <span className="intro-kicker">Staj projesi · E92</span>
          <h1>Bu görselde yapay zekâ izleri var mı?</h1>
          <p>
            Bir fotoğraf seçin. Modelimiz görüntüyü incelesin; bulduğu işaretleri ve karar
            veremediği durumları sade bir dille anlatsın.
          </p>
          <div className="intro-pills" aria-label="Demo özellikleri">
            <span>Güncel model</span><span>Ek tutarlılık kontrolü</span><span>Karar veremeyebilir</span>
          </div>
        </section>

        <section className="grid">
          <div className="panel">
            <div className="panel-title">
              <h2>Görsel</h2>
              <p>JPG, PNG veya WEBP · en fazla 12 MB · 16 megapiksel</p>
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
              <h2>İnceleme sonucu</h2>
              <p>E92 · ek kontrol ile</p>
            </div>

            {!analysis ? (
              <div className="result empty-result" aria-live="polite" aria-busy={loading}>
                <div className="result-icon" aria-hidden="true">◎</div>
                <h3>{loading ? "Fotoğraf inceleniyor" : preview ? "Analize hazır" : "Önce bir görsel yükleyin"}</h3>
                <p>
                  {loading ? "Görüntü ve sıkıştırılmış bir kopyası karşılaştırılıyor. Bu işlem biraz sürebilir." : preview
                    ? "Hazır olduğunuzda incelemeyi başlatın."
                    : "Seçtiğiniz fotoğrafın sonucu burada görünecek."}
                </p>
                {error && <p className="error-text" role="alert">{error}</p>}
                <button type="button" disabled={!preview || loading} onClick={() => analyze()}>
                  {loading ? "İnceleniyor…" : "Görseli analiz et"}
                </button>
              </div>
            ) : (
              <div className="result result-stack" aria-live="polite" aria-busy={loading}>
                <section className={`demo-verdict demo-${analysis.outcome}`}>
                  <span className="section-kicker">İnceleme tamamlandı</span>
                  <h3>{OUTCOME_COPY[analysis.outcome].title}</h3>
                  <p>{analysis.reason === "image_too_small"
                    ? "Fotoğraf yeterli ayrıntı taşımıyor. Her iki kenarı da en az 224 piksel olan asıl dosyayı deneyin."
                    : OUTCOME_COPY[analysis.outcome].text}</p>
                  {analysis.outcome === "ai_signal" && analysis.review_required &&
                    <p className="review-warning"><strong>Ek kontrolde sonuç değişti.</strong> Modelin ilk
                      işaretini gösteriyoruz, ancak bu fotoğraf için sonuç tutarlı değil. Asıl dosyayı
                      ve kaynağını kontrol etmeden bir yargıya varmayın.</p>}
                  <p className="demo-next">{OUTCOME_COPY[analysis.outcome].next}</p>
                </section>
                <details className="technical-details">
                  <summary>Bu sonuç ne anlama geliyor?</summary>
                  <p>Bu staj projesi, görselin bütünündeki üretim izlerini araştırıyor.
                    Küçük bir bölgenin değiştirilip değiştirilmediğini veya nerede değiştirildiğini henüz göstermez.</p>
                  <p>Ek kontrol, sıkıştırma sonrası sonucu yeniden inceler. İlk incelemede AI
                    işareti bulunduysa gösterilir; sonuç değişirse ayrıca uyarı eklenir. Belirsiz negatif
                    sonuçlarda karar verilmez. İki kontrolde de aynı hatanın yapılması mümkündür. Bu araç bir gerçeklik sertifikası değildir.</p>
                  <p>Model sürümü: {analysis.model_id}. Görsel boyutu: {analysis.width} × {analysis.height}.</p>
                </details>
                {error && <p className="error-text" role="alert">{error}</p>}
                <button type="button" onClick={() => analyze()} disabled={loading}>
                  {loading ? "İnceleniyor…" : "Yeniden analiz et"}
                </button>
              </div>
            )}
          </div>
        </section>

        <aside>
          <strong>Nasıl kullanmalı?</strong> Bu bir öğrenci projesi; model yanılabilir.
          Bir fotoğrafı ya da kişiyi yalnız bu sonuca dayanarak değerlendirmeyin.
          <details className="privacy-note"><summary>Fotoğrafıma ne oluyor?</summary>
            <p>İncelemeyi başlattığınızda dosya analiz servisine gönderilir. Bu yerel kurulumda
              servis bu bilgisayarda çalışır. Görseller arşivlenmez, eğitime eklenmez ve başka
              bir yapay zekâ servisine gönderilmez; geçici olarak bellekte işlenir.</p>
          </details>
        </aside>
      </div>
    </main>
  );
}
