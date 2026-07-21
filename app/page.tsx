"use client";

import { ChangeEvent, DragEvent, useRef, useState } from "react";

type Preview = { name: string; url: string; size: string; file: File };
type Analysis = { p_ai: number; verdict: "ai" | "real" | "uncertain"; model: string; resolution: string };

const API_URL = "http://127.0.0.1:8799";

const VERDICT_TEXT: Record<Analysis["verdict"], { icon: string; title: string; detail: string }> = {
  ai: { icon: "🤖", title: "Yapay zekâ üretimi görünüyor", detail: "Model bu görselin AI tarafından üretildiğini düşünüyor." },
  real: { icon: "📷", title: "Gerçek fotoğraf görünüyor", detail: "Model bu görselin gerçek bir fotoğraf olduğunu düşünüyor." },
  uncertain: { icon: "🤔", title: "Emin değilim", detail: "Tahmin kararsızlık bandında (0.4–0.6) — bu aralıkta model hatalarının yoğunlaştığını ölçtük, dürüst cevap 'bilmiyorum'." },
};

const MODEL_TEXT: Record<string, string> = {
  small_cnn_cifake: "SmallCNN (düşük çözünürlük uzmanı)",
  resnet18_genimage: "ResNet-18 (yüksek çözünürlük uzmanı)",
};

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);

  function choose(file?: File) {
    if (!file || !file.type.startsWith("image/")) return;
    if (preview) URL.revokeObjectURL(preview.url);
    setAnalysis(null);
    setError(null);
    setPreview({
      name: file.name,
      url: URL.createObjectURL(file),
      size: `${(file.size / 1024 / 1024).toFixed(2)} MB`,
      file,
    });
  }

  function clear() {
    if (preview) URL.revokeObjectURL(preview.url);
    setPreview(null);
    setAnalysis(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  function onChange(event: ChangeEvent<HTMLInputElement>) {
    choose(event.target.files?.[0]);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    choose(event.dataTransfer.files?.[0]);
  }

  async function analyze() {
    if (!preview) return;
    setLoading(true);
    setError(null);
    try {
      const body = new FormData();
      body.append("image", preview.file);
      const response = await fetch(`${API_URL}/predict`, { method: "POST", body });
      if (!response.ok) throw new Error(`Sunucu hatası: ${response.status}`);
      setAnalysis(await response.json());
    } catch {
      setError("Analiz servisine ulaşılamadı. Model servisi çalışıyor mu? (port 8799)");
    } finally {
      setLoading(false);
    }
  }

  const percent = analysis ? Math.round(analysis.p_ai * 100) : 0;

  return (
    <main>
      <header>
        <div className="header-content">
          <strong>AI Image Detector</strong>
          <span>SmallCNN + ResNet-18</span>
        </div>
      </header>

      <div className="container">
        <section className="intro">
          <h1>Görsel Gerçeklik Analizi</h1>
          <p>Bir fotoğraf yükleyerek gerçek veya yapay zekâ üretimi olma ihtimalini inceleyin.</p>
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
                onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => inputRef.current?.click()}
                onKeyDown={(event) => (event.key === "Enter" || event.key === " ") && inputRef.current?.click()}
                role="button"
                tabIndex={0}
              >
                <div className="upload-symbol">↑</div>
                <strong>Fotoğrafı buraya bırakın</strong>
                <span>veya bilgisayarınızdan seçin</span>
                <button type="button">Dosya seç</button>
              </div>
            ) : (
              <div className="preview">
                <img src={preview.url} alt="Seçilen görsel" />
                <div className="file-info">
                  <div><strong>{preview.name}</strong><span>{preview.size}</span></div>
                  <button type="button" onClick={clear}>Kaldır</button>
                </div>
              </div>
            )}
            <input ref={inputRef} hidden type="file" accept="image/jpeg,image/png,image/webp" onChange={onChange} />
          </div>

          <div className="panel">
            <div className="panel-title">
              <h2>2. Analiz sonucu</h2>
              <p>Model tahmini ve güven oranı</p>
            </div>

            {!analysis ? (
              <div className="result">
                <div className="result-icon">?</div>
                <h3>{preview ? "Görsel analize hazır" : "Henüz sonuç yok"}</h3>
                <p>{preview ? "Analiz butonuna basarak modeli çalıştırın." : "Analiz için önce bir görsel yükleyin."}</p>
                {error && <p className="error-text">{error}</p>}
                <button type="button" disabled={!preview || loading} onClick={analyze}>
                  {loading ? "Analiz ediliyor…" : "Analiz et"}
                </button>
              </div>
            ) : (
              <div className="result">
                <div className={`result-icon verdict-${analysis.verdict}`}>{VERDICT_TEXT[analysis.verdict].icon}</div>
                <h3>{VERDICT_TEXT[analysis.verdict].title}</h3>
                <p>{VERDICT_TEXT[analysis.verdict].detail}</p>
                <div className="probability">
                  <div className="probability-labels">
                    <span>Gerçek</span>
                    <strong>p(AI) = %{percent}</strong>
                    <span>AI</span>
                  </div>
                  <div className="probability-track">
                    <div className={`probability-fill verdict-${analysis.verdict}`} style={{ width: `${percent}%` }} />
                  </div>
                </div>
                <p className="model-info">
                  {MODEL_TEXT[analysis.model] ?? analysis.model} · {analysis.resolution}
                </p>
                <button type="button" onClick={analyze} disabled={loading}>
                  {loading ? "Analiz ediliyor…" : "Tekrar analiz et"}
                </button>
              </div>
            )}
          </div>
        </section>

        <aside>
          <strong>Nasıl çalışır:</strong> Görsel, çözünürlüğüne göre iki uzman modelden birine yönlendirilir
          (küçük görseller → CIFAKE ile eğitilmiş SmallCNN, büyükler → GenImage ile eğitilmiş ResNet-18).
          Kararsızlık bandındaki tahminler dürüstçe &quot;emin değilim&quot; olarak raporlanır.
          Bu bir araştırma demosudur; özellikle yeni nesil üreteçlerde hata payı vardır.
        </aside>
      </div>
    </main>
  );
}
