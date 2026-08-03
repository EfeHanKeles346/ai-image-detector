"use client";

import { ChangeEvent, DragEvent, useRef, useState } from "react";

type Preview = { name: string; url: string; size: string; file: File };
type Verdict = "ai" | "real" | "uncertain";
type Tile = { x: number; y: number; p_ai: number; texture: number };
type TileMap = { p_ai: number; tiles: Tile[]; tile_px: number; image_w: number; image_h: number };
type Analysis = {
  p_ai: number;
  verdict: Verdict;
  method: string;
  method_label: string;
  auto_selected: boolean;
  engine: string;
  resolution: string;
  enough_evidence: boolean;
  tile_map: TileMap | null;
};

const API_URL = "http://127.0.0.1:8799";

// Four methods, never blended — E9 showed a fixed blend adds +0.002 (noise).
// "auto" applies the measured 700px crossover from E11.
const METHODS = [
  { id: "auto", label: "Otomatik", hint: "Boyuta göre en güçlü yöntem" },
  { id: "cnn", label: "CNN", hint: "Küçültülmüş görsel · küçük girdilerde güçlü" },
  { id: "stats", label: "İst. v1", hint: "68 istatistik · 9.9k GenImage ile eğitildi" },
  { id: "stats2", label: "İst. v2", hint: "Aynı özellikler · 101k dengeli havuz ile eğitildi" },
  { id: "stats3", label: "İst. v3", hint: "Aynı özellikler · 74k havuz, sadece 256px+ görseller" },
  { id: "tiles", label: "Kare kare", hint: "6×6 orijinal kesit · ≥700px'te en iyi" },
];

const VERDICT_TEXT: Record<Verdict, { icon: string; title: string; detail: string }> = {
  ai: { icon: "🤖", title: "Yapay zekâ üretimi görünüyor", detail: "Model bu görselin AI tarafından üretildiğini düşünüyor." },
  real: { icon: "📷", title: "Gerçek fotoğraf görünüyor", detail: "Model bu görselin gerçek bir fotoğraf olduğunu düşünüyor." },
  uncertain: { icon: "🤔", title: "Emin değilim", detail: "Tahmin kararsızlık bandında — bu aralıkta hataların yoğunlaştığını ölçtük, dürüst cevap 'bilmiyorum'." },
};

const VERDICT_COLOR: Record<Verdict, string> = { ai: "#d92d20", real: "#12b76a", uncertain: "#f79009" };

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [method, setMethod] = useState("auto");
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);

  function choose(file?: File) {
    if (!file || !file.type.startsWith("image/")) return;
    if (preview) URL.revokeObjectURL(preview.url);
    setAnalysis(null);
    setError(null);
    setPreview({ name: file.name, url: URL.createObjectURL(file), size: `${(file.size / 1024 / 1024).toFixed(2)} MB`, file });
  }

  function clear() {
    if (preview) URL.revokeObjectURL(preview.url);
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
    setLoading(true);
    setError(null);
    try {
      const body = new FormData();
      body.append("image", preview.file);
      body.append("method", useMethod);
      const response = await fetch(`${API_URL}/predict`, { method: "POST", body });
      if (!response.ok) throw new Error(`${response.status}`);
      setAnalysis(await response.json());
    } catch {
      setError("Analiz servisine ulaşılamadı. Model servisi çalışıyor mu? (port 8799)");
    } finally {
      setLoading(false);
    }
  }

  function pick(id: string) {
    setMethod(id);
    if (analysis) analyze(id);
  }

  // A statistical model can never honestly claim 0% or 100%; clamp the display.
  const percent = analysis ? Math.min(99.9, Math.max(0.1, analysis.p_ai * 100)) : 0;

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
                onClick={() => inputRef.current?.click()}
                onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
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
                <div className="preview-stage">
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
                    Kırmızı yoğunluk = o karenin AI olasılığı. Tüm kareler kırmızıysa görselin
                    tamamı üretilmiş; sadece bir bölge kırmızıysa orası şüpheli.
                  </p>
                )}
              </div>
            )}
            <input ref={inputRef} hidden type="file" accept="image/jpeg,image/png,image/webp" onChange={(e: ChangeEvent<HTMLInputElement>) => choose(e.target.files?.[0])} />
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
                  title={m.hint}
                >
                  {m.label}
                </button>
              ))}
            </div>
            <p className="method-hint">{METHODS.find((m) => m.id === method)?.hint}</p>

            {!analysis ? (
              <div className="result">
                <div className="result-icon">?</div>
                <h3>{preview ? "Görsel analize hazır" : "Henüz sonuç yok"}</h3>
                <p>{preview ? "Analiz butonuna basarak modeli çalıştırın." : "Analiz için önce bir görsel yükleyin."}</p>
                {error && <p className="error-text">{error}</p>}
                <button type="button" disabled={!preview || loading} onClick={() => analyze()}>
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
                    <strong>p(AI) = %{percent.toFixed(1)}</strong>
                    <span>AI</span>
                  </div>
                  <div className="probability-track">
                    <div className="probability-fill" style={{ width: `${percent}%`, background: VERDICT_COLOR[analysis.verdict] }} />
                  </div>
                </div>
                <p className="model-info">
                  {analysis.method_label}
                  {analysis.auto_selected && " · otomatik seçildi"}
                  {" · "}{analysis.resolution}
                  {!analysis.enough_evidence && " · ⚠️ ölçüm için çok küçük"}
                </p>
                <button type="button" onClick={() => analyze()} disabled={loading}>
                  {loading ? "Analiz ediliyor…" : "Tekrar analiz et"}
                </button>
              </div>
            )}
          </div>
        </section>

        <aside>
          <strong>Yöntemler neden ayrı:</strong> Dördünü tek bir skora karıştırmayı ölçtük ve
          kazanç çıkmadı (+0.002, gürültü seviyesi) — çünkü her yöntem farklı girdide güçlü.
          <em>Otomatik</em> bu yüzden karıştırmaz, ölçülmüş kurala göre birini seçer: 700
          pikselin üstünde kare-kare yöntemi CNN&apos;i belirgin şekilde geçiyor, altında CNN
          kazanıyor. Bu bir araştırma demosudur; yeni nesil üreteçlerde hata payı vardır.
        </aside>
      </div>
    </main>
  );
}
