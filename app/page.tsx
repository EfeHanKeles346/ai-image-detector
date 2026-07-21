"use client";

import { ChangeEvent, DragEvent, useRef, useState } from "react";

type Preview = { name: string; url: string; size: string };

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [dragging, setDragging] = useState(false);

  function choose(file?: File) {
    if (!file || !file.type.startsWith("image/")) return;
    if (preview) URL.revokeObjectURL(preview.url);
    setPreview({
      name: file.name,
      url: URL.createObjectURL(file),
      size: `${(file.size / 1024 / 1024).toFixed(2)} MB`,
    });
  }

  function clear() {
    if (preview) URL.revokeObjectURL(preview.url);
    setPreview(null);
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

  return (
    <main>
      <header>
        <div className="header-content">
          <strong>AI Image Detector</strong>
          <span>CNN Baseline</span>
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
            <div className="result">
              <div className="result-icon">?</div>
              <h3>{preview ? "Görsel analize hazır" : "Henüz sonuç yok"}</h3>
              <p>{preview ? "Eğitilmiş model bağlandıktan sonra analiz başlatılabilecek." : "Analiz için önce bir görsel yükleyin."}</p>
              <button type="button" disabled>Model henüz eğitilmedi</button>
            </div>
          </div>
        </section>

        <aside>
          <strong>Proje durumu:</strong> Arayüz ve CNN eğitim altyapısı hazır. Bir sonraki aşama veri setiyle modeli eğitmek ve sonuçları bu ekrana bağlamak.
        </aside>
      </div>
    </main>
  );
}
