# PixelProof — staj sunumu kanıt paketi

Bu dosya sunum ve staj raporu için güncel, izlenebilir özet kaynaktır. `rapor/` klasörü E26 sonu
tarihsel teslim anını korur; M1–M6 ile gelen çalışabilir proje-modeli kilometre taşı burada eklenir.

## Tek cümlelik sonuç

Projede eğitilmiş, hash ile doğrulanan bir E20 ResNet-18 artık CLI, klasör değerlendirici, API ve
web arayüzü üzerinden gerçekten çalışıyor; fakat üç-seed ölçümündeki %86,2 ± %3,1 worst-source
gerçek yanlış alarmı nedeniyle yalnız araştırma sinyali olarak sunuluyor.

## Sunumda kullanılacak ana tablo

| Soru | Kanıtlanmış cevap | Birincil kaynak |
|---|---|---|
| Çalışan kendi modelimiz var mı? | Evet; E20 seed-2024 checkpointi dört ortak inference yüzeyinde çalışıyor | `MODEL_CARD.md`, M1–M5 commitleri |
| Artifact gerçekten aynı mı? | SHA-256 `b9f39eda...65adf` yüklemeden önce doğrulanıyor | `ml/artifacts.manifest.json`, M1 |
| Başlıca performans nedir? | AUC 0,751 ± 0,033; recall %49,9 ± %6,1 | `ml/EXPERIMENTS.md`, E20 üç-seed addendum |
| Güvenlik sınırı nedir? | Worst-source gerçek FP %86,2 ± %3,1; negatif sonuç “gerçek” değildir | `MODEL_CARD.md`, E20 üç-seed addendum |
| Kullanıcı kendi verisini ölçebilir mi? | Evet; JSON+CSV, hata satırları ve tam provenance ile | `pixelproof-evaluate-project`, M4 |
| Demo tek komut mu? | Evet; preflight, gerçek smoke, API ve web birlikte | `./tools/pixelproof-demo start`, M5 |

## M0–M5 geliştirme ve commit defteri

| Aşama | Commit | Teslim edilen davranış | Kapanış doğrulaması |
|---|---|---|---|
| M0 | `774520b` | Çalışabilir proje-modeli hedefi uygulamadan önce planlandı | M1–M6 sırası ve kabul sınırları |
| Arşiv politikası | `71d6435` | `HISTORY.md` append-only proje arşivi oldu | H0–H6 ve M0 geriye dönük defteri |
| M1 | `fa106e2` | E20 artifact hash + checkpoint sözleşmesi | Python 29/29; gerçek CPU yükleme |
| M2 | `3f87d72` | Ortak scorer, API ve CLI `project_model` varsayılanı | Python 33/33; MPS direct/API/CLI 0,2409 |
| M3 | `f158f0e` | Model-first Türkçe web deneyimi | Web 6/6; lint/type/build |
| M4 | `590c3ae` | Etiketli klasörlerden JSON/CSV değerlendirme | Python 36/36; gerçek MPS 4/4 klasör smoke |
| M5 | `95fe2b2` | Tek-komut preflight/smoke/API/web | Python 41/41 + web 6/6; canlı HTTP 200 |

Bu sayılar `PLAN.md` ve append-only `HISTORY.md` içinde aşama bazında tekrar kayıtlıdır. Commitler
sunumda “ne zaman ne çalışır hale geldi?” sorusunun değişmez zaman çizgisidir.

## Tekrarlanabilir ayrışma demosu

Kanıt dosyası: `evidence/demo_disagreement.json`.

Girdi, pinned B-Free checkoutundaki `img0000.png`; upstream `metainfo.csv` etiketi `0 = real`, girdi
SHA-256'sı `c7351aee...a79360e`. B-Free burada dedektör olarak etkin değildir; yalnız etiketli ve
revision-pinned demo girdisinin kaynağıdır. Haricî karşılaştırma kolu MIT lisanslı CF-ViT'tir.

| Katman | Skor / eşik | Ekrandaki sonuç |
|---|---:|---|
| Bizim E20 proje modelimiz | 1,0000 / 0,9895 | AI yönünde deneysel sinyal — **yanlış pozitif** |
| Haricî CF-ViT karşılaştırması | −2,4631 / 0,6617 | Yeterli kanıt yok |

Bu anlaşmazlık bir arayüz hatası değildir. İki model farklı eğitim popülasyonları ve temsiller
öğrenmiştir. E20'nin native-tile dokulara dayalı skoru bu gerçek kaynağa transfer olmazken CF-ViT
tetiklenmemiştir. Bu yüzden arayüz iki sonucu ayrı kartlarda gösterir ve hiçbiri “görsel gerçektir”
demez. Sunumda modelin çalıştığını gösterdikten hemen sonra bu örnek gösterilmelidir; güçlü tarafla
sınırı aynı anlatı içinde tutar.

Yeniden üretim için önce `ml/ARTIFACTS.md` içindeki pinned CF snapshot ve optional B-Free checkout
adımları tamamlanır, ardından API `PIXELPROOF_RUNTIME_PROFILE=full` ile çalıştırılıp aynı dosya
`project_model` yöntemiyle yüklenir. Kaydedilmiş input hash, runtime commit, model hash, tile sayısı
ve iki karar bloğu JSON içinde karşılaştırılır.

## Canlı demo sırası

1. Repo kökünde `./tools/pixelproof-demo start` çalıştır; preflight ve 0,2409 smoke çıktısını göster.
2. Tarayıcıda bir JPG/PNG/WEBP yükle; birincil kartın “E20 proje modeli” olduğunu belirt.
3. Ham skor, deneysel eşik, model revision, tile sayısı ve hash prefixini göster.
4. Negatif sonucun neden “gerçek” değil “eşik aşılmadı” olduğunu açıkla.
5. Ayrışma JSON/table örneğiyle %86,2 worst-source FP sınırını göster.
6. Kendi etiketli veri klasörlerin varsa `pixelproof-evaluate-project` çıktısındaki confusion ve
   failure sayılarını göster; başarısız decode satırlarının kaybolmadığını vurgula.

## İzlenebilir kaynak haritası

| İddia türü | Değişmez/append-only kaynak | Yerel ham kaynak |
|---|---|---|
| Deney sonucu | `ml/EXPERIMENTS.md` | `ml/artifacts/e20/results_3seed.json` |
| Artifact kimliği | `ml/artifacts.manifest.json` | owner-supplied checkpoint |
| Uygulama sırası ve test | `PLAN.md`, `HISTORY.md` | Git commitleri |
| Modelin izin verilen kullanımı | `MODEL_CARD.md`, `LICENSE.md` | — |
| Demo ayrışması | `evidence/demo_disagreement.json` | pinned B-Free demo input |
| Çalıştırma sözleşmesi | `README.md`, `ml/SERVING.md` | `tools/pixelproof-demo` |

Rapor veya slaytta yeni bir sayı kullanılmadan önce bu tabloda bir kaynağı olmalıdır. Kaynağı
olmayan sayı sonuç değil, nottur.
